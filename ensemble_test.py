"""
SignBridge - ensemble test: does combining the RandomForest and the GRU beat
either one alone?

WHY
---
Measured on the same val split, the two models land close in aggregate but
fail on almost disjoint sets of words:

    WHAT        RF f1 0.18   GRU f1 0.93      <- RF's worst class, GRU perfect
    NAME        RF f1 0.73   GRU f1 1.00
    SICK        RF f1 0.67   GRU f1 0.86
    WATER       RF f1 0.89   GRU f1 0.29      <- and the reverse
    UNDERSTAND  RF f1 0.86   GRU f1 0.29
    BATHROOM    RF f1 0.80   GRU f1 0.25

Two models that make the SAME mistakes gain nothing from being combined.
Two models that make DIFFERENT mistakes can cover for each other, which is
the textbook condition for an ensemble to beat both parts. These two qualify,
so it's worth the twenty minutes to find out.

HOW
---
Both models output a probability per class. This blends them:

    P_ensemble = w * P_randomforest + (1 - w) * P_gru

and takes the highest. w=1.0 is the RF alone, w=0.0 is the GRU alone, w=0.5
weights them equally.

The two models read different inputs - the RF wants v2's 926 engineered
features, the GRU wants a 32-step sequence - so this script walks the .npz
files ONCE and builds both representations per clip, in one order. That
guarantees row i is the same clip for both models, which a CSV join could
silently get wrong.

Class ordering is checked rather than assumed: sklearn sorts its classes_ and
the GRU checkpoint stores its own list, and blending two probability matrices
whose columns mean different things would produce confident nonsense.

CAVEAT ON THE WEIGHT SWEEP
--------------------------
The sweep prints accuracy at every w so you can see the shape of the curve.
Picking the w that scores best on val IS tuning against val - that's what val
is for, so it's legitimate, but it means the tuned number is optimistic.
Report the 0.5 equal-weight number as the honest headline, and treat the
sweep as diagnostic. Don't then tune w on the test set.

Run:
    python ensemble_test.py
"""

import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
RF_PATH = os.path.join(BASE_DIR, "models", "random_forest_v2.joblib")
GRU_PATH = os.path.join(BASE_DIR, "models", "pytorch_gru.pt")


def align_columns(probs, from_classes, to_classes):
    """
    Reorder a probability matrix so its columns follow `to_classes`.

    Blending two matrices whose column k means a different word is the kind of
    bug that produces plausible-looking garbage, so this is explicit rather
    than trusting both sources to have sorted identically.
    """
    index = {c: i for i, c in enumerate(from_classes)}
    missing = [c for c in to_classes if c not in index]
    if missing:
        raise SystemExit(f"ERROR: model is missing classes {missing}")
    return probs[:, [index[c] for c in to_classes]]


def main():
    try:
        import torch
        import joblib
    except ImportError as e:
        raise SystemExit(f"ERROR: missing dependency ({e}). Need torch and joblib.")

    from sklearn.metrics import accuracy_score, classification_report

    from build_feature_vectors_v2 import LABEL_MERGES, build_clip_features, make_feature_names
    from train_pytorch import clip_to_sequence, build_model, PER_FRAME

    if not os.path.exists(RF_PATH):
        raise SystemExit(f"ERROR: {RF_PATH} not found. Run train_classifier.py data/features_v2.csv")
    if not os.path.exists(GRU_PATH):
        raise SystemExit(f"ERROR: {GRU_PATH} not found. Run train_pytorch.py")

    rf_bundle = joblib.load(RF_PATH)
    rf, rf_cols = rf_bundle["model"], rf_bundle["feature_columns"]

    ckpt = torch.load(GRU_PATH, map_location="cpu", weights_only=False)
    gru_classes = list(ckpt["classes"])
    n_steps = int(ckpt["timesteps"])

    import torch.nn as nn
    gru = build_model(ckpt.get("model", "gru"), n_steps, len(gru_classes),
                      int(ckpt.get("hidden", 64)), 0.0, torch, nn)
    gru.load_state_dict(ckpt["state_dict"])
    gru.eval()

    # --- one pass over the clips, building BOTH representations together ---
    feats, seqs, labels, splits = [], [], [], []
    for path in sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]
        left = left if left.shape[0] > 0 else None
        right = right if right.shape[0] > 0 else None
        if left is None and right is None:
            continue
        raw = str(d["label"])
        labels.append(LABEL_MERGES.get(raw, raw))
        splits.append(str(d["split"]))
        feats.append(build_clip_features(left, right))
        seqs.append(clip_to_sequence(left, right, n_steps))

    splits = np.array(splits)
    labels = np.array(labels)
    X_feat = np.stack(feats)
    X_seq = np.stack(seqs)

    if X_feat.shape[1] != len(rf_cols):
        raise SystemExit(f"ERROR: feature width {X_feat.shape[1]} != RF's {len(rf_cols)}. "
                         f"Rebuild with build_feature_vectors_v2.py and retrain.")

    va = splits == "val"
    y_true = labels[va]
    print(f"Evaluating on {va.sum()} val clips (same split both models trained against)\n")

    classes = sorted(set(labels))          # the common ordering both get mapped onto

    p_rf = align_columns(rf.predict_proba(X_feat[va]), list(rf.classes_), classes)
    with torch.no_grad():
        logits = gru(torch.tensor(X_seq[va]))
        p_gru = torch.softmax(logits, dim=1).numpy()
    p_gru = align_columns(p_gru, gru_classes, classes)

    cls_arr = np.array(classes)

    def acc_at(w):
        blend = w * p_rf + (1.0 - w) * p_gru
        return accuracy_score(y_true, cls_arr[blend.argmax(1)])

    rf_acc, gru_acc, eq_acc = acc_at(1.0), acc_at(0.0), acc_at(0.5)

    print(f"{'weight (w)':<12}{'blend':<28}{'val accuracy'}")
    print("-" * 56)
    best_w, best_acc = 0.5, -1.0
    for w in [i / 10 for i in range(11)]:
        a = acc_at(w)
        if a > best_acc:
            best_acc, best_w = a, w
        tag = "  <- RF alone" if w == 1.0 else ("  <- GRU alone" if w == 0.0 else
              ("  <- equal weight" if w == 0.5 else ""))
        bar = "#" * int(round(a * 40))
        print(f"  w={w:<9.1f}{bar:<28}{a:.3f}{tag}")

    print(f"\nRF alone:        {rf_acc:.3f}")
    print(f"GRU alone:       {gru_acc:.3f}")
    print(f"Equal weight:    {eq_acc:.3f}   <- report THIS one (no tuning against val)")
    print(f"Best swept w:    {best_acc:.3f} at w={best_w:.1f}   (optimistic - w was chosen on val)")

    delta = eq_acc - max(rf_acc, gru_acc)
    n = len(y_true)
    print(f"\nEqual-weight ensemble vs the better single model: {delta:+.3f} "
          f"({round(delta * n):+.0f} of {n} val clips)")
    se = (0.25 / n) ** 0.5
    print(f"For scale, 1 standard error on {n} samples is about {se:.3f} "
          f"({round(se * n):.0f} clips) - changes smaller than that are noise.")

    # --- where does the combination actually help? ---
    rf_pred = cls_arr[p_rf.argmax(1)]
    gru_pred = cls_arr[p_gru.argmax(1)]
    agree = rf_pred == gru_pred
    print(f"\nAgreement analysis:")
    print(f"  models agree on      {agree.sum():>3}/{n} clips, "
          f"and are right on {(rf_pred[agree] == y_true[agree]).sum()} of those")
    dis = ~agree
    if dis.sum():
        print(f"  models disagree on   {dis.sum():>3}/{n} clips:")
        print(f"      RF right, GRU wrong:  {((rf_pred == y_true) & dis).sum()}")
        print(f"      GRU right, RF wrong:  {((gru_pred == y_true) & dis).sum()}")
        print(f"      both wrong:           {((rf_pred != y_true) & (gru_pred != y_true) & dis).sum()}")
        print("  (the middle row is the headroom an ensemble is trying to capture)")

    blend = 0.5 * p_rf + 0.5 * p_gru
    print(f"\nPer-class report, equal-weight ensemble:")
    print(classification_report(y_true, cls_arr[blend.argmax(1)],
                                labels=classes, target_names=classes, zero_division=0))


if __name__ == "__main__":
    main()
