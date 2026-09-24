"""
SignBridge - the ONE final evaluation of the frozen ensemble on the test split.

Run this once, after build_ensemble_bundle.py, and never tune anything
afterwards: the moment a test number influences a choice, it stops being a
test number. The script enforces that. It writes its result to
sweeps/final/test_result_v3.json and refuses to run again while that file
exists (it prints the stored result instead). --force overrides, and the
override is recorded.

The test split is 416 ASL Citizen clips from 11 signers who appear nowhere in
train, dev or val. Its landmarks are extracted into their OWN folder so they
can never sit beside the training data:

    python extract_landmarks_v3.py --source citizen --splits test --out-dir data/landmarks_v3_test
    python evaluate_ensemble_on_test.py

(~7 s per clip to extract, so budget about 50 minutes.)

WHAT IT REPORTS
---------------
* ensemble accuracy with a 95% confidence interval (Wilson)
* the same, counting every test clip the extractor could not use (no hand
  detected, unreadable) as WRONG - the conservative number
* the RF alone and the 5-GRU average alone, on the same clips, for context
* abstention at the frozen threshold: how much is accepted, and how accurate
  the accepted part is
* accuracy per test signer (how much it varies from person to person)
* per-class recall with how often each class was predicted, and the most
  common confusions
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from build_feature_vectors_v2 import LABEL_MERGES
import signbridge_ensemble as SE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_DIR = os.path.join(BASE_DIR, "data", "landmarks_v3_test")
DEFAULT_OUT = os.path.join(BASE_DIR, "sweeps", "final", "test_result_v3.json")
MANIFEST = os.path.join(BASE_DIR, "data", "clips_manifest.csv")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def expected_test_count():
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if r.get("split") == "test")


def load_test_clips(test_dir):
    clips, non_test = [], 0
    for path in sorted(glob.glob(os.path.join(test_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        if str(d["split"]) != "test":
            non_test += 1
            continue
        raw = str(d["label"])
        clips.append({
            "file": os.path.basename(path),
            "left_hand": d["left_hand"], "right_hand": d["right_hand"],
            "pose": d["pose"] if "pose" in d else None,
            "frame_w": int(d["frame_w"]), "frame_h": int(d["frame_h"]),
            "label": LABEL_MERGES.get(raw, raw),
            "participant": str(d["participant"]),
        })
    return clips, non_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", default=SE.DEFAULT_BUNDLE)
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="run again even though a final result exists (recorded)")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh)
        print(f"A final test result already exists ({args.out}, {prev['created']}):")
        print(f"  ensemble {prev['ensemble']['correct']}/{prev['n_clips']} = "
              f"{prev['ensemble']['accuracy']:.3f} "
              f"(95% CI {prev['ensemble']['ci95'][0]:.3f}-{prev['ensemble']['ci95'][1]:.3f})")
        print("The test split is spent. Not running again (use --force only if "
              "something was genuinely broken, and say so when you report it).")
        return

    ens = SE.SignBridgeEnsemble.load(args.bundle)
    if ens.bundle["provenance"].get("smoke"):
        raise SystemExit("ERROR: that is a --smoke bundle. Build the real one first.")
    print("=" * 76)
    print("FINAL TEST EVALUATION - frozen SignBridge ensemble v3")
    print("=" * 76)
    print(ens.describe())

    clips, non_test = load_test_clips(args.test_dir)
    expected = expected_test_count()
    if not clips:
        raise SystemExit(f"ERROR: no test-split .npz files in {args.test_dir}. Run:\n"
                         f"  python extract_landmarks_v3.py --source citizen --splits test "
                         f"--out-dir data/landmarks_v3_test")
    if non_test:
        print(f"  note: {non_test} non-test files in {args.test_dir} were ignored")
    n = len(clips)
    unusable = (expected - n) if expected else 0
    print(f"\ntest clips: {n} with landmarks"
          + (f" of {expected} in the manifest ({unusable} unusable: no hand / unreadable)"
             if expected else ""))
    if expected and n < 0.95 * expected:
        raise SystemExit(f"ERROR: only {n}/{expected} test clips have landmarks. The "
                         f"extraction looks unfinished (it is resumable - run it again). "
                         f"Not spending the test split on a partial set.")
    unknown = sorted({c["label"] for c in clips} - set(ens.classes))
    if unknown:
        raise SystemExit(f"ERROR: test labels not in the model: {unknown}")

    idx = {c: i for i, c in enumerate(ens.classes)}
    y = np.array([idx[c["label"]] for c in clips])
    inputs = [SE.clip_inputs(c["left_hand"], c["right_hand"], c["pose"],
                             c["frame_w"], c["frame_h"], ens.gru_cfg) for c in clips]
    rf_X = np.stack([a for a, _ in inputs]).astype(np.float64)
    gru_X = np.stack([b for _, b in inputs]).astype(np.float32)

    P_rf, P_gru = ens.component_probs(rf_X, gru_X)
    P = ens.blend(P_rf, P_gru)
    pred = P.argmax(1)
    ok = pred == y
    k = int(ok.sum())
    lo, hi = wilson(k, n)
    rf_ok = P_rf.argmax(1) == y
    gru_avg = np.mean([SE.temper(G, t) for G, t in zip(P_gru, ens.gru_temps)], axis=0)
    gru_ok = gru_avg.argmax(1) == y

    print("\n" + "=" * 76)
    print(f"  ENSEMBLE   {k}/{n} = {k / n:.3f}   95% CI {lo:.3f}-{hi:.3f}")
    if expected and unusable > 0:
        print(f"  counting the {unusable} unusable clips as wrong: "
              f"{k}/{expected} = {k / expected:.3f}")
    print(f"  RF alone   {int(rf_ok.sum())}/{n} = {rf_ok.mean():.3f}")
    print(f"  GRU x5     {int(gru_ok.sum())}/{n} = {gru_ok.mean():.3f}")
    print("=" * 76)

    thr = ens.threshold
    abst = None
    if thr is not None:
        conf = P.max(1)
        acc_mask = conf >= thr
        cov = float(acc_mask.mean())
        acc_acc = float(ok[acc_mask].mean()) if acc_mask.any() else float("nan")
        wrong_acc = int((acc_mask & ~ok).sum())
        print(f"\nABSTENTION at the dev-chosen threshold {thr:.3f}:")
        print(f"  accepted {int(acc_mask.sum())}/{n} ({cov:.0%}) at {acc_acc:.1%} accuracy "
              f"- {wrong_acc} wrong answers acted on")
        print(f"  asked to confirm: {int((~acc_mask).sum())} "
              f"({int((~acc_mask & ok).sum())} of those were right anyway)")
        abst = {"threshold": thr, "coverage": cov, "accepted_accuracy": acc_acc,
                "accepted_wrong": wrong_acc, "confirm": int((~acc_mask).sum())}

    by_signer = defaultdict(list)
    for c, o in zip(clips, ok):
        by_signer[c["participant"]].append(bool(o))
    print(f"\nPER SIGNER ({len(by_signer)} people never seen in training):")
    signer_rows = []
    for p in sorted(by_signer, key=lambda s: np.mean(by_signer[s])):
        v = by_signer[p]
        signer_rows.append((p, len(v), float(np.mean(v))))
        print(f"  {p:<10} {sum(v):>3}/{len(v):<3} {np.mean(v):.3f}")

    C = len(ens.classes)
    support = np.bincount(y, minlength=C)
    predicted = np.bincount(pred, minlength=C)
    recall = np.array([ok[y == c].mean() if support[c] else np.nan for c in range(C)])
    print("\nPER CLASS (worst 10 by recall; 'pred' = times predicted)")
    for c in [c for c in np.argsort(np.nan_to_num(recall, nan=2.0)) if support[c]][:10]:
        print(f"  {ens.classes[c]:<12} n={support[c]:>3}  recall {recall[c]:.2f}  "
              f"pred {predicted[c]:>3}")
    conf_pairs = Counter((ens.classes[t], ens.classes[p]) for t, p in zip(y, pred) if t != p)
    print("\nMOST COMMON CONFUSIONS (true -> predicted)")
    for (t, p), cnt in conf_pairs.most_common(8):
        print(f"  {t:>12} -> {p:<12} {cnt}")

    result = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "forced_rerun": bool(args.force and os.path.exists(args.out)),
        "bundle": os.path.relpath(args.bundle, BASE_DIR),
        "bundle_sha256": SE.file_sha256(args.bundle),
        "n_clips": n, "n_manifest": expected, "n_unusable": unusable,
        "ensemble": {"correct": k, "accuracy": k / n, "ci95": [lo, hi],
                     "accuracy_counting_unusable_as_wrong":
                         (k / expected) if expected else None},
        "rf_alone": float(rf_ok.mean()), "gru_x5_alone": float(gru_ok.mean()),
        "abstention": abst,
        "per_signer": [{"signer": p, "n": m, "accuracy": a} for p, m, a in signer_rows],
        "per_class": {ens.classes[c]: {"n": int(support[c]), "recall": float(recall[c]),
                                       "predicted": int(predicted[c])} for c in range(C)},
        "confusions": [{"true": t, "pred": p, "count": cnt}
                       for (t, p), cnt in conf_pairs.most_common()],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWritten to {args.out}. This is the final number - don't tune on it.")


if __name__ == "__main__":
    main()
