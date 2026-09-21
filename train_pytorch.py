"""
SignBridge - PyTorch sequence model, as a comparison against the RandomForest.

WHAT THIS TESTS
---------------
The RF trains on v2's hand-engineered summary of each clip: 10 evenly-spaced
keyframes plus path length and peak speed per landmark (926 numbers). Those
features were designed by hand after v1's summary statistics were measured to
destroy oscillating motion (see the feature-representation doc).

This asks a different question: instead of hand-picking what to measure about
the trajectory, feed the trajectory itself to a model that learns temporal
structure on its own. Same clips, same splits, same normalized landmarks -
only the representation and the classifier change:

    RF   : 10 keyframes  -> engineered features -> trees
    GRU  : 32 timesteps  -> learned recurrence  -> linear head

If the GRU wins, the hand-engineered features were leaving temporal
information on the table. If it doesn't, that's a legitimate result too, and
with ~964 training rows it is the more likely outcome - neural nets are
generally hungrier for data than trees. Either way it makes the report's
"we evaluated classical ML against deep learning" claim real rather than
asserted.

  --model mlp  runs the same pipeline through a plain feed-forward net on the
  flattened sequence. That's the control: if the GRU only matches the MLP,
  the recurrence isn't earning its keep and the gain (if any) is just "it's a
  neural net", not "it models time".

INPUT
-----
Reads data/landmarks/*.npz directly - the same per-frame normalized
sequences the feature builder reads, NOT features_v2.csv. Splits come from
each .npz's own `split` field, so train/val/test stay exactly as they are for
the RF (signer-disjoint), and the test split is never touched here.

The two hands are stored as separate sequences, each covering only the frames
that hand was detected in, so they are NOT frame-aligned with each other.
Each is resampled independently to the same T timesteps and then concatenated,
which is the same trick v2 uses for its keyframes - just at higher temporal
resolution (32 vs 10).

SETUP
    pip install torch

RUN
    python train_pytorch.py                  # GRU, 32 timesteps
    python train_pytorch.py --model mlp      # feed-forward control
    python train_pytorch.py --timesteps 48   # more temporal resolution

Compare the val accuracy it prints against the RF's on the same data.
"""

import argparse
import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
MODEL_DIR = os.path.join(BASE_DIR, "models")

N_LANDMARKS = 21
N_AXES = 3
PER_HAND = N_LANDMARKS * N_AXES          # 63
PER_FRAME = PER_HAND * 2 + 2             # both hands + two presence flags = 128

# Imported so the class list can't drift from the RF's. Same merge, same 27
# classes, so the two models' numbers are comparable.
try:
    from build_feature_vectors_v2 import LABEL_MERGES
except Exception:
    LABEL_MERGES = {"HURT": "HURT_PAIN", "PAIN": "HURT_PAIN"}


# ---------------------------------------------------------------------------
# Data pipeline (pure numpy - no torch, so it can be tested on its own)
# ---------------------------------------------------------------------------

def resample(seq, n_steps):
    """(n_frames, 21, 3) -> (n_steps, 21, 3) by linear interpolation.

    Same approach as build_feature_vectors_v2.resample_to_keyframes, so a
    clip's trajectory is represented the same way here as it is for the RF,
    just sampled more finely.
    """
    n_frames = seq.shape[0]
    if n_frames == 0:
        return np.zeros((n_steps, N_LANDMARKS, N_AXES), dtype=np.float32)
    if n_frames == 1:
        return np.repeat(seq, n_steps, axis=0).astype(np.float32)

    old_idx = np.linspace(0, n_frames - 1, num=n_frames)
    new_idx = np.linspace(0, n_frames - 1, num=n_steps)
    out = np.empty((n_steps, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for lm in range(seq.shape[1]):
        for ax in range(seq.shape[2]):
            out[:, lm, ax] = np.interp(new_idx, old_idx, seq[:, lm, ax])
    return out


def clip_to_sequence(left, right, n_steps):
    """
    Build one clip's (n_steps, PER_FRAME) array.

    Layout per timestep: [left 63][right 63][left_present][right_present]
    - mirrors the RF feature vector's [left block, left_present, right block,
    right_present] ordering closely enough that the two models are looking at
    the same information, just shaped differently.
    """
    l_present = 1.0 if (left is not None and left.shape[0] > 0) else 0.0
    r_present = 1.0 if (right is not None and right.shape[0] > 0) else 0.0

    l = resample(left if l_present else np.zeros((0, N_LANDMARKS, N_AXES)), n_steps)
    r = resample(right if r_present else np.zeros((0, N_LANDMARKS, N_AXES)), n_steps)

    seq = np.concatenate([
        l.reshape(n_steps, PER_HAND),
        r.reshape(n_steps, PER_HAND),
        np.full((n_steps, 1), l_present, dtype=np.float32),
        np.full((n_steps, 1), r_present, dtype=np.float32),
    ], axis=1)
    return seq.astype(np.float32)


def load_dataset(n_steps, landmarks_dir=LANDMARKS_DIR):
    """
    Returns X (N, n_steps, PER_FRAME), y (N,) int labels, splits (N,) str,
    and the sorted class list.
    """
    files = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
    if not files:
        raise SystemExit(f"ERROR: no .npz files in {landmarks_dir}. "
                         f"Run extract_clip_landmarks.py first.")

    seqs, labels, splits = [], [], []
    for path in files:
        d = np.load(path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]
        left = left if left.shape[0] > 0 else None
        right = right if right.shape[0] > 0 else None
        if left is None and right is None:
            continue                      # no hand ever detected - nothing to learn from

        raw_label = str(d["label"])
        labels.append(LABEL_MERGES.get(raw_label, raw_label))
        splits.append(str(d["split"]))
        seqs.append(clip_to_sequence(left, right, n_steps))

    classes = sorted(set(labels))
    idx = {c: i for i, c in enumerate(classes)}
    X = np.stack(seqs)
    y = np.array([idx[l] for l in labels], dtype=np.int64)
    return X, y, np.array(splits), classes


# ---------------------------------------------------------------------------
# Models + training (torch)
# ---------------------------------------------------------------------------

def build_model(kind, n_steps, n_classes, hidden, dropout, torch, nn):
    if kind == "gru":
        class GRUClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(PER_FRAME, hidden, num_layers=1, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden, n_classes)

            def forward(self, x):
                _, h = self.gru(x)          # h: (1, batch, hidden)
                return self.fc(self.drop(h[-1]))
        return GRUClassifier()

    class MLPClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_steps * PER_FRAME, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            return self.net(x)
    return MLPClassifier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "mlp"], default="gru")
    ap.add_argument("--timesteps", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("ERROR: PyTorch not installed. Run:  pip install torch")

    from sklearn.metrics import accuracy_score, classification_report

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, splits, classes = load_dataset(args.timesteps)
    print(f"Loaded {len(X)} clips, {len(classes)} classes, "
          f"sequences of {args.timesteps} steps x {PER_FRAME} features")

    tr, va = splits == "train", splits == "val"
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    print(f"train: {len(Xtr)} rows, val: {len(Xva)} rows  "
          f"(test split untouched, as with the RF)")

    # Same intent as the RF's class_weight='balanced': merged-variant words
    # (EAT, DRINK, HOW, WHAT, DEAF) carry ~2x the clips of everything else.
    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(len(y[tr]) / (len(classes) * np.maximum(counts, 1)))

    model = build_model(args.model, args.timesteps, len(classes),
                        args.hidden, args.dropout, torch, nn)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model.upper()}, {n_params:,} parameters\n")

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc, best_state, since_best = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva).argmax(1)
            va_acc = (va_pred == yva).float().mean().item()

        if va_acc > best_acc:
            best_acc, since_best = va_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1

        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
            print(f"  epoch {epoch:>3}  train {tr_acc:.3f}  val {va_acc:.3f}"
                  f"  (best {best_acc:.3f})")

        # Early stopping is not optional at this data size - without it the
        # model memorizes 964 rows long before it generalizes.
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch} "
                  f"({args.patience} epochs without improvement)")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
        va_pred = model(Xva).argmax(1).numpy()
    va_acc = accuracy_score(y[va], va_pred)

    print(f"\nTrain accuracy: {tr_acc:.3f}")
    print(f"Val accuracy:   {va_acc:.3f}   <- compare against the RF on the same split")
    print("\nPer-class report on val:")
    print(classification_report(y[va], va_pred, labels=range(len(classes)),
                                target_names=classes, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, f"pytorch_{args.model}.pt")
    torch.save({"state_dict": best_state, "classes": classes,
                "timesteps": args.timesteps, "model": args.model,
                "hidden": args.hidden}, out)
    print(f"Model saved to: {out}")
    print("\nNOTE: the test split was NOT touched. Keep it for one final number.")


if __name__ == "__main__":
    main()
