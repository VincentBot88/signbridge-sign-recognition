"""
SignBridge - v3 ensemble: does RandomForest + GRU beat both, on identical rows?

WHY NOW
-------
The v2 ensemble (ensemble_test.py) lost to the RF alone, and the post-mortem
was specific: the GRU was ~10 points behind (0.69 vs 0.78) and its softmax
was confidently wrong, so averaging let the weaker model overrule the stronger
one twice as often as it rescued it.

Neither condition holds any more. The v3 body-only GRU with strength-3
augmentation is level with the v3 RF on val (0.888 vs 0.895), and the two read
genuinely different inputs:

    RF   1,221 ENGINEERED columns: hand-local keyframes (handshape) + body
         frame location, path length, peak speed
    GRU  the raw 32-step BODY-FRAME trajectory, no hand-local block at all

Two equally good models that see different things are the textbook case for
an ensemble. Early hints that they fail differently: the RF v3 got BYE 2/3 and
WAIT 3/4 against the GRU's ~1.4/3 and ~2.5/4, and the GRU is ahead on HUNGRY.
Those rest on 3-4 clips each, which is why this script measures rather than
assumes.

PROTOCOL - fixed before any result was seen
-------------------------------------------
* Both models train on the SAME 1,000-row fit set. Dev (250 rows, 8 signers)
  is held out from BOTH. The RF has so far trained on all 1,250 train rows,
  which is why its 0.895 was never a like-for-like number against the GRU.
* Pairs: GRU seed k of stream s is paired with an RF of random_state
  k + 100*s, so the 10 pairs are independent in both components. Stream 0
  therefore uses RF seeds 42-46, which includes the recorded seed 42.
* HEADLINE = equal-weight average of TEMPERATURE-CALIBRATED probabilities.
  One temperature per model per seed, fit on DEV by log-loss. Chosen in
  advance because a GRU trained ~1,000 epochs to fit accuracy ~1.0 has a far
  more confident softmax than an RF's vote fractions; averaged raw, the GRU's
  vote would win nearly every disagreement regardless of who is right. That
  is the v2 failure mode, and calibration is the standard fix for it.
* Secondary, reported but never headlined: raw average; a dev-tuned weight;
  5-seed averages (the deployable "average everything" model).
* VAL IS NEVER USED TO CHOOSE ANYTHING. Temperatures and the secondary weight
  come from dev. There is no val weight sweep - ensemble_test.py printed one,
  and reading it is tuning on val.
* Decision rule: the ensemble is SUPPORTED only if it beats BOTH single models
  on val on BOTH streams, each clearing the paired t threshold (n=5,
  crit 2.78), AND dev agrees on direction. Anything less is reported as
  suggestive or not supported, whatever the pooled mean looks like.

CHECKS BEFORE ANY NUMBER IS PRINTED
-----------------------------------
* Row alignment: GRU row indices, labels and signers must match the rows
  rebuilt here, exactly, or the script stops.
* argmax of each GRU seed's saved val softmax must reproduce that seed's
  recorded val@best-dev.
* If a sweep JSON with the same training config exists, the GRU's per-seed
  val@best-dev must equal it. Training is deterministic, so a mismatch means
  the code or the environment changed.
* An RF fit on all 1,250 train rows at random_state 42 must reproduce the
  recorded RF v3 val result (111/124). A mismatch means the features or the
  scikit-learn version differ, and the RF numbers are then not comparable to
  the recorded ones.

INPUT
-----
  GRU probability files written by train_pytorch_v3.py --save-probs, one per
  augmentation stream.
  RF features: rebuilt from data/landmarks_v3/*.npz with
  build_clip_features_v3() (the single source of truth), or read from
  --features-csv as a fast path. The two give identical features (checked:
  max abs difference 7e-15 over the same clips). Either way row i is the same
  clip for both models, and the alignment check above proves it.

RUN (about 80 min of GRU training, then ~1 min here)
---
  python train_pytorch_v3.py --no-hand-local --augment --aug-strength 3 ^
      --aug-stream 0 --seeds 5 --patience 200 --epochs 2000 --torch-threads 1 ^
      --no-save --save-probs sweeps/ensemble/gru_s3_stream0.npz
  python train_pytorch_v3.py --no-hand-local --augment --aug-strength 3 ^
      --aug-stream 1 --seeds 5 --patience 200 --epochs 2000 --torch-threads 1 ^
      --no-save --save-probs sweeps/ensemble/gru_s3_stream1.npz
  python ensemble_v3.py sweeps/ensemble/gru_s3_stream0.npz sweeps/ensemble/gru_s3_stream1.npz

The test split is never loaded.
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np

import train_pytorch_v3 as T      # loader + the exact fit/dev split; no torch
from build_feature_vectors_v3 import build_clip_features_v3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(BASE_DIR, "sweeps", "aug_strength")
DEFAULT_OUT = os.path.join(BASE_DIR, "sweeps", "ensemble", "ensemble_v3.json")

# Identical to train_classifier.py, the recipe behind the recorded RF v3 result.
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, class_weight="balanced")
RECORDED_RF_V3_VAL_CLIPS = 111        # all 1,250 train rows, random_state 42
RF_SEED_STREAM_OFFSET = 100

TEMPS = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 2)      # RF share, for dev tuning
HEADLINE = "avg-cal"
SWEEP_CONFIG_KEYS = ("model", "timesteps", "hidden", "dropout", "lr", "batch",
                     "epochs", "patience", "seed", "seeds", "dev_frac",
                     "holdout_seed", "with_hand_local", "with_body", "augment",
                     "aug_strength", "aug_stream", "legacy_aug_rng")


# ---------------------------------------------------------------------------
# Rows and features
# ---------------------------------------------------------------------------

def load_rows(landmarks_dir, features_csv, dev_frac, holdout_seed):
    """
    Rebuild the GRU's row order (train+val, sorted .npz files) with RF
    features, labels, signers and the fit/dev/val masks.
    """
    t0 = time.time()
    if features_csv:
        import pandas as pd
        df = pd.read_csv(features_csv)
        df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
        feat_cols = [c for c in df.columns
                     if c not in ("label", "split", "participant")]
        X = df[feat_cols].to_numpy(dtype=np.float64)
        labels = df["label"].astype(str).tolist()
        splits = np.array(df["split"].astype(str).tolist(), dtype=str)
        participants = np.array(df["participant"].astype(str).tolist(), dtype=str)
        source = f"{os.path.relpath(features_csv, BASE_DIR)} (fast path)"
    else:
        clips = T.load_raw_clips(landmarks_dir, quiet=True)
        X = np.stack([build_clip_features_v3(c["left_hand"], c["right_hand"],
                                             c["pose"], c["frame_w"],
                                             c["frame_h"])
                      for c in clips]).astype(np.float64)
        labels = [c["label"] for c in clips]
        splits = np.array([c["split"] for c in clips])
        participants = np.array([c["participant"] for c in clips])
        source = f"{os.path.relpath(landmarks_dir, BASE_DIR)} via build_clip_features_v3"

    classes = sorted(set(labels))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[l] for l in labels], dtype=np.int64)
    train = splits == "train"
    val = splits == "val"
    fit, dev = T.make_dev_split(participants, train, dev_frac, holdout_seed)
    return {"X": X, "y": y, "classes": classes, "participants": participants,
            "train": train, "val": val, "fit": fit, "dev": dev,
            "source": source, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# GRU probability files
# ---------------------------------------------------------------------------

def load_gru_file(path):
    z = np.load(path, allow_pickle=False)
    g = {k: z[k] for k in z.files}
    g["config"] = json.loads(str(g["config"]))
    g["classes"] = [str(c) for c in g["classes"]]
    g["path"] = path
    return g


def check_gru_alignment(g, rows):
    """Refuse to blend unless every row is provably the same clip."""
    name = os.path.basename(g["path"])
    problems = []
    if g["classes"] != rows["classes"]:
        problems.append("class list / order differs")
    for part in ("dev", "val"):
        if not np.array_equal(g[f"{part}_rows"], np.flatnonzero(rows[part])):
            problems.append(f"{part} row indices differ")
            continue
        if not np.array_equal(g[f"y_{part}"], rows["y"][rows[part]]):
            problems.append(f"{part} labels differ")
        if not np.array_equal(g[f"{part}_participants"].astype(str),
                              rows["participants"][rows[part]]):
            problems.append(f"{part} signers differ")
    if problems:
        raise SystemExit(f"ERROR: {name} does not line up with the rows rebuilt "
                         f"here: {'; '.join(problems)}.\n  Blending would mix "
                         f"different clips. Was a clip added, removed or "
                         f"skipped since the GRU run?")

    y_val = g["y_val"]
    recomputed = (g["val_prob"].argmax(2) == y_val[None, :]).mean(1)
    if not np.allclose(recomputed, g["val_at_best_dev"], atol=1e-6):
        raise SystemExit(f"ERROR: {name}: argmax of the saved val softmax gives "
                         f"{np.round(recomputed, 3)} but the run recorded "
                         f"{np.round(g['val_at_best_dev'], 3)}.")


def sweep_crosscheck(g):
    """Compare per-seed val@best-dev against a sweep cell with the same config."""
    cfg = g["config"]
    for p in sorted(glob.glob(os.path.join(SWEEP_DIR, "*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        scfg = s.get("config", {})
        if any(scfg.get(k) != cfg.get(k) for k in SWEEP_CONFIG_KEYS):
            continue
        rec = {r["seed"]: r["val_at_best_dev"] for r in s.get("seeds", [])}
        mine = dict(zip(g["seeds"].tolist(), g["val_at_best_dev"].tolist()))
        common = sorted(set(rec) & set(mine))
        if not common:
            continue
        same = all(abs(rec[k] - mine[k]) < 1e-6 for k in common)
        return os.path.basename(p), same, common
    return None, None, None


# ---------------------------------------------------------------------------
# Probabilities, calibration, scoring
# ---------------------------------------------------------------------------

def rf_probs(X, y, train_mask, eval_masks, seed, n_classes):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=-1)
    clf.fit(X[train_mask], y[train_mask])
    out = []
    for m in eval_masks:
        P = np.zeros((int(m.sum()), n_classes))
        P[:, clf.classes_] = clf.predict_proba(X[m])
        out.append(P)
    return out


def temper(P, temp):
    """Temperature-scale a probability matrix: softmax(log P / T)."""
    z = np.log(np.clip(P, 1e-7, 1.0)) / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(P, y):
    losses = [nll(temper(P, t), y) for t in TEMPS]
    i = int(np.argmin(losses))
    return float(TEMPS[i]), (i == 0 or i == len(TEMPS) - 1)


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def selective_acc(P, y, coverage):
    """Accuracy on the `coverage` fraction of clips the model is surest of."""
    k = max(1, int(math.ceil(coverage * len(y))))
    keep = np.argsort(-P.max(1), kind="stable")[:k]
    return float((P[keep].argmax(1) == y[keep]).mean())


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    t = d.mean() / (sd / math.sqrt(n)) if sd > 0 else (math.inf if d.mean() > 0 else
                                                      (-math.inf if d.mean() < 0 else 0.0))
    return float(d.mean()), float(t)


def crit_t(n):
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.975, n - 1))
    except ImportError:
        return {5: 2.776, 10: 2.262}.get(n, 2.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gru_files", nargs="+",
                    help=".npz files from train_pytorch_v3.py --save-probs, one "
                         "per augmentation stream")
    ap.add_argument("--features-csv", default=None,
                    help="read RF features from this CSV (e.g. data/features_v3.csv) "
                         "instead of rebuilding them from the landmarks. Faster; "
                         "the row-alignment check still applies.")
    ap.add_argument("--landmarks-dir", default=T.LANDMARKS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT, help="JSON results path")
    args = ap.parse_args()

    grus = [load_gru_file(p) for p in args.gru_files]
    streams = [g["config"].get("aug_stream", 0) for g in grus]
    if len(set(streams)) != len(streams):
        raise SystemExit(f"ERROR: two GRU files share an augmentation stream "
                         f"({streams}). Pass one file per stream.")
    base_cfg = {k: v for k, v in grus[0]["config"].items() if k != "aug_stream"}
    for g in grus[1:]:
        other = {k: v for k, v in g["config"].items() if k != "aug_stream"}
        if other != base_cfg:
            diff = sorted(k for k in set(base_cfg) | set(other)
                          if base_cfg.get(k) != other.get(k))
            raise SystemExit(f"ERROR: GRU files differ in more than the stream: {diff}")
    cfg = grus[0]["config"]

    print("=" * 76)
    print("ENSEMBLE v3 - RandomForest (hand-local + body) + GRU (body-only), same rows")
    print("=" * 76)
    rows = load_rows(args.landmarks_dir, args.features_csv,
                     cfg["dev_frac"], cfg["holdout_seed"])
    X, y, classes = rows["X"], rows["y"], rows["classes"]
    fit, dev, val, train = rows["fit"], rows["dev"], rows["val"], rows["train"]
    C = len(classes)
    y_dev, y_val = y[dev], y[val]
    n_val, n_dev = int(val.sum()), int(dev.sum())
    print(f"RF features: {X.shape[1]} columns from {rows['source']} "
          f"({rows['seconds']:.0f}s)")
    print(f"rows: fit {int(fit.sum())} | dev {n_dev} | val {n_val} | "
          f"{C} classes. Test split never loaded.")
    print(f"GRU: {'body-only' if not cfg['with_hand_local'] else 'hand-local+body'}, "
          f"aug strength {cfg['aug_strength']:g}, patience {cfg['patience']}, "
          f"epochs cap {cfg['epochs']}, seeds {cfg['seed']}-{cfg['seed'] + cfg['seeds'] - 1}, "
          f"streams {streams}")

    # ---------------------------------------------------------------- checks
    print("\nCHECKS")
    for g in grus:
        check_gru_alignment(g, rows)
        name = os.path.basename(g["path"])
        clips = [int(round(v * n_val)) for v in g["val_at_best_dev"]]
        print(f"  {name}: rows, labels and signers line up; argmax reproduces "
              f"val@best-dev {clips}")
        cell, same, common = sweep_crosscheck(g)
        if cell is None:
            print(f"    (no sweep JSON with this exact config to compare against)")
        elif same:
            print(f"    matches recorded sweep cell {cell} exactly on seeds "
                  f"{common[0]}-{common[-1]} - training is unchanged")
        else:
            print(f"    !! DIFFERS from recorded sweep cell {cell}. Training is "
                  f"deterministic, so the code or the environment changed. "
                  f"Numbers below are internally valid but not comparable to "
                  f"the recorded grid.")

    (P_ref,) = rf_probs(X, y, train, [val], 42, C)
    ref_clips = int(round(acc(P_ref, y_val) * n_val))
    if ref_clips == RECORDED_RF_V3_VAL_CLIPS:
        print(f"  RF on all {int(train.sum())} train rows, seed 42: "
              f"{ref_clips}/{n_val} - reproduces the recorded RF v3 result")
    else:
        print(f"  !! RF on all {int(train.sum())} train rows, seed 42: "
              f"{ref_clips}/{n_val}, recorded {RECORDED_RF_V3_VAL_CLIPS}/{n_val}. "
              f"Features or scikit-learn version differ from the recorded run; "
              f"RF numbers below are internally valid but not comparable to it.")

    # ------------------------------------------------------------ per pair
    methods = ["RF", "GRU", "avg-cal", "avg-raw", "dev-weight"]
    pairs = []
    per_class_pred = {m: [] for m in ("RF", "GRU", HEADLINE)}
    stream_probs = {}
    t0 = time.time()
    for g in grus:
        stream = int(g["config"].get("aug_stream", 0))
        cal = {"RF": [], "GRU": []}
        for k, gseed in enumerate(g["seeds"].tolist()):
            rseed = gseed + RF_SEED_STREAM_OFFSET * stream
            R_dev, R_val = rf_probs(X, y, fit, [dev, val], rseed, C)
            G_dev, G_val = g["dev_prob"][k].astype(np.float64), g["val_prob"][k].astype(np.float64)
            t_rf, edge_rf = fit_temperature(R_dev, y_dev)
            t_gru, edge_gru = fit_temperature(G_dev, y_dev)
            Rc_dev, Rc_val = temper(R_dev, t_rf), temper(R_val, t_rf)
            Gc_dev, Gc_val = temper(G_dev, t_gru), temper(G_val, t_gru)
            cal["RF"].append(Rc_val)
            cal["GRU"].append(Gc_val)

            dev_accs = [acc(w * Rc_dev + (1 - w) * Gc_dev, y_dev) for w in WEIGHTS]
            best = max(dev_accs)
            w_star = float(min((w for w, a in zip(WEIGHTS, dev_accs) if a == best),
                               key=lambda w: abs(w - 0.5)))

            probs = {
                "RF": (R_dev, R_val),
                "GRU": (G_dev, G_val),
                "avg-cal": (0.5 * Rc_dev + 0.5 * Gc_dev, 0.5 * Rc_val + 0.5 * Gc_val),
                "avg-raw": (0.5 * R_dev + 0.5 * G_dev, 0.5 * R_val + 0.5 * G_val),
                "dev-weight": (w_star * Rc_dev + (1 - w_star) * Gc_dev,
                               w_star * Rc_val + (1 - w_star) * Gc_val),
            }
            # For confidence, single models are scored on their calibrated
            # probabilities (temperature is what a deployed model would use).
            conf_probs = {"RF": Rc_val, "GRU": Gc_val, "avg-cal": probs["avg-cal"][1],
                          "avg-raw": probs["avg-raw"][1], "dev-weight": probs["dev-weight"][1]}

            rf_ok = R_val.argmax(1) == y_val
            gru_ok = G_val.argmax(1) == y_val
            ens_ok = probs[HEADLINE][1].argmax(1) == y_val
            rec = {
                "stream": stream, "gru_seed": gseed, "rf_seed": rseed,
                "t_rf": t_rf, "t_gru": t_gru, "t_edge": bool(edge_rf or edge_gru),
                "w_star": w_star,
                "val": {m: acc(probs[m][1], y_val) for m in methods},
                "dev": {m: acc(probs[m][0], y_dev) for m in methods},
                "sel90": {m: selective_acc(conf_probs[m], y_val, 0.9) for m in methods},
                "sel80": {m: selective_acc(conf_probs[m], y_val, 0.8) for m in methods},
                "mean_conf_wrong": {
                    "RF-raw": float(R_val.max(1)[~rf_ok].mean()) if (~rf_ok).any() else None,
                    "GRU-raw": float(G_val.max(1)[~gru_ok].mean()) if (~gru_ok).any() else None,
                },
                "overlap": {
                    "both_right": int((rf_ok & gru_ok).sum()),
                    "rf_only": int((rf_ok & ~gru_ok).sum()),
                    "gru_only": int((~rf_ok & gru_ok).sum()),
                    "both_wrong": int((~rf_ok & ~gru_ok).sum()),
                    "ens_right_of_rf_only": int((ens_ok & rf_ok & ~gru_ok).sum()),
                    "ens_right_of_gru_only": int((ens_ok & ~rf_ok & gru_ok).sum()),
                    "ens_right_of_both_wrong": int((ens_ok & ~rf_ok & ~gru_ok).sum()),
                },
            }
            pairs.append(rec)
            for m, P in (("RF", R_val), ("GRU", G_val), (HEADLINE, probs[HEADLINE][1])):
                per_class_pred[m].append(P.argmax(1))
        stream_probs[stream] = cal
    print(f"  {len(pairs)} RF fits on the {int(fit.sum())}-row fit set "
          f"({time.time() - t0:.0f}s)")
    if any(p["t_edge"] for p in pairs):
        print("  !! a fitted temperature hit the edge of its search grid - "
              "calibration for that pair is unreliable")

    # ---------------------------------------------------------------- report
    def clips_of(vals):
        return np.asarray(vals) * n_val

    def fmt(vals):
        c = clips_of(vals)
        sd = c.std(ddof=1) if len(c) > 1 else 0.0
        return f"{np.mean(vals):.3f} ({c.mean():5.1f} +/- {sd:3.1f})"

    n_fit = int(fit.sum())
    labels = {"RF": f"RF alone ({n_fit:,} fit rows)", "GRU": "GRU alone",
              "avg-cal": "RF+GRU calibrated avg  <- HEADLINE",
              "avg-raw": "RF+GRU raw avg", "dev-weight": "RF+GRU dev-tuned weight"}
    print("\n" + "=" * 76)
    print(f"VAL ACCURACY  (clips out of {n_val}; mean +/- sd over seed pairs)")
    print("=" * 76)
    head = f"  {'method':<36}" + "".join(f"{'stream ' + str(s):>22}" for s in streams)
    head += f"{'pooled':>22}"
    print(head)
    for m in methods:
        line = f"  {labels[m]:<36}"
        for s in streams:
            line += f"{fmt([p['val'][m] for p in pairs if p['stream'] == s]):>22}"
        line += f"{fmt([p['val'][m] for p in pairs]):>22}"
        print(line)
    print(f"\n  dev ({n_dev} clips, cross-check only; temperatures and the dev-tuned "
          f"weight were fit on dev,\n  and the GRU's checkpoint was selected on "
          f"it, so dev flatters those):")
    for m in methods:
        vals = [p["dev"][m] for p in pairs]
        print(f"    {labels[m]:<36} {np.mean(vals):.3f} ({np.mean(vals) * n_dev:5.1f}/{n_dev})")
    print(f"\n  dev-tuned RF weight per pair: {[p['w_star'] for p in pairs]}")
    print(f"  fitted temperatures - RF: {np.round([p['t_rf'] for p in pairs], 2).tolist()}")
    print(f"                        GRU: {np.round([p['t_gru'] for p in pairs], 2).tolist()}")

    # paired tests
    print("\n" + "=" * 76)
    print("PAIRED TESTS  (difference in val clips, same seed pair)")
    print("=" * 76)
    verdict_rows = []
    for a, b in ((HEADLINE, "RF"), (HEADLINE, "GRU"), ("GRU", "RF")):
        line = f"  {a + ' vs ' + b:<18}"
        ok_all = True
        for s in streams:
            sub = [p for p in pairs if p["stream"] == s]
            d, t = paired_t(clips_of([p["val"][a] for p in sub]),
                            clips_of([p["val"][b] for p in sub]))
            cr = crit_t(len(sub))
            sig = abs(t) > cr
            line += f"   stream {s}: {d:+5.1f}, t={t:+6.2f}{' *' if sig else '  '}"
            if a == HEADLINE:
                ok_all &= (d > 0 and t > cr)
        d, t = paired_t(clips_of([p["val"][a] for p in pairs]),
                        clips_of([p["val"][b] for p in pairs]))
        line += f"   pooled: {d:+5.1f}, t={t:+6.2f}{' *' if abs(t) > crit_t(len(pairs)) else ''}"
        print(line)
        if a == HEADLINE:
            dev_d = float(np.mean([p["dev"][a] - p["dev"][b] for p in pairs]))
            verdict_rows.append((b, ok_all, dev_d))
    print(f"  (* clears 0.05; crit t = {crit_t(5):.2f} at n=5, "
          f"{crit_t(len(pairs)):.2f} pooled. Per-stream tests are primary.)")

    beats_all = all(ok for _, ok, _ in verdict_rows)
    dev_agrees = all(dd > 0 for _, _, dd in verdict_rows)
    mean_gain = min(np.mean([p["val"][HEADLINE] - p["val"][b] for p in pairs])
                    for b, _, _ in verdict_rows) * n_val
    if beats_all and dev_agrees:
        verdict = "SUPPORTED - beats both models on both streams (paired t), dev agrees"
    elif mean_gain > 0 and dev_agrees:
        verdict = ("SUGGESTIVE - ahead of both on average, but not significant on "
                   "both streams")
    else:
        verdict = "NOT SUPPORTED - does not beat both single models"
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")

    # error overlap
    print("\n" + "=" * 76)
    print("WHERE THE MODELS DISAGREE  (val, mean clips per seed pair)")
    print("=" * 76)
    ov = {k: np.mean([p["overlap"][k] for p in pairs]) for k in pairs[0]["overlap"]}
    print(f"  both right {ov['both_right']:5.1f} | RF only {ov['rf_only']:4.1f} | "
          f"GRU only {ov['gru_only']:4.1f} | both wrong {ov['both_wrong']:4.1f}")
    print(f"  ceiling for ANY blend of these two (an oracle that always picks the "
          f"right one): {n_val - ov['both_wrong']:.1f}/{n_val}")
    print(f"  headline ensemble keeps {ov['ens_right_of_rf_only']:.1f} of the "
          f"{ov['rf_only']:.1f} RF-only clips and {ov['ens_right_of_gru_only']:.1f} "
          f"of the {ov['gru_only']:.1f} GRU-only clips; "
          f"recovers {ov['ens_right_of_both_wrong']:.1f} that both got wrong")
    mcw = [p["mean_conf_wrong"] for p in pairs]
    rf_w = [m["RF-raw"] for m in mcw if m["RF-raw"] is not None]
    gru_w = [m["GRU-raw"] for m in mcw if m["GRU-raw"] is not None]
    if rf_w and gru_w:
        print(f"  mean top-probability when WRONG (raw): RF {np.mean(rf_w):.2f}, "
              f"GRU {np.mean(gru_w):.2f}  - why the raw average is not the headline")

    # everything averaged
    print("\n" + "=" * 76)
    print("AVERAGE EVERYTHING  (all 5 seeds per model, calibrated; one number per stream)")
    print("=" * 76)
    for s in streams:
        cal = stream_probs[s]
        rf5, gru5 = np.mean(cal["RF"], axis=0), np.mean(cal["GRU"], axis=0)
        both = 0.5 * rf5 + 0.5 * gru5
        single_rf = np.mean([acc(P, y_val) for P in cal["RF"]]) * n_val
        single_gru = np.mean([acc(P, y_val) for P in cal["GRU"]]) * n_val
        print(f"  stream {s}:  RF x5 {acc(rf5, y_val) * n_val:5.1f} (single mean "
              f"{single_rf:5.1f}) | GRU x5 {acc(gru5, y_val) * n_val:5.1f} (single "
              f"mean {single_gru:5.1f}) | RF x5 + GRU x5 {acc(both, y_val) * n_val:5.1f}")
    print("  No spread on these - one ensemble per stream. Diagnostic, not headline.")

    # confidence
    print("\n" + "=" * 76)
    print("CONFIDENCE  (val accuracy on the clips each model is surest of - the kiosk's")
    print("abstention question: if it declines the least-confident 10% / 20%)")
    print("=" * 76)
    for m in ("RF", "GRU", HEADLINE):
        s90 = np.mean([p["sel90"][m] for p in pairs])
        s80 = np.mean([p["sel80"][m] for p in pairs])
        print(f"  {labels[m]:<36} all {np.mean([p['val'][m] for p in pairs]):.3f} | "
              f"top 90% {s90:.3f} | top 80% {s80:.3f}")

    # per class
    print("\n" + "=" * 76)
    print("PER CLASS  (val recall, mean over pairs; 'pred' = times predicted per run)")
    print("=" * 76)
    support = np.bincount(y_val, minlength=C)
    rec_of = {}
    pred_of = {}
    for m in per_class_pred:
        preds = np.stack(per_class_pred[m])                       # (pairs, n_val)
        hit = preds == y_val[None, :]
        rec_of[m] = np.array([[hit[i][y_val == c].mean() for c in range(C)]
                              for i in range(len(preds))])        # (pairs, C)
        pred_of[m] = np.array([[(preds[i] == c).sum() for c in range(C)]
                               for i in range(len(preds))])
    stream_idx = {s: [i for i, p in enumerate(pairs) if p["stream"] == s] for s in streams}
    order = np.argsort(rec_of[HEADLINE].mean(0), kind="stable")
    hdr = f"  {'class':<12}{'n':>3}  {'RF':>5}  {'GRU':>5}  {'ENS':>5}"
    hdr += "".join(f"  {'ENS s' + str(s):>7}" for s in streams)
    hdr += f"  {'pred RF/GRU/ENS':>17}"
    print(hdr)
    for c in order[:12]:
        line = (f"  {classes[c]:<12}{support[c]:>3}  {rec_of['RF'][:, c].mean():5.2f}  "
                f"{rec_of['GRU'][:, c].mean():5.2f}  {rec_of[HEADLINE][:, c].mean():5.2f}")
        for s in streams:
            line += f"  {rec_of[HEADLINE][stream_idx[s], c].mean():7.2f}"
        line += (f"  {pred_of['RF'][:, c].mean():5.1f}/{pred_of['GRU'][:, c].mean():4.1f}/"
                 f"{pred_of[HEADLINE][:, c].mean():4.1f}")
        print(line)
    print("  Worst 12 by ensemble recall. Per-class claims need the ENS columns to "
          "agree across streams,\n  and rest on 3-9 val clips from 5 signers.")

    # save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "gru_files": [os.path.relpath(g["path"], BASE_DIR) for g in grus],
            "gru_config": cfg, "streams": streams, "rf_params": RF_PARAMS,
            "rf_reference_val_clips": ref_clips, "features": rows["source"],
            "n_val": n_val, "n_dev": n_dev, "classes": classes,
            "pairs": pairs, "verdict": verdict,
            "per_class_recall": {m: rec_of[m].tolist() for m in rec_of},
            "per_class_predicted": {m: pred_of[m].tolist() for m in pred_of},
        }, fh, indent=2)
    print(f"\nJSON written to: {args.out}")
    print("Test split never loaded. Val was read, never selected on.")


if __name__ == "__main__":
    main()
