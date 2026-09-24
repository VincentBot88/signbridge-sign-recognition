"""
SignBridge - freeze the v3 ensemble into ONE file the kiosk loads.

Builds models/signbridge_ensemble_v3.joblib from:
  * the 5 GRU seeds saved by  train_pytorch_v3.py --save-seeds
  * a RandomForest (random_state 42) trained here on the same 1,000 fit rows
  * one temperature per model, fit on dev
  * an abstention threshold, chosen on dev

Nothing here is chosen on val. Val is read once at the end and reported, so
the bundle's numbers can be compared with the ensemble experiment's.

WHAT IT CHECKS BEFORE SAVING (and refuses to save if any fails)
---------------------------------------------------------------
1. Rows: the GRU seeds' class list, dev rows and val rows match the rows
   rebuilt here.
2. Train/serve parity: the RF features and GRU sequences built by
   signbridge_ensemble.clip_inputs() - the path the kiosk uses - are
   identical to the training path's (build_clip_features_v3 / build_X).
3. numpy == PyTorch: signbridge_ensemble's numpy GRU gives the same
   probabilities as the PyTorch model on every dev and val clip, so the
   kiosk can run without PyTorch.
4. Same models: each seed reproduces its recorded val@best-dev, and, when
   sweeps/ensemble/gru_s3_stream<N>.npz exists with the same config, its
   dev/val probabilities match the ones the ensemble experiment measured.

THE ABSTENTION THRESHOLD
------------------------
The lowest confidence threshold at which the clips the ensemble ACCEPTS on
dev are at least --target-accuracy correct (default 0.95), provided at least
--min-coverage of dev is accepted (default 0.5). Below the threshold the
kiosk should ask the user to confirm instead of acting.
The GRU's checkpoints and all temperatures were fit on dev, so dev is
slightly flattering and the chosen threshold slightly permissive. The val
and test numbers printed later are the honest check on it.

RUN
---
  # 10-second smoke test first:
  python train_pytorch_v3.py --no-hand-local --seeds 2 --epochs 3 --patience 3 ^
      --no-save --save-seeds sweeps/ensemble/_smoke_seeds.pt
  python build_ensemble_bundle.py --gru-seeds sweeps/ensemble/_smoke_seeds.pt --smoke

  # the real thing (~40 min of GRU training, then ~1 min here):
  python train_pytorch_v3.py --no-hand-local --augment --aug-strength 3 ^
      --aug-stream 0 --seeds 5 --patience 200 --epochs 2000 --torch-threads 1 ^
      --no-save --save-seeds models/gru_v3_s3_seeds.pt
  python build_ensemble_bundle.py --gru-seeds models/gru_v3_s3_seeds.pt

The test split is never loaded.
"""

import argparse
import datetime
import json
import os
import time

import numpy as np

import train_pytorch_v3 as T
from build_feature_vectors_v3 import build_clip_features_v3
from ensemble_v3 import fit_temperature, RF_PARAMS
import signbridge_ensemble as SE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE = {"with_hand_local": False, "with_body": True, "augment": True,
          "aug_strength": 3.0, "seeds": 5, "patience": 200, "epochs": 2000}


def fail(msg):
    raise SystemExit(f"ERROR: {msg}\nNothing was saved.")


def choose_threshold(P, y, target, min_coverage):
    """Lowest threshold whose accepted dev clips reach `target` accuracy."""
    conf = P.max(axis=1)
    correct = P.argmax(axis=1) == y
    order = np.argsort(conf)                      # ascending
    conf_s, corr_s = conf[order], correct[order]
    n = len(y)
    # accepted set for threshold conf_s[i] is order[i:]  (ties: >=)
    suffix_correct = np.cumsum(corr_s[::-1])[::-1]
    best = None
    for i in range(n):
        if i > 0 and conf_s[i] == conf_s[i - 1]:
            continue                               # same threshold as i-1
        k = n - i
        coverage = k / n
        if coverage < min_coverage:
            break
        acc = suffix_correct[i] / k
        if acc >= target:
            best = (float(conf_s[i]), coverage, float(acc))
            break
    return best


def coverage_table(P, y, levels=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)):
    conf = P.max(axis=1)
    correct = P.argmax(axis=1) == y
    order = np.argsort(-conf, kind="stable")
    rows = []
    for c in levels:
        k = max(1, int(round(c * len(y))))
        keep = order[:k]
        rows.append((c, float(conf[keep].min()), float(correct[keep].mean())))
    return rows


def at_threshold(P, y, thr):
    conf = P.max(axis=1)
    acc_mask = conf >= thr
    correct = P.argmax(axis=1) == y
    cov = float(acc_mask.mean())
    acc = float(correct[acc_mask].mean()) if acc_mask.any() else float("nan")
    return cov, acc, int((acc_mask & ~correct).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gru-seeds", required=True,
                    help=".pt written by train_pytorch_v3.py --save-seeds")
    ap.add_argument("--rf-seed", type=int, default=42)
    ap.add_argument("--target-accuracy", type=float, default=0.95,
                    help="accuracy the ACCEPTED dev clips must reach (default 0.95)")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="never accept fewer than this fraction of dev (default 0.5)")
    ap.add_argument("--check-probs", default=None,
                    help="ensemble-experiment .npz to compare against (default: "
                         "sweeps/ensemble/gru_s3_stream<N>.npz if present)")
    ap.add_argument("--landmarks-dir", default=T.LANDMARKS_DIR)
    ap.add_argument("--out", default=None,
                    help="bundle path (default models/signbridge_ensemble_v3.joblib, "
                         "or sweeps/ensemble/_smoke_bundle.joblib with --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing test: skip the recipe guard and write to a "
                         "throwaway path")
    args = ap.parse_args()
    out_path = args.out or (os.path.join(BASE_DIR, "sweeps", "ensemble", "_smoke_bundle.joblib")
                            if args.smoke else SE.DEFAULT_BUNDLE)

    try:
        import torch
        import torch.nn as nn
    except (ImportError, OSError) as e:
        fail(f"PyTorch is needed to BUILD the bundle (not to run it): {e}")
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.time()
    print("=" * 76)
    print("BUILD ENSEMBLE BUNDLE - RF + 5 GRUs, calibrated on dev, threshold on dev")
    print("=" * 76)

    # ------------------------------------------------------------ GRU seeds
    S = torch.load(args.gru_seeds, map_location="cpu", weights_only=False)
    if S.get("format") != "signbridge-gru-seeds-v1":
        fail(f"{args.gru_seeds} is not a --save-seeds file")
    cfg = S["config"]
    print(f"GRU seeds: {os.path.relpath(args.gru_seeds, BASE_DIR)} - seeds {S['seeds']}, "
          f"strength {cfg['aug_strength']:g} stream {cfg['aug_stream']}, "
          f"augment {cfg['augment']}, patience {cfg['patience']}, epochs {cfg['epochs']}")
    if not args.smoke:
        wrong = {k: (cfg.get(k), v) for k, v in RECIPE.items() if cfg.get(k) != v}
        if wrong:
            fail(f"these GRUs are not the frozen recipe: {wrong}. "
                 f"(--smoke skips this check for plumbing tests.)")

    # ------------------------------------------------------------------ rows
    clips = T.load_raw_clips(args.landmarks_dir, quiet=True)
    labels = [c["label"] for c in clips]
    classes = sorted(set(labels))
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for l in labels], dtype=np.int64)
    splits = np.array([c["split"] for c in clips])
    participants = np.array([c["participant"] for c in clips])
    train, val = splits == "train", splits == "val"
    fit, dev = T.make_dev_split(participants, train, cfg["dev_frac"], cfg["holdout_seed"])
    dev_i, val_i, fit_i = np.flatnonzero(dev), np.flatnonzero(val), np.flatnonzero(fit)
    y_dev, y_val = y[dev_i], y[val_i]

    print("\nCHECKS")
    if list(S["classes"]) != classes:
        fail("class list differs from the GRU run's")
    if S["dev_rows"] != dev_i.tolist() or S["val_rows"] != val_i.tolist():
        fail("dev/val rows differ from the GRU run's - was a clip added or removed?")
    if S["y_dev"] != y_dev.tolist() or S["y_val"] != y_val.tolist():
        fail("dev/val labels differ from the GRU run's")
    print(f"  [ok] rows: fit {len(fit_i)} | dev {len(dev_i)} | val {len(val_i)} | "
          f"{len(classes)} classes - same as the GRU run")

    # ---------------------------------------------- serve path == train path
    serve = [SE.clip_inputs(c["left_hand"], c["right_hand"], c["pose"],
                            c["frame_w"], c["frame_h"], cfg) for c in clips]
    rf_X = np.stack([s[0] for s in serve]).astype(np.float64)
    seq_all = np.stack([s[1] for s in serve]).astype(np.float32)
    rf_train_path = np.stack([build_clip_features_v3(c["left_hand"], c["right_hand"],
                                                     c["pose"], c["frame_w"], c["frame_h"])
                              for c in clips]).astype(np.float64)
    ev_rows = np.concatenate([dev_i, val_i])
    X_train_path, _ = T.build_X([clips[i] for i in ev_rows], cfg["timesteps"],
                                with_hand_local=cfg["with_hand_local"],
                                with_body=cfg["with_body"])
    d_rf = float(np.max(np.abs(np.nan_to_num(rf_X) - np.nan_to_num(rf_train_path))))
    d_gru = float(np.max(np.abs(seq_all[ev_rows] - X_train_path)))
    if d_rf != 0.0 or d_gru != 0.0:
        fail(f"serving features differ from training features (RF {d_rf}, GRU {d_gru})")
    print(f"  [ok] train/serve parity: RF features and GRU sequences identical "
          f"(max diff 0.0 over {len(clips)} / {len(ev_rows)} clips)")

    # --------------------------------------------- numpy GRU == torch GRU
    X_dev, X_val = seq_all[dev_i], seq_all[val_i]
    weights, G_dev, G_val, worst = [], [], [], 0.0
    for k, (seed, state) in enumerate(zip(S["seeds"], S["states"])):
        w = SE.gru_weights_from_state(state)
        model = T.build_model("gru", cfg["timesteps"], len(classes), cfg["hidden"],
                              cfg["dropout"], torch, nn, cfg["per_frame"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            ref_dev = torch.softmax(model(torch.tensor(X_dev)), 1).numpy()
            ref_val = torch.softmax(model(torch.tensor(X_val)), 1).numpy()
        np_dev, np_val = SE.gru_forward_numpy(w, X_dev), SE.gru_forward_numpy(w, X_val)
        diff = max(np.abs(np_dev - ref_dev).max(), np.abs(np_val - ref_val).max())
        worst = max(worst, float(diff))
        if diff > 1e-4 or not (np.array_equal(np_dev.argmax(1), ref_dev.argmax(1))
                               and np.array_equal(np_val.argmax(1), ref_val.argmax(1))):
            fail(f"seed {seed}: numpy GRU disagrees with PyTorch (max prob diff {diff:.2e})")
        val_acc = float((np_val.argmax(1) == y_val).mean())
        if abs(val_acc - S["val_at_best_dev"][k]) > 1e-6:
            fail(f"seed {seed}: val {val_acc:.4f} but the run recorded "
                 f"{S['val_at_best_dev'][k]:.4f} - not the same model")
        weights.append(w)
        G_dev.append(np_dev)
        G_val.append(np_val)
    rec = [int(round(v * len(y_val))) for v in S["val_at_best_dev"]]
    print(f"  [ok] numpy GRU == PyTorch on every dev/val clip (max prob diff "
          f"{worst:.1e}); seeds reproduce their recorded val {rec}")

    check = args.check_probs or os.path.join(
        BASE_DIR, "sweeps", "ensemble", f"gru_s3_stream{cfg['aug_stream']}.npz")
    if os.path.exists(check):
        z = np.load(check, allow_pickle=False)
        zcfg = json.loads(str(z["config"]))
        same_cfg = all(zcfg.get(k) == cfg.get(k) for k in zcfg if k != "per_frame")
        if same_cfg and z["seeds"].tolist() == S["seeds"]:
            dd = max(np.abs(np.stack(G_dev) - z["dev_prob"]).max(),
                     np.abs(np.stack(G_val) - z["val_prob"]).max())
            if dd > 1e-4:
                fail(f"these GRUs differ from the ones the ensemble experiment "
                     f"measured ({os.path.basename(check)}, max diff {dd:.1e})")
            print(f"  [ok] identical to the models measured in {os.path.basename(check)} "
                  f"(max prob diff {dd:.1e})")
        else:
            print(f"  [--] {os.path.basename(check)} has a different config; not compared")
    else:
        print("  [--] no ensemble-experiment probabilities to compare against")

    # ------------------------------------------------------------------ RF
    rf = RandomForestClassifier(**RF_PARAMS, random_state=args.rf_seed, n_jobs=-1)
    rf.fit(rf_X[fit_i], y[fit_i])
    if list(rf.classes_) != list(range(len(classes))):
        fail("a class is missing from the fit set")
    R_dev, R_val = rf.predict_proba(rf_X[dev_i]), rf.predict_proba(rf_X[val_i])
    print(f"  [ok] RF trained on the {len(fit_i)} fit rows (random_state {args.rf_seed})")

    # -------------------------------------------------------- calibration
    t_rf, edge = fit_temperature(R_dev, y_dev)
    t_gru, edges = zip(*[fit_temperature(G, y_dev) for G in G_dev])
    if edge or any(edges):
        print("  !! a temperature hit the edge of its search grid")
    print(f"\nTEMPERATURES (fit on dev)  RF {t_rf:.2f} | GRU "
          f"{', '.join(f'{t:.2f}' for t in t_gru)}")

    bundle = {
        "format": SE.BUNDLE_FORMAT,
        "classes": classes,
        "rf": {"model": rf, "temperature": t_rf, "random_state": args.rf_seed,
               "params": RF_PARAMS, "n_features": int(rf_X.shape[1]),
               "train_rows": "fit (1,000-row signer-disjoint fit set)"},
        "gru": {"config": {k: cfg[k] for k in ("timesteps", "hidden", "per_frame",
                                               "with_hand_local", "with_body")},
                "train_config": cfg, "seeds": S["seeds"], "weights": weights,
                "temperatures": list(t_gru)},
        "blend": {"rf": 0.5, "gru": 0.5},
        "threshold": {"value": None},
    }
    ens = SE.SignBridgeEnsemble(bundle)
    P_dev = ens.blend(R_dev, G_dev)
    P_val = ens.blend(R_val, G_val)

    # ------------------------------------------------------------ threshold
    print(f"\nABSTENTION THRESHOLD - chosen on dev ({len(y_dev)} clips), "
          f"target: accepted clips >= {args.target_accuracy:.0%} correct")
    print(f"  {'dev accepted':>13} {'threshold':>10} {'dev accuracy of accepted':>26}")
    for c, thr, a in coverage_table(P_dev, y_dev):
        print(f"  {c:>12.0%} {thr:>10.3f} {a:>26.3f}")
    choice = choose_threshold(P_dev, y_dev, args.target_accuracy, args.min_coverage)
    if choice is None:
        print(f"  !! no threshold reaches {args.target_accuracy:.0%} while accepting "
              f">= {args.min_coverage:.0%} of dev. Bundle saved WITHOUT a threshold "
              f"(everything accepted). Lower --target-accuracy to choose one.")
        bundle["threshold"] = {"value": None, "target_accuracy": args.target_accuracy}
    else:
        thr, cov, a = choice
        print(f"  -> threshold {thr:.3f}: accepts {cov:.0%} of dev at {a:.1%} accuracy")
        bundle["threshold"] = {"value": thr, "target_accuracy": args.target_accuracy,
                               "min_coverage": args.min_coverage,
                               "dev_coverage": cov, "dev_accepted_accuracy": a}
    ens = SE.SignBridgeEnsemble(bundle)

    # --------------------------------------------------- report (val read once)
    n_val = len(y_val)

    def clips_right(P, yy):
        return int((P.argmax(1) == yy).sum())
    gru_avg_val = np.mean([SE.temper(G, t) for G, t in zip(G_val, t_gru)], axis=0)
    gru_avg_dev = np.mean([SE.temper(G, t) for G, t in zip(G_dev, t_gru)], axis=0)
    print("\nRESULTS (nothing below was chosen on val)")
    print(f"  {'':<28}{'dev /' + str(len(y_dev)):>12}{'val /' + str(n_val):>12}")
    for name, Pd, Pv in (("RF alone", R_dev, R_val),
                         ("GRU, 5-seed average", gru_avg_dev, gru_avg_val),
                         ("ENSEMBLE (bundle)", P_dev, P_val)):
        print(f"  {name:<28}{clips_right(Pd, y_dev):>12}{clips_right(Pv, y_val):>12}"
              f"   val {clips_right(Pv, y_val) / n_val:.3f}")
    print("  (ensemble experiment, same recipe with 5 RFs: 116-117/124 on val)")
    if bundle["threshold"]["value"] is not None:
        cov, a, n_wrong = at_threshold(P_val, y_val, bundle["threshold"]["value"])
        print(f"  val at the dev threshold: accepts {cov:.0%} at {a:.1%} accuracy "
              f"({n_wrong} accepted errors); the rest are asked to confirm")
        bundle["threshold"].update({"val_coverage": cov, "val_accepted_accuracy": a})

    # ----------------------------------------------------------------- save
    bundle["provenance"] = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "gru_seeds_file": os.path.relpath(args.gru_seeds, BASE_DIR),
        "gru_seeds_sha256": SE.file_sha256(args.gru_seeds),
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "torch_used_to_build": torch.__version__,
        "dev_clips": int(clips_right(P_dev, y_dev)), "val_clips": int(clips_right(P_val, y_val)),
        "n_dev": int(len(y_dev)), "n_val": int(n_val),
        "smoke": bool(args.smoke),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)

    # the saved file must load and reproduce what was just measured
    reloaded = SE.SignBridgeEnsemble.load(out_path)
    P_chk = reloaded.predict_proba_batch(rf_X[val_i], X_val)
    if not np.allclose(P_chk, P_val, atol=1e-9):
        fail("the saved bundle does not reproduce the numbers above")
    size = os.path.getsize(out_path) / 1e6
    print(f"\nSaved {out_path} ({size:.1f} MB) - reloaded and re-checked. "
          f"{time.time() - t0:.0f}s total.")
    print(reloaded.describe())
    print("\nNext (once, when you are done): extract the test split and run "
          "evaluate_ensemble_on_test.py")


if __name__ == "__main__":
    main()
