"""
SignBridge - augmentation sweep driver for train_pytorch_v3.py.

WHAT THIS IS FOR
----------------
--aug-strength has never been tuned. It sits at the default 1.0, and the only
two points on that curve are 0 (off) and 1 (on), with a positive slope between
them. Nothing says 1.0 is the peak rather than the near side of it.

Each 5-seed run takes ~16 minutes, so the sweep is a background job, not an
interactive one. This script runs the grid, survives being interrupted, and
collates the results into one table with the statistics that actually decide
whether a difference is real.

THE TWO-STREAM RULE
-------------------
Section 4 of signbridge-pytorch-v3-ablation.md retracted four claims that were
perfectly stable across five seeds and evaporated on a second augmentation
stream. Five seeds inside one stream share that stream, so "small spread"
measures the wrong thing.

So this sweeps strength x stream, not strength alone. A strength that wins on
one stream and loses on the other has not won. The collation reports both the
per-stream figures and the stream-to-stream disagreement, and refuses to name
a winner that only holds on one.

COST
----
  strengths x streams x ~16 min.  The default grid (4 strengths, 2 streams)
  is 8 runs, about 2 hours. --dry-run prints the plan and the estimate
  without running anything.

Runs are skipped if their JSON already exists, so an interrupted sweep resumes
where it stopped, and a widened grid only runs the new cells.

RUN
    python sweep_aug_v3.py --probe-threads      # 3 min: pick --torch-threads first
    python sweep_aug_v3.py --dry-run
    python sweep_aug_v3.py
    python sweep_aug_v3.py --collate            # re-print the table, run nothing
"""

import argparse
import itertools
import json
import os
import statistics as st
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(BASE_DIR, "train_pytorch_v3.py")
RESULTS_DIR = os.path.join(BASE_DIR, "sweeps", "aug_strength")

DEFAULT_STRENGTHS = [0.5, 1.0, 1.5, 2.0]
DEFAULT_STREAMS = [0, 1]


def run_path(strength, stream, tag=""):
    """
    Result path for one cell.

    `tag` exists because the filename is also the resume key: without it, a
    re-run of the same strength/stream under a DIFFERENT --patience or
    --epochs would find the old file and skip, silently reporting the old
    config's numbers as if they were the new one's. Tag any grid that changes
    a training setting (--tag _p100) so it lands beside the original instead
    of shadowing it.
    """
    return os.path.join(RESULTS_DIR, f"s{strength:g}_stream{stream}{tag}.json")


def baseline_path(tag=""):
    return os.path.join(RESULTS_DIR, f"baseline_noaug{tag}.json")


def trainer_cmd(args, strength, stream, out_path, baseline=False):
    # --no-save is not optional here: every run writes the same checkpoint
    # path, so a sweep without it would leave whichever cell ran last sitting
    # in models/pytorch_v3_bodyonly_gru.pt.
    cmd = [sys.executable, TRAINER,
           "--no-hand-local", "--seeds", str(args.seeds), "--no-save",
           "--patience", str(args.patience), "--epochs", str(args.epochs),
           "--json-out", out_path]
    if not baseline:
        cmd += ["--augment", "--aug-strength", f"{strength:g}",
                "--aug-stream", str(stream)]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.torch_threads:
        cmd += ["--torch-threads", str(args.torch_threads)]
    return cmd


def run_one(args, strength, stream, out_path, label, baseline=False):
    if os.path.exists(out_path):
        print(f"  [skip] {label}  (already have {os.path.basename(out_path)})")
        return True
    cmd = trainer_cmd(args, strength, stream, out_path, baseline)
    print(f"  [run ] {label}")
    print(f"         {' '.join(cmd[1:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    mins = (time.time() - t0) / 60
    if proc.returncode != 0 or not os.path.exists(out_path):
        print(f"  [FAIL] {label} exited {proc.returncode} after {mins:.1f} min")
        return False
    print(f"  [done] {label}  {mins:.1f} min")
    warn_if_truncated(out_path, label)
    return True


def warn_if_truncated(path, label):
    """
    A seed that stopped at the epoch cap did not early-stop - it ran out of
    budget. Its result is a lower bound, not a measurement, and comparing it
    against a cell that converged is comparing two different experiments.
    Reported as soon as the cell finishes so a bad grid can be aborted in
    minutes rather than discovered two hours later.
    """
    p = load(path)
    if not p:
        return
    cap = p.get("config", {}).get("epochs")
    hit = [r["seed"] for r in p["seeds"] if cap and r["epochs"] >= cap]
    if hit:
        print(f"  [WARN] {label}: seed(s) {hit} hit the {cap}-epoch cap and were "
              f"CUT OFF, not early-stopped.")
        print(f"         That cell understates its strength. Re-run this grid "
              f"with a larger --epochs.")
    return not hit


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired(a_by_seed, b_by_seed):
    """(mean difference, sd, t) for a - b over the seeds both share."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2:
        return None
    d = [a_by_seed[s] - b_by_seed[s] for s in seeds]
    m, sd = st.mean(d), st.stdev(d)
    se = sd / len(d) ** 0.5
    return m, sd, (m / se if se > 0 else float("inf")), len(d)


CRIT_T = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
          7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def crit_t(n):
    """Two-tailed 0.05 critical t for n paired observations (df = n - 1)."""
    return CRIT_T.get(n, 1.96)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def by_seed(payload, key="val_at_best_dev"):
    return {r["seed"]: r[key] for r in payload["seeds"]}


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(args):
    base = load(baseline_path(args.tag))
    cells = {}
    for strength, stream in itertools.product(args.strengths, args.streams):
        p = load(run_path(strength, stream, args.tag))
        if p:
            cells[(strength, stream)] = p

    if not cells:
        print("\nNothing to collate yet.")
        return

    n_val = next(iter(cells.values()))["summary"]["n_val"]
    print("\n" + "=" * 78)
    print("AUGMENTATION STRENGTH SWEEP - body-only GRU, val@best-dev")
    print("=" * 78)

    if base:
        b = base["summary"]
        print(f"\nbaseline (no augment): {b['val_at_best_dev_mean']:.3f} "
              f"+/- {b['val_at_best_dev_std']:.3f}   "
              f"({b['val_at_best_dev_mean'] * n_val:.0f}/{n_val} clips)")

    header = f"\n{'strength':>9}"
    for stream in args.streams:
        header += f"  {'stream ' + str(stream):>16}"
    header += f"  {'pooled':>16}  {'stream gap':>10}"
    print(header)
    print("-" * 78)

    pooled_by_strength = {}
    for strength in args.strengths:
        row = f"{strength:>9.2g}"
        per_stream_means, all_vals = [], []
        for stream in args.streams:
            p = cells.get((strength, stream))
            if not p:
                row += f"  {'-':>16}"
                continue
            s = p["summary"]
            row += (f"  {s['val_at_best_dev_mean']:>8.3f} "
                    f"+/-{s['val_at_best_dev_std']:<5.3f}")
            per_stream_means.append(s["val_at_best_dev_mean"])
            all_vals += [r["val_at_best_dev"] for r in p["seeds"]]
        if all_vals:
            pooled = st.mean(all_vals)
            pooled_sd = st.stdev(all_vals) if len(all_vals) > 1 else 0.0
            pooled_by_strength[strength] = (pooled, all_vals)
            row += f"  {pooled:>8.3f} +/-{pooled_sd:<5.3f}"
            gap = (max(per_stream_means) - min(per_stream_means)
                   if len(per_stream_means) > 1 else float("nan"))
            row += f"  {gap:>10.3f}" if gap == gap else f"  {'-':>10}"
        print(row)

    print(f"\n'stream gap' is how far the two streams disagree on the SAME "
          f"strength.\nAnything smaller than that gap is not a real difference "
          f"between strengths.")

    if len(pooled_by_strength) < 2:
        print("\nToo few strengths finished to compare.")
        return

    gaps = []
    for strength in args.strengths:
        ms = [cells[(strength, s)]["summary"]["val_at_best_dev_mean"]
              for s in args.streams if (strength, s) in cells]
        if len(ms) > 1:
            gaps.append(max(ms) - min(ms))
    noise_floor = max(gaps) if gaps else 0.0

    best = max(pooled_by_strength, key=lambda k: pooled_by_strength[k][0])
    best_mean = pooled_by_strength[best][0]
    ref = 1.0 if 1.0 in pooled_by_strength else None

    print(f"\nbest pooled strength: {best:g} at {best_mean:.3f} "
          f"({best_mean * n_val:.0f}/{n_val} clips)")

    if ref is not None and best != ref:
        margin = best_mean - pooled_by_strength[ref][0]
        print(f"  margin over the current default (1.0): {margin:+.3f} "
              f"({margin * n_val:+.1f} clips)")
        if margin <= noise_floor:
            print(f"  NOT a finding: that margin is within the "
                  f"{noise_floor:.3f} stream-to-stream gap.")
        else:
            wins = all(
                cells[(best, s)]["summary"]["val_at_best_dev_mean"]
                > cells[(ref, s)]["summary"]["val_at_best_dev_mean"]
                for s in args.streams
                if (best, s) in cells and (ref, s) in cells)
            print(f"  beats 1.0 on every stream: {'yes' if wins else 'NO'}"
                  + ("" if wins else " - not a finding, it only wins on average"))
    elif best == ref:
        print("  the current default is the best of the grid - nothing to change")

    if base:
        print("\npaired against the no-augment baseline (same seeds):")
        base_seed = by_seed(base)
        for strength in args.strengths:
            for stream in args.streams:
                p = cells.get((strength, stream))
                if not p:
                    continue
                r = paired(by_seed(p), base_seed)
                if not r:
                    continue
                m, sd, t, n = r
                mark = "*" if abs(t) > crit_t(n) else " "
                print(f"  strength {strength:<4g} stream {stream}: "
                      f"{m:+.4f} ({m * n_val:+.1f} clips)  "
                      f"t={t:+.2f} vs crit {crit_t(n):.2f} {mark}")
        print("  * = clears the 0.05 threshold. Everything else is suggestive "
              "at best.")

    settings = {}
    for key, p in list(cells.items()) + ([("baseline", base)] if base else []):
        c = p.get("config", {})
        settings[key] = (c.get("patience"), c.get("epochs"), c.get("seeds"),
                         c.get("holdout_seed"), c.get("dev_frac"))
    distinct = set(settings.values())
    if len(distinct) > 1:
        print("\n  !! CELLS WERE NOT RUN UNDER THE SAME SETTINGS - this is not "
              "a curve:")
        for key, v in sorted(settings.items(), key=lambda kv: str(kv[0])):
            print(f"       {str(key):22s} patience={v[0]} epochs={v[1]} "
                  f"seeds={v[2]} holdout={v[3]} dev_frac={v[4]}")
        print("     Differences between strengths here are confounded with the "
              "settings.\n     Re-run the odd ones out, or separate the grids "
              "with --tag.")

    truncated = []
    for (strength, stream), p in cells.items():
        cap = p.get("config", {}).get("epochs")
        if cap and any(r["epochs"] >= cap for r in p["seeds"]):
            truncated.append((strength, stream))
    if truncated:
        print("\n  !! TRUNCATED CELLS (hit the epoch cap instead of early "
              "stopping):")
        for strength, stream in sorted(truncated):
            print(f"       strength {strength:g}, stream {stream}")
        print("     These understate their strength. Do not read the curve "
              "until they are re-run\n     with a larger --epochs.")

    print("\nper-class, checked ACROSS streams (the check that matters):")
    report_per_class(args, cells, n_val)


def report_per_class(args, cells, n_val):
    """Classes whose behaviour is consistent across streams, and which is not."""
    strength = 1.0 if any(k[0] == 1.0 for k in cells) else args.strengths[0]
    runs = [cells[(strength, s)] for s in args.streams if (strength, s) in cells]
    if len(runs) < 2:
        print("  (needs two streams of the same strength - not there yet)")
        return

    classes = runs[0]["classes"]
    support = runs[0]["support"]
    import numpy as np

    means = [np.array(r["per_class"]["recall"]).mean(axis=0) for r in runs]
    within = [np.array(r["per_class"]["recall"]).std(axis=0) for r in runs]

    rows = []
    for i, c in enumerate(classes):
        across = abs(means[0][i] - means[1][i])
        worst_within = max(w[i] for w in within)
        rows.append((c, support[i], means[0][i], means[1][i], across, worst_within))

    rows.sort(key=lambda r: -r[4])
    print(f"  strength {strength:g}, recall per class, "
          f"stream {args.streams[0]} vs stream {args.streams[1]}:")
    print(f"  {'class':<13}{'n':>3}  {'strm' + str(args.streams[0]):>6}  "
          f"{'strm' + str(args.streams[1]):>6}  {'across':>7}  {'within':>7}")
    for c, n, m0, m1, across, within_sd in rows[:8]:
        print(f"  {c:<13}{n:>3}  {m0:>6.2f}  {m1:>6.2f}  {across:>7.2f}  "
              f"{within_sd:>7.2f}")
    biggest = rows[0]
    print(f"\n  Largest cross-stream swing: {biggest[0]} moves {biggest[4]:.2f} "
          f"recall between streams")
    print(f"  ({biggest[4] * biggest[1]:.1f} of its {biggest[1]} val clips) while "
          f"looking stable within each.")
    print("  Any per-class claim has to be small in the 'across' column, not "
          "just the 'within' one.")


# ---------------------------------------------------------------------------
# Thread probe
# ---------------------------------------------------------------------------

def probe_threads(args):
    """
    Short runs at several torch thread counts, before spending hours.

    The rebuild is now ~0.25s of a ~0.87s epoch; the rest is the gradient step
    plus the per-epoch dev/val forward passes, all serial torch. At batch 32 x
    32 steps x 109 features through a 64-unit GRU the tensors are small enough
    that thread synchronisation can cost more than it saves, so torch's default
    (one thread per core) is not obviously right.
    """
    print("\nthread probe: 1 seed, 20 epochs each, timing only\n")
    results = {}
    for n in args.probe_thread_counts:
        out = os.path.join(RESULTS_DIR, f"_probe_threads{n}.json")
        if os.path.exists(out):
            os.remove(out)
        cmd = [sys.executable, TRAINER, "--no-hand-local", "--augment",
               "--seeds", "1", "--epochs", "20", "--patience", "999",
               "--no-save", "--json-out", out, "--torch-threads", str(n)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=BASE_DIR,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"  {n:>2} threads: FAILED")
            continue
        payload = load(out)
        epochs = payload["seeds"][0]["epochs"] if payload else 20
        per_epoch = dt / max(epochs, 1)
        results[n] = per_epoch
        print(f"  {n:>2} threads: {dt:5.1f}s total, ~{per_epoch:.2f}s/epoch "
              f"({epochs} epochs incl. startup)")
        if payload:
            os.remove(out)

    if len(results) > 1:
        best = min(results, key=results.get)
        worst = max(results, key=results.get)
        print(f"\n  fastest: {best} threads at {results[best]:.2f}s/epoch")
        print(f"  slowest: {worst} threads at {results[worst]:.2f}s/epoch "
              f"({results[worst] / results[best]:.2f}x)")
        print(f"\n  Startup (~5-10s of imports and the first rebuild) is inside "
              f"every one of\n  these, so short runs understate the difference. "
              f"Treat it as a ranking,\n  not a measurement, and pass "
              f"--torch-threads {best} to the sweep.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    ap.add_argument("--streams", type=int, nargs="+", default=DEFAULT_STREAMS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=500,
                    help="epoch cap. Higher than the trainer's 300 default on "
                         "purpose: at strength 1.0 seeds already ran 174-279 "
                         "epochs, so a stronger setting could hit 300 and be "
                         "CUT OFF rather than early-stopped. A truncated cell "
                         "looks worse than it is, which would read as 'stronger "
                         "augmentation is worse' when it only means 'stronger "
                         "augmentation needs longer'. Patience should be what "
                         "stops every run, never the cap.")
    ap.add_argument("--workers", type=int, default=0,
                    help="passed through to the trainer (0 = its default)")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="passed through to the trainer; --probe-threads suggests one")
    ap.add_argument("--minutes-per-run", type=float, default=16.0,
                    help="only used for the time estimate")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the unaugmented reference run")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for this grid's result filenames. Required "
                         "when re-running a strength under different training "
                         "settings, or the old files are treated as done.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collate", action="store_true",
                    help="collate whatever JSONs exist and exit")
    ap.add_argument("--probe-threads", action="store_true")
    ap.add_argument("--probe-thread-counts", type=int, nargs="+",
                    default=[1, 2, 4, 8])
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.collate:
        collate(args)
        return
    if args.probe_threads:
        probe_threads(args)
        return

    plan = [(s, st_) for s, st_ in itertools.product(args.strengths, args.streams)]
    todo = [(s, st_) for s, st_ in plan
            if not os.path.exists(run_path(s, st_, args.tag))]
    need_base = (not args.no_baseline
                 and not os.path.exists(baseline_path(args.tag)))

    print(f"grid      : {len(args.strengths)} strengths x {len(args.streams)} "
          f"streams = {len(plan)} runs")
    print(f"already   : {len(plan) - len(todo)} done")
    print(f"to run    : {len(todo) + (1 if need_base else 0)}")
    print(f"estimate  : ~{(len(todo) + (1 if need_base else 0)) * args.minutes_per_run / 60:.1f} hours "
          f"at {args.minutes_per_run:g} min/run")
    print(f"results   : {RESULTS_DIR}")
    print("\nInterrupting is safe - finished runs are skipped on the next start.")

    if args.dry_run:
        print("\nplan:")
        if need_base:
            print("  baseline (no augment)")
        for s, st_ in todo:
            print(f"  strength {s:g}, stream {st_}")
        return

    print()
    if need_base:
        run_one(args, 0.0, 0, baseline_path(args.tag), "baseline (no augment)",
                baseline=True)
    for i, (s, st_) in enumerate(plan, 1):
        run_one(args, s, st_, run_path(s, st_, args.tag),
                f"[{i}/{len(plan)}] strength {s:g}, stream {st_}")

    collate(args)


if __name__ == "__main__":
    main()
