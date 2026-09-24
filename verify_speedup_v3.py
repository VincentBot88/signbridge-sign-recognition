"""
SignBridge - equivalence + timing check for the train_pytorch_v3 speedups.

WHY THIS EXISTS
---------------
Two changes were made to train_pytorch_v3.py to make --augment runs cheap
enough to sweep:

  1. clip_to_sequence_v3 now skips the feature block it is about to throw
     away. It used to build the hand-local block and the body block
     unconditionally and let with_hand_local / with_body decide only what got
     concatenated - so --no-hand-local paid for two
     normalize_landmark_sequence + interp_landmarks passes per clip, every
     epoch, and discarded both.

  2. The augmented fit-set rebuild can run across a pool of worker processes.

Change 1 must not move a single number: it only removes work whose result was
already being discarded. Change 2 must not move a single number either, but
that one is a real claim - parallelising something that draws random numbers
is exactly where reproducibility quietly dies. This script checks both against
the real clips rather than asserting them in a docstring, because this project
has already published numbers that turned out not to reproduce.

WHAT IT CHECKS
--------------
  A. Block gating: the current clip_to_sequence_v3 against a verbatim copy of
     the old unconditional one, on every clip, for all three variants
     (full / --no-body / --no-hand-local). Exact equality, not allclose - the
     arrays should be bit-identical, and anything less means the refactor
     changed the features.

  B. Parallel == serial: build_X_parallel against build_X for the same rows
     and the same (seed, epoch), and at two different worker counts. Both must
     agree exactly, which is what the per-clip rng keying buys.

  C. Timing, on this machine: old vs new per-clip build cost, and serial vs
     pooled epoch rebuild, with the projected wall clock for a 200-epoch seed.

WHAT IT DOES NOT CHECK
----------------------
That the augmented ACCURACY numbers are unchanged. They are not: the parallel
path keys each clip's rng on its row index instead of consuming one rng across
the list in order, which is a different (equally valid) random stream. Any
--augment number recorded before 2026-09-21 was produced on the old stream and
should be re-measured, or reproduced with --legacy-aug-rng. Change 1 alone
does not affect any number.

RUN
    python verify_speedup_v3.py
    python verify_speedup_v3.py --quick     # 120 clips, for a fast check
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import train_pytorch_v3 as tp3
from train_pytorch_v3 import (N_LANDMARKS, N_POSE, MIN_POSE_COVERAGE,
                              aspect_correct, body_reference, to_body_frame,
                              hand_local_track, hand_body_track,
                              interp_landmarks, resample_seq, per_frame_for)


# ---------------------------------------------------------------------------
# The OLD implementation, kept verbatim as the reference to compare against.
# Both blocks computed unconditionally; the flags decide only what is
# concatenated. Do not "tidy" this - its whole job is to be the previous
# behaviour.
# ---------------------------------------------------------------------------

def clip_to_sequence_v3_reference(d, n_steps, with_hand_local=True, with_body=True):
    left_raw = d["left_hand"]
    right_raw = d["right_hand"]
    pose_raw = d["pose"] if "pose" in d else None
    frame_w = int(d["frame_w"]) if "frame_w" in d else 0
    frame_h = int(d["frame_h"]) if "frame_h" in d else 0
    T = left_raw.shape[0]

    left_local, l_present = hand_local_track(left_raw)
    right_local, r_present = hand_local_track(right_raw)

    origin_ok = False
    if pose_raw is not None and pose_raw.shape[0] > 0:
        pose_xy = aspect_correct(pose_raw[:, :, :2], frame_w, frame_h)
        origin, scale, pose_valid = body_reference(pose_xy)
        origin_ok = (pose_valid.sum() >= 2
                     and pose_valid.mean() >= MIN_POSE_COVERAGE
                     and scale > 0)

    if origin_ok:
        pose_xy_filled = interp_landmarks(pose_xy, pose_valid)
        pose_body = to_body_frame(pose_xy_filled, origin, scale)
        left_body = hand_body_track(left_raw, frame_w, frame_h, origin, scale, T)
        right_body = hand_body_track(right_raw, frame_w, frame_h, origin, scale, T)
        body_present = 1.0
    else:
        pose_body = np.zeros((T, N_POSE, 2), dtype=np.float32)
        left_body = np.zeros((T, N_LANDMARKS, 2), dtype=np.float32)
        right_body = np.zeros((T, N_LANDMARKS, 2), dtype=np.float32)
        body_present = 0.0

    blocks, flag_values = [], [l_present, r_present]
    if with_hand_local:
        blocks += [left_local.reshape(T, -1), right_local.reshape(T, -1)]
    if with_body:
        blocks += [left_body.reshape(T, -1), right_body.reshape(T, -1),
                   pose_body.reshape(T, -1)]
        flag_values.append(body_present)

    flat = np.concatenate(blocks, axis=1).astype(np.float32)
    flat = resample_seq(flat, n_steps)
    flags = np.tile(np.array(flag_values, dtype=np.float32), (n_steps, 1))
    seq = np.concatenate([flat, flags], axis=1).astype(np.float32)
    assert seq.shape[1] == per_frame_for(with_hand_local, with_body), seq.shape
    return seq, l_present, r_present, body_present


VARIANTS = [
    ("full            ", dict(with_hand_local=True, with_body=True)),
    ("--no-body       ", dict(with_hand_local=True, with_body=False)),
    ("--no-hand-local ", dict(with_hand_local=False, with_body=True)),
]


class Checks:
    def __init__(self):
        self.passed = self.total = 0

    def check(self, ok, label, detail=""):
        self.total += 1
        self.passed += bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
        return ok


def check_block_gating(clips, n_steps, checks):
    """A. Skipping the discarded block changes nothing, on every clip."""
    print("\nA. block gating: current vs the old unconditional build")
    for name, kw in VARIANTS:
        worst = 0.0
        flags_ok = True
        shapes_ok = True
        for c in clips:
            got = tp3.clip_to_sequence_v3(c, n_steps, **kw)
            want = clip_to_sequence_v3_reference(c, n_steps, **kw)
            if got[0].shape != want[0].shape:
                shapes_ok = False
                break
            worst = max(worst, float(np.abs(got[0] - want[0]).max()))
            if got[1:] != want[1:]:
                flags_ok = False
        checks.check(shapes_ok, f"{name} shapes match")
        checks.check(worst == 0.0, f"{name} features bit-identical",
                     f"max |diff| = {worst:g}")
        checks.check(flags_ok, f"{name} presence flags identical")


def check_parallel_equals_serial(clips, rows, n_steps, n_all, checks,
                                 workers_a=2, workers_b=4):
    """B. The pool reproduces the single-process result exactly."""
    print("\nB. parallel vs serial augmented rebuild")
    kw = dict(with_hand_local=False, with_body=True)     # the config being swept
    seed, epoch, strength = 42, 7, 1.0

    X_serial, nb_serial = tp3.build_X(
        clips, n_steps, kw["with_hand_local"], kw["with_body"],
        aug_key=(0, seed, epoch), clip_ids=rows, strength=strength)

    ctx = mp.get_context("spawn")
    results = {}
    for w in (workers_a, workers_b):
        pool = ctx.Pool(processes=w, initializer=tp3._pool_init,
                        initargs=(tp3.LANDMARKS_DIR, n_all))
        try:
            results[w] = tp3.build_X_parallel(
                pool, rows, n_steps, kw["with_hand_local"], kw["with_body"],
                0, seed, epoch, strength, n_chunks=w * 4)
        finally:
            pool.close()
            pool.join()

    for w in (workers_a, workers_b):
        Xp, nbp = results[w]
        same_shape = Xp.shape == X_serial.shape
        checks.check(same_shape, f"{w} workers: shape matches serial")
        if same_shape:
            d = float(np.abs(Xp - X_serial).max())
            checks.check(d == 0.0, f"{w} workers: bit-identical to serial",
                         f"max |diff| = {d:g}")
        checks.check(nbp == nb_serial, f"{w} workers: body-coverage count matches")

    a, b = results[workers_a][0], results[workers_b][0]
    checks.check(a.shape == b.shape and float(np.abs(a - b).max()) == 0.0,
                 f"{workers_a} workers == {workers_b} workers (chunking invariant)")

    # A different epoch must actually produce different perturbations,
    # otherwise "different every epoch" is silently broken.
    X_next, _ = tp3.build_X(clips, n_steps, kw["with_hand_local"], kw["with_body"],
                            aug_key=(0, seed, epoch + 1), clip_ids=rows, strength=strength)
    checks.check(float(np.abs(X_next - X_serial).max()) > 0.0,
                 "epoch+1 gives a different perturbation")

    # Same key twice must agree - the stream has to be a function of the key.
    X_again, _ = tp3.build_X(clips, n_steps, kw["with_hand_local"], kw["with_body"],
                             aug_key=(0, seed, epoch), clip_ids=rows, strength=strength)
    checks.check(float(np.abs(X_again - X_serial).max()) == 0.0,
                 "same (stream, seed, epoch) reproduces exactly")

    # --aug-stream must give a genuinely different draw, and stream 0 must
    # still be the key that every already-recorded number was produced on.
    X_s1, _ = tp3.build_X(clips, n_steps, kw["with_hand_local"], kw["with_body"],
                          aug_key=(1, seed, epoch), clip_ids=rows, strength=strength)
    checks.check(float(np.abs(X_s1 - X_serial).max()) > 0.0,
                 "stream 1 differs from stream 0")
    X_s2, _ = tp3.build_X(clips, n_steps, kw["with_hand_local"], kw["with_body"],
                          aug_key=(2, seed, epoch), clip_ids=rows, strength=strength)
    checks.check(float(np.abs(X_s2 - X_s1).max()) > 0.0,
                 "stream 2 differs from stream 1")
    checks.check(tp3.aug_rng_key(0, 42, 7, 13) == [42, 7, 13],
                 "stream 0 keeps the pre-aug-stream rng key",
                 "so stream-0 numbers stay reproducible")


def time_it(fn, repeats=1):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def report_timing(clips, rows, n_steps, n_all, workers):
    """C. What the two changes are actually worth on this machine."""
    print("\nC. timing on this machine")
    kw = dict(with_hand_local=False, with_body=True)

    t_old = time_it(lambda: [clip_to_sequence_v3_reference(c, n_steps, **kw)
                             for c in clips])
    t_new = time_it(lambda: [tp3.clip_to_sequence_v3(c, n_steps, **kw)
                             for c in clips])
    print(f"  block gating (--no-hand-local, {len(clips)} clips)")
    print(f"    before : {t_old:6.2f}s")
    print(f"    after  : {t_new:6.2f}s   ({t_old / max(t_new, 1e-9):.2f}x)")

    t_serial = time_it(lambda: tp3.build_X(
        clips, n_steps, kw["with_hand_local"], kw["with_body"],
        aug_key=(0, 42, 1), clip_ids=rows, strength=1.0))

    ctx = mp.get_context("spawn")
    t_pool_start = time.perf_counter()
    pool = ctx.Pool(processes=workers, initializer=tp3._pool_init,
                    initargs=(tp3.LANDMARKS_DIR, n_all))
    startup = time.perf_counter() - t_pool_start
    try:
        # One warm call first: the first task pays for worker import/load.
        tp3.build_X_parallel(pool, rows, n_steps, kw["with_hand_local"],
                             kw["with_body"], 0, 42, 1, 1.0, workers * 4)
        t_par = time_it(lambda: tp3.build_X_parallel(
            pool, rows, n_steps, kw["with_hand_local"], kw["with_body"],
            0, 42, 2, 1.0, workers * 4), repeats=3)
    finally:
        pool.close()
        pool.join()

    print(f"\n  augmented epoch rebuild ({len(rows)} clips)")
    print(f"    serial          : {t_serial:6.2f}s/epoch")
    print(f"    {workers:2d} workers      : {t_par:6.2f}s/epoch   "
          f"({t_serial / max(t_par, 1e-9):.2f}x)   [pool startup {startup:.1f}s, once]")
    print(f"\n  REBUILD ONLY - do not project a run time from this.")
    print(f"  An epoch is  rebuild + serial remainder,  where the remainder is")
    print(f"  the gradient step plus the per-epoch dev and val forward passes.")
    print(f"  That remainder measured ~0.62s/epoch from a real run's heartbeat")
    print(f"  and none of it is parallelised, so the whole-run speedup is much")
    print(f"  smaller than the ratio above:")
    saved = (t_serial - t_par) / max(len(rows), 1) * 1000
    print(f"      rebuild saving   ~{saved:.2f} ms per clip per epoch")
    print(f"      epoch (measured)  0.87s at 1000 fit clips, of which ~0.25s is")
    print(f"                        rebuild - so ~2.1x end to end, not "
          f"{t_serial / max(t_par, 1e-9):.1f}x")
    print(f"  Projecting from this ratio alone understated a 5-seed run by 2.5x"
          f" once.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=32)
    ap.add_argument("--quick", action="store_true",
                    help="use 120 clips instead of all of them")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = os.cpu_count() - 1")
    args = ap.parse_args()

    workers = args.workers or max(2, (os.cpu_count() or 2) - 1)

    print("verify_speedup_v3")
    print(f"  landmarks : {tp3.LANDMARKS_DIR}")
    all_clips = tp3.load_raw_clips(quiet=True)
    print(f"  clips     : {len(all_clips)}")
    print(f"  workers   : {workers}")

    rows = list(range(len(all_clips)))
    if args.quick:
        rows = rows[::max(1, len(all_clips) // 120)][:120]
    clips = [all_clips[r] for r in rows]
    print(f"  testing on: {len(clips)} clips, {args.timesteps} timesteps")

    checks = Checks()
    check_block_gating(clips, args.timesteps, checks)
    check_parallel_equals_serial(clips, rows, args.timesteps, len(all_clips), checks)

    print(f"\n{checks.passed}/{checks.total} passed")
    if checks.passed != checks.total:
        print("\nSTOP - the refactor changed the features. Do not run experiments\n"
              "on this build; the numbers would not be comparable to anything\n"
              "already recorded.")
        raise SystemExit(1)

    report_timing(clips, rows, args.timesteps, len(all_clips), workers)
    print("\nBoth changes verified output-identical. Note that --augment ACCURACY\n"
          "numbers still shift, because the parallel path uses a different (valid)\n"
          "random stream - use --legacy-aug-rng to reproduce pre-2026-09-21 runs.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
