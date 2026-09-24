"""
SignBridge - PyTorch sequence model on v3 (body-relative) landmarks.

WHAT THIS IS
------------
train_pytorch.py already asked "does a model that learns temporal structure
on its own beat the RF's hand-picked keyframe summary?" and answered "not at
~964 training rows" (GRU 0.685-0.694 vs RF 0.78). That comparison used v2
data: hands only, already normalized, each hand an independently-timed
sequence with no body reference.

v3 changed what the RF sees - adding pose and re-expressing hand position in
a body-centred frame moved RF val accuracy 0.798 -> 0.895 (see
signbridge-body-relative-features-v3.md). This script asks the same question
for the sequence model: give the GRU/MLP the same richer, LOCATION-AWARE
input, instead of the hands-only one, and see whether the temporal-modeling
comparison changes too.

    RF v3   : 10 keyframes (hand-local + body)         -> engineered -> trees
    GRU/MLP : per-frame   (hand-local + body + pose)    -> learned    -> head

PER-FRAME FEATURE LAYOUT (235 numbers/timestep)
------------------------------------------------
  left hand-local   21 landmarks x 3 axes = 63   (own-wrist frame, as v2)
  right hand-local  21 landmarks x 3 axes = 63
  left hand, body frame   21 landmarks x 2 axes = 42   (shoulder-mid origin,
  right hand, body frame  21 landmarks x 2 axes = 42    shoulder-width scale)
  pose, body frame        11 landmarks x 2 axes = 22   (nose/ears/mouth/
                                                          shoulders/elbows/wrists)
  presence flags: left, right, body                = 3
                                                    -----
                                                     235

The hand-local block reuses mp_hand_detector.normalize_landmark_sequence
(same reference as v2/the RF). The body blocks reuse
mp_body_detector.aspect_correct / body_reference / to_body_frame and
build_feature_vectors_v3.compact / interp_landmarks - imported, not
reimplemented, for the same reason build_feature_vectors_v3.py gives for
importing v2's block: two copies of a feature definition drift apart
silently and invalidate the comparison.

A DELIBERATE DIFFERENCE FROM v2's GRU
--------------------------------------
train_pytorch.py resamples each hand independently, because v2's storage has
no shared clock between them (see that file's docstring). v3's landmarks are
frame-aligned across both hands AND pose (that alignment is the whole reason
extract_landmarks_v3.py re-extracts instead of reusing v2's data - see its
docstring). Throwing that away here to match v2's approach would waste the
one thing v3 data can do that v2 couldn't: let the model see relative timing
between the two hands, and between hand motion and body posture.

So here, ONE shared per-clip timeline is built (length = the clip's frame
count), short detector dropouts are linearly interpolated on that timeline
(same technique build_feature_vectors_v3 uses for pose gaps), and the whole
multi-block frame is resampled to n_steps together. This also fixes a small
correctness gap in the original approach: v2's GRU concatenates only the
frames a hand was detected in, so two detected frames on either side of a
dropout become "adjacent" in the resampled sequence even though they weren't
adjacent in time. Interpolating across the gap on the real timeline first
avoids that.

THE ABLATION (read this before quoting any number)
---------------------------------------------------
Comparing this script's val accuracy against the v2 GRU's changes THREE
things at once: the body block is added, the two hands move onto a shared
clock, and dropouts are interpolated instead of concatenated away. A
difference between those two runs cannot be attributed to any one of them.
build_feature_vectors_v3.py had this problem too and solved it with
--no-body, which reproduces v2's feature set exactly; the same flags exist
here for the same reason:

    python train_pytorch_v3.py                    # full, 235 cols/frame
    python train_pytorch_v3.py --no-body          # control, 128 cols/frame
    python train_pytorch_v3.py --no-hand-local    # body only, 109 cols/frame

--no-body leaves exactly 63 + 63 + 2 = 128 columns per frame, which is the
v2 GRU's feature layout number for number. So the two comparisons separate
cleanly:

    full vs --no-body        -> what the BODY BLOCK is worth
    --no-body vs v2's GRU    -> what the SHARED CLOCK is worth

Each variant saves to its own model file, for the reason train_classifier.py
documents at model_path_for(): one shared filename across variants has
already silently invalidated one comparison in this project.

EVALUATION PROTOCOL (changed - earlier numbers from this script were biased)
-----------------------------------------------------------------------------
The first version of this script early-stopped on val AND reported val. That
number is a maximum over ~100 noisy evaluations of a 124-clip set, so it is
optimistically biased - and the RandomForest it was being compared against
fits once and reports, with no per-epoch selection at all. The comparison was
tilted toward the GRU.

Now a SIGNER-DISJOINT slice of the train split ("dev") drives early stopping
and checkpoint selection, and val is used only to report the selected model.
Val is still evaluated each epoch, but nothing selects on it - that is purely
to measure how large the old bias was.

Three numbers come out of every run, and the differences between them are the
point:

    val@best-dev     the honest number. Quote this one.
    val@best-val     what the old protocol would have reported.
    dev@best-dev     the early-stopping signal itself.

    val@best-val - val@best-dev  = the selection bias, measured directly
                                   (same run, same data, same model - only
                                   the selection criterion differs)

    val@best-val vs the old 0.766 = the cost of the smaller fit set, since
                                    dev is carved out of train

Those two effects are separable exactly because both numbers come out of the
same run, which is why val is still tracked per epoch.

MULTI-SEED
----------
One standard error on 124 val clips is ~4.5pp (~6 clips), and this project has
already measured a 3-clip swing from deleting two training rows. A single run
cannot distinguish an improvement from initialization luck. --seeds N trains N
models on the same split and reports mean and spread, so every later
optimization can be judged against a band rather than a point.

    python train_pytorch_v3.py --seeds 5

The dev split is held FIXED across seeds (--holdout-seed, separate from
--seed) so the spread measures training noise, not split noise.

Per-class results are aggregated across seeds (mean, spread and range per
class) rather than printed from one run. At 3-9 val clips per class a single
seed can swing a class from 0.00 to 1.00 on luck, and this project has
already published one per-class claim that turned out to be exactly that.

Per-epoch printing is suppressed under --seeds > 1 (it would be hundreds of
lines per seed), so normally the only output between "seed N ..." and that
seed's result line is nothing. On a slow config - --augment rebuilds the fit
tensor from raw coordinates every epoch, so a patience-30 run that needs
150+ epochs can run for several minutes - that silence is indistinguishable
from a hung process. --heartbeat (default 15s, 0 disables it) prints a
one-line "epoch N/epochs, Xs/epoch, best dev, epochs since improvement"
update on that interval regardless, so there is never a silent gap longer
than --heartbeat seconds. It does nothing to a single-seed run, which already
gets full per-epoch output.

AUGMENTATION
------------
--augment perturbs the FIT set on the fly, differently every epoch: a
non-uniform time warp, a temporal crop, small in-plane rotation, landmark
jitter and per-hand frame dropout. All of it happens in raw coordinate space
before normalisation, so the body frame and the hand-local block see a
perturbed clip exactly as they would see a real one. augment_v3.py explains
each choice, and explains why global translation, uniform scale and uniform
speed changes are all no-ops against this pipeline.

Dev and val are never augmented. Off by default, so every earlier run
reproduces unchanged.

    python train_pytorch_v3.py --no-hand-local --augment --seeds 5

COST OF AUGMENTATION, AND --workers
------------------------------------
An --augment epoch rebuilds the entire fit set from raw coordinates, and that
rebuild - not the gradient step - is the epoch. A 35k-parameter GRU over 1000
rows is milliseconds; the rebuild is a Python loop over 1000 clips. That is
why an --augment run is ~100x slower per epoch than one without it, and why
the first 5-seed augmented run took ~35 minutes while the unaugmented one
took under a minute.

Two things cut it. clip_to_sequence_v3 now builds only the block that is
going into the feature vector (--no-hand-local used to compute the hand-local
block and discard it, which was roughly half the per-clip cost). And the
rebuild is spread across worker processes:

    --workers 0   os.cpu_count() - 1   (default)
    --workers 1   no pool, build in this process
    --workers N   N processes

The pool is created once and reused by every epoch of every seed. Output is
identical at any worker count - each clip's perturbation is keyed on its row
index alone, so it does not depend on how the work is split. verify_speedup_v3.py
checks exactly that, against the real clips.

ONE REPRODUCIBILITY NOTE. Making the rebuild splittable meant changing how the
augmentation rng is drawn: previously ONE rng was consumed across the clip
list in order, so a clip's perturbation depended on every clip before it. Now
each clip draws from default_rng([seed, epoch, row]). Both are valid streams
and neither is "more random", but they are not the same stream, so --augment
numbers recorded before 2026-09-21 will not reproduce exactly on the new path.
--legacy-aug-rng restores the old stream (single process, necessarily) for
reproducing them. Runs without --augment are bit-identical either way.

INPUT
-----
Reads data/landmarks_v3/*.npz - raw, frame-aligned hands + pose (see
extract_landmarks_v3.py). Only "train" and "val" splits exist there by
design (test is extracted separately, right before the one final
evaluation, and is never touched while iterating) - this script filters to
those two splits explicitly and never loads a "test" row, so there's no way
to accidentally peek even if the test split is extracted into the same
folder later.

SETUP
    pip install torch

RUN
    python train_pytorch_v3.py                  # GRU, 32 timesteps
    python train_pytorch_v3.py --model mlp      # feed-forward control
    python train_pytorch_v3.py --timesteps 48   # more temporal resolution

Compare the val accuracy it prints against:
  - the v3 RandomForest (0.895, models/random_forest_v3.joblib)
  - the v2 GRU (0.685-0.694, from train_pytorch.py / model-comparison doc)
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks_v3")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Imported, not reimplemented - see module docstring.
from build_feature_vectors_v3 import compact, interp_landmarks, MIN_POSE_COVERAGE
from mp_hand_detector import normalize_landmark_sequence
from mp_body_detector import aspect_correct, body_reference, to_body_frame
from augment_v3 import augment_clip

try:
    from build_feature_vectors_v2 import LABEL_MERGES
except Exception:
    LABEL_MERGES = {"HURT": "HURT_PAIN", "PAIN": "HURT_PAIN"}

N_LANDMARKS = 21
N_POSE = 11

N_LOCAL_PER_HAND = N_LANDMARKS * 3   # 63  - hand-local, xyz
N_BODY_PER_HAND = N_LANDMARKS * 2    # 42  - hand in body frame, xy
N_BODY_POSE = N_POSE * 2             # 22  - pose in body frame, xy


def per_frame_for(with_hand_local=True, with_body=True):
    """
    Columns per timestep for a given block selection.

    Blocks are OMITTED, not zeroed, matching what
    build_feature_vectors_v3.py's --no-body does for the RF: a zeroed block
    still costs the model input width and leaves it to discover the columns
    are dead, which is a different experiment from not having them.

    The body-presence flag counts as part of the body block - it is derived
    from whether a pose reference existed, so leaving it in under --no-body
    would leak body information into the control.

        full             235     no-body   128 (= the v2 GRU's layout)
        no-hand-local    109
    """
    n = 0
    if with_hand_local:
        n += N_LOCAL_PER_HAND * 2
    if with_body:
        n += N_BODY_PER_HAND * 2 + N_BODY_POSE + 1   # + body_present
    return n + 2                                      # + left/right presence


PER_FRAME = per_frame_for()          # 235, the full v3 layout


# ---------------------------------------------------------------------------
# Data pipeline (pure numpy - no torch, so it can be tested on its own)
# ---------------------------------------------------------------------------

def resample_seq(seq2d, n_steps):
    """(T, F) -> (n_steps, F) by per-column linear interpolation.

    Generic 2D version of build_feature_vectors_v3's keyframe resampling and
    v2 train_pytorch.py's resample() - same technique, just over however
    many feature columns are concatenated together here.
    """
    n_frames, n_feat = seq2d.shape
    if n_frames == 0:
        return np.zeros((n_steps, n_feat), dtype=np.float32)
    if n_frames == 1:
        return np.repeat(seq2d, n_steps, axis=0).astype(np.float32)
    old_idx = np.linspace(0, n_frames - 1, num=n_frames)
    new_idx = np.linspace(0, n_frames - 1, num=n_steps)
    out = np.empty((n_steps, n_feat), dtype=np.float32)
    for c in range(n_feat):
        out[:, c] = np.interp(new_idx, old_idx, seq2d[:, c])
    return out.astype(np.float32)


def hand_local_track(raw):
    """
    raw: (T, 21, 3) raw MediaPipe coords, NaN where undetected.

    Returns ((T, 21, 3) hand-local normalized track filled across the whole
    clip, presence flag). Detected frames are normalized exactly as v2/the
    RF do (normalize_landmark_sequence on the compacted, gap-free
    sequence); the result is then placed back at its real frame positions
    and short dropouts are interpolated - see module docstring for why this
    differs from v2's GRU.
    """
    valid = np.isfinite(raw[:, 0, 0])
    T = raw.shape[0]
    if not valid.any():
        return np.zeros((T, N_LANDMARKS, 3), dtype=np.float32), 0.0

    normed = normalize_landmark_sequence(raw[valid])
    track = np.full((T, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    track[valid] = normed
    track = interp_landmarks(track, valid)
    return track, 1.0


def hand_presence(raw):
    """
    The presence flag alone, without building the normalized track.

    hand_local_track returns (track, presence) and costs a
    normalize_landmark_sequence + interp_landmarks across the whole clip.
    Under --no-hand-local the track is discarded, but the FLAG is still part
    of the feature vector (the +2 in per_frame_for), so the flag is read
    directly here instead. Same rule hand_local_track uses: present iff any
    frame has a finite wrist.
    """
    return 1.0 if np.isfinite(raw[:, 0, 0]).any() else 0.0


def hand_body_track(raw, frame_w, frame_h, origin, scale, T):
    """(T, 21, 2) hand position in the body frame, gaps interpolated."""
    valid = np.isfinite(raw[:, 0, 0])
    if not valid.any():
        return np.zeros((T, N_LANDMARKS, 2), dtype=np.float32)
    hand_xy = aspect_correct(raw[:, :, :2], frame_w, frame_h)
    body = to_body_frame(hand_xy, origin, scale)   # NaN propagates from raw
    return interp_landmarks(body, valid)


def clip_to_sequence_v3(d, n_steps, with_hand_local=True, with_body=True):
    """
    One clip's .npz -> ((n_steps, per_frame_for(...)) sequence, l_present,
    r_present, body_present).

    body_present is still reported under --no-body (the pose reference is
    cheap, and load_dataset prints a body-coverage count for every run), but
    the three body-frame TRACKS are only built when the body block is
    actually going into the feature vector.

    Both blocks are computed only when they are used. An earlier version
    built each block unconditionally and then let the with_* flags decide
    what to concatenate, which meant --no-hand-local still paid for two
    normalize_landmark_sequence + interp_landmarks passes per clip and threw
    them away - roughly half the per-clip cost, on every clip, on every epoch
    of an --augment run. Skipping the unused block cannot change the output:
    the discarded arrays never reached `blocks`.
    """
    left_raw = d["left_hand"]
    right_raw = d["right_hand"]
    pose_raw = d["pose"] if "pose" in d else None
    frame_w = int(d["frame_w"]) if "frame_w" in d else 0
    frame_h = int(d["frame_h"]) if "frame_h" in d else 0
    T = left_raw.shape[0]

    if with_hand_local:
        left_local, l_present = hand_local_track(left_raw)
        right_local, r_present = hand_local_track(right_raw)
    else:
        l_present = hand_presence(left_raw)
        r_present = hand_presence(right_raw)

    origin_ok = False
    if pose_raw is not None and pose_raw.shape[0] > 0:
        pose_xy = aspect_correct(pose_raw[:, :, :2], frame_w, frame_h)
        origin, scale, pose_valid = body_reference(pose_xy)
        origin_ok = (pose_valid.sum() >= 2
                     and pose_valid.mean() >= MIN_POSE_COVERAGE
                     and scale > 0)
    body_present = 1.0 if origin_ok else 0.0

    if with_body:
        if origin_ok:
            pose_xy_filled = interp_landmarks(pose_xy, pose_valid)
            pose_body = to_body_frame(pose_xy_filled, origin, scale)      # (T, 11, 2)
            left_body = hand_body_track(left_raw, frame_w, frame_h, origin, scale, T)
            right_body = hand_body_track(right_raw, frame_w, frame_h, origin, scale, T)
        else:
            pose_body = np.zeros((T, N_POSE, 2), dtype=np.float32)
            left_body = np.zeros((T, N_LANDMARKS, 2), dtype=np.float32)
            right_body = np.zeros((T, N_LANDMARKS, 2), dtype=np.float32)

    blocks, flag_values = [], [l_present, r_present]
    if with_hand_local:
        blocks += [left_local.reshape(T, -1), right_local.reshape(T, -1)]
    if with_body:
        blocks += [left_body.reshape(T, -1), right_body.reshape(T, -1),
                   pose_body.reshape(T, -1)]
        flag_values.append(body_present)

    flat = np.concatenate(blocks, axis=1).astype(np.float32)               # (T, F - n_flags)
    flat = resample_seq(flat, n_steps)
    flags = np.tile(np.array(flag_values, dtype=np.float32), (n_steps, 1))
    seq = np.concatenate([flat, flags], axis=1).astype(np.float32)
    assert seq.shape[1] == per_frame_for(with_hand_local, with_body), seq.shape
    return seq, l_present, r_present, body_present


def load_raw_clips(landmarks_dir=LANDMARKS_DIR, quiet=False):
    """
    Read every train/val clip's RAW arrays into memory once.

    Returns a list of dicts keyed exactly like the .npz files
    (left_hand / right_hand / pose / frame_w / frame_h) plus label, split and
    participant - so a clip dict can be handed straight to
    clip_to_sequence_v3() in place of an npz.

    Raw arrays are kept rather than finished feature vectors because
    augmentation has to happen in raw coordinate space, before normalisation,
    and has to be redone every epoch (see augment_v3.py). ~70 MB for the full
    train+val set.
    """
    files = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
    if not files:
        raise SystemExit(f"ERROR: no .npz files in {landmarks_dir}. "
                         f"Run extract_landmarks_v3.py --splits train,val first.")

    clips = []
    n_no_hand = n_test_skipped = 0
    for path in files:
        d = np.load(path, allow_pickle=True)
        split = str(d["split"])
        if split not in ("train", "val"):
            n_test_skipped += 1
            continue

        left, right = d["left_hand"], d["right_hand"]
        # Same rule as the rest of the pipeline: a clip with no hand on any
        # frame is not usable.
        if not (np.isfinite(left[:, 0, 0]).any() or np.isfinite(right[:, 0, 0]).any()):
            n_no_hand += 1
            continue

        raw_label = str(d["label"])
        clips.append({
            "left_hand": left,
            "right_hand": right,
            "pose": d["pose"] if "pose" in d else None,
            "frame_w": int(d["frame_w"]) if "frame_w" in d else 0,
            "frame_h": int(d["frame_h"]) if "frame_h" in d else 0,
            "label": LABEL_MERGES.get(raw_label, raw_label),
            "split": split,
            "participant": str(d["participant"]),
        })

    if not quiet:
        print(f"  {n_no_hand} clips skipped (no hand ever detected)"
              + (f"; {n_test_skipped} test-split clips skipped" if n_test_skipped else ""))
    return clips


def augment_one(clip, n_steps, with_hand_local, with_body, rng, strength):
    """Perturb one clip in raw space, then build its sequence."""
    left, right, pose = augment_clip(
        clip["left_hand"], clip["right_hand"], clip["pose"],
        clip["frame_w"], clip["frame_h"], rng, strength)
    d = dict(clip, left_hand=left, right_hand=right, pose=pose)
    return clip_to_sequence_v3(d, n_steps, with_hand_local=with_hand_local,
                               with_body=with_body)


def aug_rng_key(stream, seed, epoch, row):
    """
    The rng key for one clip's perturbation in one epoch.

    stream 0 returns [seed, epoch, row] — byte-for-byte the key used before
    --aug-stream existed — so every number already recorded on the default
    stream stays reproducible. A non-zero stream prepends the stream id,
    which is a different draw of the same experiment.

    Independent streams matter more than they look. Running one augmented
    config five times shares ONE stream across those five seeds, so a result
    can look perfectly stable across seeds and still be an artifact of the
    stream — which is exactly how four per-class claims got into the ablation
    doc and straight back out again. Cheap extra streams are what make the
    "two streams before you believe it" rule practical rather than
    aspirational.
    """
    return [seed, epoch, row] if stream == 0 else [stream, seed, epoch, row]


def build_X(clips, n_steps, with_hand_local=True, with_body=True,
            aug_rng=None, strength=1.0, aug_key=None, clip_ids=None):
    """
    Feature tensor for a list of raw clips.

    Augmentation perturbs each clip in RAW coordinate space before the feature
    build, so normalisation, the body frame and the resampling all see the
    perturbed clip exactly as they would see a real one. Two ways to drive it:

      aug_rng   ONE rng consumed across the whole clip list in order (LEGACY).
                Reproducible only at a fixed clip order and a single process,
                because clip i's perturbation depends on how much randomness
                clips 0..i-1 drew. This is the stream every --augment number
                recorded before 2026-09-21 was produced on; kept so those
                runs can be reproduced exactly.

      aug_key   (stream, seed, epoch), with clip_ids giving each clip's STABLE
                id. Each clip gets default_rng(aug_rng_key(...)), so its
                perturbation depends on nothing but its own id. That makes the
                result independent of clip order, of how the work is split,
                and of how many worker processes run it - which is what lets
                build_X_parallel produce bit-identical output to this
                function. Use clip ids that are stable across splits (the row
                index into the full clip list), not positions within a subset.

    Passing neither leaves the clips unaugmented.
    """
    if aug_rng is not None and aug_key is not None:
        raise ValueError("pass aug_rng or aug_key, not both")
    if aug_key is not None and clip_ids is None:
        raise ValueError("aug_key needs clip_ids (stable per-clip ids)")

    seqs, n_no_body = [], 0
    for j, c in enumerate(clips):
        if aug_rng is not None:
            seq, _, _, body_present = augment_one(
                c, n_steps, with_hand_local, with_body, aug_rng, strength)
        elif aug_key is not None:
            stream, seed, epoch = aug_key
            rng = np.random.default_rng(
                aug_rng_key(stream, seed, epoch, int(clip_ids[j])))
            seq, _, _, body_present = augment_one(
                c, n_steps, with_hand_local, with_body, rng, strength)
        else:
            seq, _, _, body_present = clip_to_sequence_v3(
                c, n_steps, with_hand_local=with_hand_local, with_body=with_body)
        n_no_body += (body_present == 0.0)
        seqs.append(seq)
    return np.stack(seqs), n_no_body


# ---------------------------------------------------------------------------
# Parallel feature building
#
# An --augment epoch rebuilds the whole fit set from raw coordinates, and that
# rebuild - not the gradient step - is essentially the entire cost of the
# epoch: a 35k-parameter GRU over 1000 rows is milliseconds, while the rebuild
# is a single-threaded Python loop over 1000 clips. Handing the loop to a pool
# of processes is the difference between a ~35-minute 5-seed run and a few
# minutes, which is what makes sweeping --aug-strength or ablating the
# individual perturbations practical at all.
#
# Three things this has to get right:
#
#   1. The pool is built ONCE and reused for every epoch of every seed.
#      Windows spawns fresh interpreters rather than forking, and each one
#      re-imports this module (and mediapipe underneath it), so a pool per
#      epoch would cost far more than it saves.
#   2. Workers load the clips themselves in the initializer rather than
#      receiving them through initargs. The raw arrays are ~70 MB; pickling
#      them to every worker on every startup is slower than each worker
#      reading the same files, which the OS page cache serves from memory
#      after the first one.
#   3. Work is addressed by ROW INDEX into the full clip list, and the rng key
#      is built from that same index, so the split into chunks has no effect
#      on the output. --workers 8 and --workers 1 must agree exactly.
# ---------------------------------------------------------------------------

_POOL_CLIPS = None


def _pool_init(landmarks_dir, expected_n):
    """Runs once per worker process: load the clips this worker will serve."""
    global _POOL_CLIPS
    _POOL_CLIPS = load_raw_clips(landmarks_dir, quiet=True)
    if len(_POOL_CLIPS) != expected_n:
        raise RuntimeError(
            f"worker loaded {len(_POOL_CLIPS)} clips, parent has {expected_n} - "
            f"the landmarks folder changed under a running job")


def _build_chunk(task):
    """Worker entry point: build the sequences for one chunk of row indices."""
    rows, n_steps, with_hand_local, with_body, stream, seed, epoch, strength = task
    out = []
    for r in rows:
        clip = _POOL_CLIPS[r]
        rng = np.random.default_rng(aug_rng_key(stream, seed, epoch, int(r)))
        seq, _, _, body_present = augment_one(
            clip, n_steps, with_hand_local, with_body, rng, strength)
        out.append((r, seq, body_present))
    return out


def build_X_parallel(pool, rows, n_steps, with_hand_local, with_body,
                     stream, seed, epoch, strength, n_chunks):
    """
    Augmented feature tensor for `rows` (row indices into the full clip list),
    built across `pool`. Bit-identical to

        build_X(clips_for(rows), ..., aug_key=(stream, seed, epoch),
                clip_ids=rows)

    because both key each clip's rng on its row index alone. Rows come back in
    the order given regardless of which worker finished first.
    """
    rows = list(rows)
    n_chunks = max(1, min(n_chunks, len(rows)))
    bounds = np.linspace(0, len(rows), n_chunks + 1).astype(int)
    tasks = [(rows[a:b], n_steps, with_hand_local, with_body,
              stream, seed, epoch, strength)
             for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    by_row = {}
    for chunk in pool.imap_unordered(_build_chunk, tasks):
        for r, seq, body_present in chunk:
            by_row[r] = (seq, body_present)

    seqs = [by_row[r][0] for r in rows]
    n_no_body = sum(by_row[r][1] == 0.0 for r in rows)
    return np.stack(seqs), n_no_body


def load_dataset(n_steps, landmarks_dir=LANDMARKS_DIR,
                 with_hand_local=True, with_body=True):
    """
    Returns X (N, n_steps, per_frame), y (N,) int labels, splits (N,) str,
    participants (N,) str, the sorted class list, and the raw clip list. Only
    "train" and "val" rows are loaded - v3's test split is extracted
    separately and never lands in this folder while iterating, but this filter
    makes that guarantee explicit here too.

    participants is returned because the early-stopping set has to be carved
    out by signer, not by clip - see make_dev_split(). The raw clips are
    returned so the fit set can be re-augmented every epoch without
    re-reading disk.
    """
    clips = load_raw_clips(landmarks_dir)
    X, n_no_body = build_X(clips, n_steps, with_hand_local, with_body)

    labels = [c["label"] for c in clips]
    classes = sorted(set(labels))
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for l in labels], dtype=np.int64)
    splits = np.array([c["split"] for c in clips])
    participants = np.array([c["participant"] for c in clips])
    print(f"  {n_no_body} clips have no usable body reference "
          f"(hand-local only, body block zeroed)")
    return X, y, splits, participants, classes, clips


# ---------------------------------------------------------------------------
# The early-stopping split
# ---------------------------------------------------------------------------

def base_participant(p):
    """
    'P11_mirror' -> 'P11'.

    mirror_augment_v3.py stores a mirrored clip's participant as
    "<original>_mirror". That is the SAME HUMAN as the original, so grouping
    on the raw string would happily put P11 in the fit set and P11_mirror in
    the early-stopping set - reintroducing, one level down, exactly the
    signer leakage the signer-disjoint splits exist to prevent.
    """
    return p[:-len("_mirror")] if p.endswith("_mirror") else p


def make_dev_split(participants, train_mask, frac, seed):
    """
    Carve a signer-disjoint early-stopping set out of the train split.
    Returns (fit_mask, dev_mask) over the full row index.

    Why signer-disjoint rather than a random slice of clips: early stopping
    selects the epoch that generalizes best to whatever the dev set measures.
    A dev set sharing signers with fit would measure "generalizes to new clips
    of people we trained on", which keeps improving for longer than
    "generalizes to new people" does - so it would select a more overfit model
    than val and test will reward. The dev set has to be the same KIND of
    held-out set as val for the early-stopping signal to point the right way.
    """
    people = np.array([base_participant(p) for p in participants])
    train_people = sorted(set(people[train_mask]))
    if len(train_people) < 3:
        raise SystemExit(f"ERROR: only {len(train_people)} signers in train - "
                         f"cannot carve a signer-disjoint dev set.")

    rng = np.random.default_rng(seed)
    target = frac * int(train_mask.sum())

    dev_people, taken = set(), 0
    for i in rng.permutation(len(train_people)):
        if taken >= target:
            break
        person = train_people[i]
        dev_people.add(person)
        taken += int(((people == person) & train_mask).sum())

    dev_mask = train_mask & np.isin(people, sorted(dev_people))
    fit_mask = train_mask & ~dev_mask

    if not dev_mask.any() or not fit_mask.any():
        raise SystemExit("ERROR: dev split left one side empty - check --dev-frac.")
    overlap = set(people[fit_mask]) & set(people[dev_mask])
    assert not overlap, f"signer leak between fit and dev: {sorted(overlap)}"
    return fit_mask, dev_mask


# ---------------------------------------------------------------------------
# Models + training (torch) - same architectures as train_pytorch.py
# ---------------------------------------------------------------------------

def build_model(kind, n_steps, n_classes, hidden, dropout, torch, nn, per_frame):
    if kind == "gru":
        class GRUClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(per_frame, hidden, num_layers=1, batch_first=True)
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
                nn.Linear(n_steps * per_frame, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            return self.net(x)
    return MLPClassifier()


def train_one_seed(tensors, classes, args, seed, per_frame, torch, nn, verbose,
                   aug=None, heartbeat=0.0):
    """
    Train one model. Selection is on DEV only; val is evaluated per epoch
    purely to measure the bias the old protocol carried (see module docstring).

    aug, when given, is {"clips", "with_hand_local", "with_body", "strength"}
    for the FIT set. The fit tensor is then rebuilt from raw every epoch with a
    fresh rng, so no two epochs see the same perturbation. Dev and val are
    never augmented: dev is the selection signal and val is the reported
    number, and perturbing either would change what is being measured rather
    than what is being learned.

    heartbeat: with verbose off (the --seeds > 1 case), the per-epoch print
    is suppressed to keep multi-seed output readable - but on a slow config
    (--augment rebuilds the fit tensor from raw every epoch) a seed can run
    silently for minutes. That is indistinguishable from a hang. When
    heartbeat > 0, a one-line progress print fires every `heartbeat` seconds
    of wall clock regardless of verbose, so there is never a silent gap
    longer than that.

    Returns a dict of metrics plus the selected state_dict and its val
    predictions.
    """
    Xfit, yfit, Xdev, ydev, Xval, yval = tensors

    torch.manual_seed(seed)
    np.random.seed(seed)

    # class_weight='balanced', computed on the FIT set only - dev is held out
    # and must not influence the loss.
    counts = np.bincount(yfit.numpy(), minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(len(yfit) / (len(classes) * np.maximum(counts, 1)))

    model = build_model(args.model, args.timesteps, len(classes),
                        args.hidden, args.dropout, torch, nn, per_frame)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dev, best_state, since_best = -1.0, None, 0
    val_at_best_dev = 0.0
    best_val_any_epoch = 0.0       # what the OLD protocol would have reported
    epochs_run = 0

    t_start = time.time()
    last_beat = t_start

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch

        if aug is None:
            Xep = Xfit
        elif aug["pool"] is not None:
            Xaug, _ = build_X_parallel(
                aug["pool"], aug["rows"], args.timesteps,
                aug["with_hand_local"], aug["with_body"],
                aug["stream"], seed, epoch, aug["strength"], aug["n_chunks"])
            Xep = torch.tensor(Xaug)
        elif aug["legacy_rng"]:
            # One rng consumed across the clip list in order - the stream every
            # --augment number recorded before 2026-09-21 was produced on.
            Xaug, _ = build_X(aug["clips"], args.timesteps,
                              aug["with_hand_local"], aug["with_body"],
                              aug_rng=np.random.default_rng([seed, epoch]),
                              strength=aug["strength"])
            Xep = torch.tensor(Xaug)
        else:
            # Per-clip rng keyed on the row index: same output the pool
            # produces, just built in this process.
            Xaug, _ = build_X(aug["clips"], args.timesteps,
                              aug["with_hand_local"], aug["with_body"],
                              aug_key=(aug["stream"], seed, epoch),
                              clip_ids=aug["rows"], strength=aug["strength"])
            Xep = torch.tensor(Xaug)

        model.train()
        perm = torch.randperm(len(Xep))
        for i in range(0, len(Xep), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xep[b]), yfit[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_acc = (model(Xdev).argmax(1) == ydev).float().mean().item()
            val_acc = (model(Xval).argmax(1) == yval).float().mean().item()

        best_val_any_epoch = max(best_val_any_epoch, val_acc)

        if dev_acc > best_dev:
            best_dev, since_best = dev_acc, 0
            val_at_best_dev = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            with torch.no_grad():
                fit_acc = (model(Xfit).argmax(1) == yfit).float().mean().item()
            print(f"  epoch {epoch:>3}  fit {fit_acc:.3f}  dev {dev_acc:.3f}"
                  f"  (best dev {best_dev:.3f})  [val {val_acc:.3f}, not selected on]")

        elif heartbeat and not verbose:
            now = time.time()
            if now - last_beat >= heartbeat or epoch == 1:
                elapsed = now - t_start
                print(f"    seed {seed}: epoch {epoch}/{args.epochs}  "
                      f"({elapsed:.0f}s elapsed, ~{elapsed / epoch:.1f}s/epoch)  "
                      f"best dev {best_dev:.3f}  ({since_best}/{args.patience} "
                      f"epochs since improvement)", flush=True)
                last_beat = now

        if since_best >= args.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"({args.patience} epochs without dev improvement)")
            elif heartbeat:
                print(f"    seed {seed}: early stop at epoch {epoch} "
                      f"({args.patience} epochs without dev improvement)", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fit_acc = (model(Xfit).argmax(1) == yfit).float().mean().item()
        val_logits = model(Xval)
        val_pred = val_logits.argmax(1).numpy()
        # Softmax at the selected (best-dev) checkpoint, for --save-probs /
        # ensemble_v3.py. Computed after training has finished and in eval
        # mode, so it draws no randomness and cannot change any number above:
        # argmax(val_prob) is exactly the prediction val@best-dev was read from.
        val_prob = torch.softmax(val_logits, dim=1).numpy()
        dev_prob = torch.softmax(model(Xdev), dim=1).numpy()

    return {
        "seed": seed,
        "fit_acc": fit_acc,
        "dev_best": best_dev,
        "val_at_best_dev": val_at_best_dev,
        "val_at_best_val": best_val_any_epoch,
        "epochs": epochs_run,
        "state": best_state,
        "val_pred": val_pred,
        "val_prob": val_prob,
        "dev_prob": dev_prob,
    }


def summarize(values):
    """mean and population spread, printed as 'mean +/- sd'."""
    a = np.asarray(values, dtype=np.float64)
    return a.mean(), a.std()


def per_class_arrays(results, y_val, classes):
    """
    (recall, f1, support) with recall/f1 shaped (n_seeds, n_classes).

    Split out of print_per_class so a sweep driver can compare per-class
    behaviour ACROSS runs rather than only print one run's table. That
    comparison is the point: section 4 of the ablation doc retracted four
    per-class claims that were stable across seeds within a single
    augmentation stream and evaporated on the second stream.
    """
    from sklearn.metrics import precision_recall_fscore_support

    rec = np.zeros((len(results), len(classes)))
    f1 = np.zeros((len(results), len(classes)))
    for i, r in enumerate(results):
        _, rec[i], f1[i], _ = precision_recall_fscore_support(
            y_val, r["val_pred"], labels=range(len(classes)), zero_division=0)
    return rec, f1, np.bincount(y_val, minlength=len(classes))


def print_per_class(results, y_val, classes, n_seeds):
    """
    Per-class recall and f1 aggregated ACROSS seeds.

    A single seed's per-class report is close to unreadable at this scale:
    every class has 3-9 val clips, so one clip is 11-33 points of recall and
    a class can swing from 0.00 to 1.00 on luck alone. This project already
    published one per-class claim (HUNGRY recall 0.00 under body-only) that
    was a single-seed artifact and did not reproduce. Reporting mean, spread
    and range across seeds makes that failure mode visible instead of
    inviting it.

    Sorted worst-f1 first, since the point of the table is to find the
    classes that need work.
    """
    rec, f1, support = per_class_arrays(results, y_val, classes)

    print(f"\nPer-class on val, aggregated over {n_seeds} seeds (worst f1 first):")
    print(f"{'class':<13}{'n':>3}  {'1 clip':>7}  {'recall':>15}  "
          f"{'range':>13}  {'f1':>15}")
    for c in np.argsort(f1.mean(axis=0)):
        n = support[c]
        grain = (1.0 / n) if n else float("nan")
        print(f"{classes[c]:<13}{n:>3}  {grain:>7.2f}  "
              f"{rec[:, c].mean():>7.2f} +/-{rec[:, c].std():<5.2f}  "
              f"{rec[:, c].min():>5.2f}-{rec[:, c].max():<5.2f}  "
              f"{f1[:, c].mean():>7.2f} +/-{f1[:, c].std():<5.2f}")

    unstable = int((rec.std(axis=0) > 0.15).sum())
    print(f"\n  '1 clip' is how much one val clip moves that class's recall.")
    print(f"  {unstable}/{len(classes)} classes swing more than 0.15 recall across "
          f"seeds - treat those as unmeasured, not as findings.")
    print("  No per-class claim from this table is safe unless it is large "
          "AND its spread is small.")
    print("  Small spread here is NOT sufficient either: these seeds share one")
    print("  augmentation stream. Confirm on a second --aug-stream before "
          "quoting anything.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "mlp"], default="gru")
    ap.add_argument("--no-body", action="store_true",
                    help="hand-local only (128 cols/frame) - the control for "
                         "measuring what the body block is worth")
    ap.add_argument("--no-hand-local", action="store_true",
                    help="body block only - does location alone carry the signal")
    ap.add_argument("--timesteps", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42,
                    help="first seed; --seeds N runs seed, seed+1, ... seed+N-1")
    ap.add_argument("--seeds", type=int, default=1,
                    help="how many seeds to train; reports mean +/- spread")
    ap.add_argument("--dev-frac", type=float, default=0.18,
                    help="fraction of train rows held out (by signer) for early "
                         "stopping. Never touches val.")
    ap.add_argument("--augment", action="store_true",
                    help="on-the-fly augmentation of the FIT set only (see "
                         "augment_v3.py). Off by default so earlier runs reproduce.")
    ap.add_argument("--aug-strength", type=float, default=1.0,
                    help="scales every augmentation magnitude; 0 disables them")
    ap.add_argument("--holdout-seed", type=int, default=7,
                    help="seed for the dev split only. Deliberately separate from "
                         "--seed so the dev set stays FIXED across seeds and the "
                         "reported spread is training noise, not split noise.")
    ap.add_argument("--aug-stream", type=int, default=0,
                    help="which augmentation random stream to draw from. Stream 0 "
                         "is the default and reproduces every number recorded on "
                         "it. Any other integer is an independent draw of the same "
                         "experiment - use one to confirm a result before quoting "
                         "it, since five seeds inside ONE stream can look stable "
                         "and still be a stream artifact. Ignored without --augment.")
    ap.add_argument("--no-save", action="store_true",
                    help="do not write the model checkpoint. Sweeps and probes "
                         "must pass this: every run otherwise writes the same "
                         "models/pytorch_v3<variant>_<model>.pt, so an 8-run "
                         "sweep would leave whichever config happened to run "
                         "last sitting in the canonical checkpoint. One shared "
                         "filename across variants has silently invalidated a "
                         "comparison in this project before.")
    ap.add_argument("--json-out", type=str, default=None,
                    help="write the run's config, per-seed metrics and per-class "
                         "arrays to this path as JSON, for a sweep driver to "
                         "collate. Printed output is unchanged.")
    ap.add_argument("--save-seeds", type=str, default=None,
                    help="write EVERY seed's best-dev weights (not just the "
                         "best one) to this .pt, with the config, class list "
                         "and dev/val rows. Input to build_ensemble_bundle.py, "
                         "which averages all of them - averaging the 5 seeds "
                         "measured ~4.5 val clips better than one seed. "
                         "Independent of --no-save, which only governs the "
                         "single-model checkpoint.")
    ap.add_argument("--save-probs", type=str, default=None,
                    help="write every seed's softmax on dev and val (at that "
                         "seed's best-dev checkpoint) to this .npz, with the row "
                         "indices, labels and signers needed to line them up "
                         "with another model. Input to ensemble_v3.py. Training "
                         "and printed results are unchanged.")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch intra-op threads (0 = leave torch's default). The "
                         "gradient step is now the larger half of an --augment "
                         "epoch, and at this tensor size (batch 32 x 32 steps x "
                         "109 features through a 64-unit GRU) thread "
                         "synchronisation can cost more than it saves, so a small "
                         "number is worth measuring.")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes used to rebuild the augmented fit set each "
                         "epoch. 0 = os.cpu_count() - 1, 1 = no pool (build in "
                         "this process). Only affects --augment runs; the "
                         "output is identical at any worker count.")
    ap.add_argument("--legacy-aug-rng", action="store_true",
                    help="use the pre-2026-09-21 augmentation stream: ONE rng "
                         "consumed across the clip list in order. Forces "
                         "--workers 1, since that stream cannot be split across "
                         "processes. Only needed to reproduce an --augment "
                         "number recorded before that date.")
    ap.add_argument("--heartbeat", type=float, default=15.0,
                    help="seconds between progress lines during a --seeds > 1 run "
                         "(per-epoch output is otherwise suppressed there to keep "
                         "multi-seed output short - see train_one_seed). A slow "
                         "config, especially --augment, can otherwise go minutes "
                         "with no output at all, which looks identical to a hang. "
                         "0 disables heartbeats and restores the old silent "
                         "behaviour.")
    args = ap.parse_args()
    t_run_start = time.time()

    with_hand_local = not args.no_hand_local
    with_body = not args.no_body
    if not (with_hand_local or with_body):
        raise SystemExit("ERROR: --no-body and --no-hand-local together leave no features.")
    per_frame = per_frame_for(with_hand_local, with_body)
    variant = "" if (with_hand_local and with_body) else \
              ("_handlocal" if with_hand_local else "_bodyonly")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("ERROR: PyTorch not installed. Run:  pip install torch")

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    from sklearn.metrics import classification_report

    print(f"Loading v3 landmarks from {LANDMARKS_DIR} ...")
    print(f"blocks: hand-local={with_hand_local}, body={with_body}")
    X, y, splits, participants, classes, clips = load_dataset(
        args.timesteps, with_hand_local=with_hand_local, with_body=with_body)
    print(f"Loaded {len(X)} clips, {len(classes)} classes, "
          f"sequences of {args.timesteps} steps x {per_frame} features")

    train_mask, val_mask = splits == "train", splits == "val"
    fit_mask, dev_mask = make_dev_split(participants, train_mask,
                                        args.dev_frac, args.holdout_seed)

    dev_people = sorted({base_participant(p) for p in participants[dev_mask]})
    n_classes_in_dev = len(set(y[dev_mask]))
    print(f"\nfit: {int(fit_mask.sum())} rows | dev: {int(dev_mask.sum())} rows "
          f"({len(dev_people)} signers held out for early stopping) | "
          f"val: {int(val_mask.sum())} rows")
    print(f"dev signers: {', '.join(dev_people)}")
    print(f"dev covers {n_classes_in_dev}/{len(classes)} classes")
    if n_classes_in_dev < 0.8 * len(classes):
        print("  WARNING: dev covers few classes - try a different --holdout-seed")
    print("val is NOT used for selection; test was never loaded.")

    tensors = (torch.tensor(X[fit_mask]), torch.tensor(y[fit_mask]),
               torch.tensor(X[dev_mask]), torch.tensor(y[dev_mask]),
               torch.tensor(X[val_mask]), torch.tensor(y[val_mask]))

    aug = None
    if args.augment and args.aug_strength > 0:
        fit_rows = [int(i) for i in np.flatnonzero(fit_mask)]
        fit_clips = [clips[i] for i in fit_rows]
        assert len(fit_clips) == int(fit_mask.sum()), "fit clip/row mismatch"
        aug = {"clips": fit_clips, "rows": fit_rows,
               "with_hand_local": with_hand_local, "with_body": with_body,
               "strength": args.aug_strength, "stream": args.aug_stream,
               "legacy_rng": args.legacy_aug_rng,
               "pool": None, "n_chunks": 1}
        if args.legacy_aug_rng and args.aug_stream != 0:
            raise SystemExit("ERROR: --legacy-aug-rng is a single fixed stream; "
                             "--aug-stream does not apply to it.")

    probe = build_model(args.model, args.timesteps, len(classes),
                        args.hidden, args.dropout, torch, nn, per_frame)
    print(f"\nmodel: {args.model.upper()}, "
          f"{sum(p.numel() for p in probe.parameters()):,} parameters")
    print(f"augmentation: {'ON (strength %.2f, stream %d, fit set only)' % (args.aug_strength, args.aug_stream) if aug else 'off'}")
    if args.torch_threads > 0:
        print(f"torch threads: {args.torch_threads}")

    # The pool only ever rebuilds the augmented fit set, so it is created once
    # here and reused by every epoch of every seed - never per epoch, which on
    # Windows would re-import this module (and mediapipe) each time.
    n_workers = 1
    if aug is not None:
        if args.legacy_aug_rng:
            n_workers = 1
            if args.workers not in (0, 1):
                print("  --legacy-aug-rng forces --workers 1 (that stream is "
                      "sequential by construction)")
        elif args.workers:
            n_workers = args.workers
        else:
            # Capped rather than simply cpu_count() - 1. Every worker is a
            # spawned interpreter that imports mediapipe and loads its own copy
            # of the clips; 13 of them took a 16 GB machine to 97% memory, and
            # paging would cost more than the extra workers earn. The rebuild is
            # also no longer the bottleneck (it is ~0.25s of a ~0.87s epoch), so
            # the marginal worker buys little. Override with --workers.
            n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            aug["pool"] = ctx.Pool(processes=n_workers,
                                   initializer=_pool_init,
                                   initargs=(LANDMARKS_DIR, len(clips)))
            aug["n_chunks"] = n_workers * 4
            print(f"augmented rebuild: {n_workers} worker processes "
                  f"(identical output at any worker count; --workers 1 to disable)")
        else:
            print("augmented rebuild: single process"
                  + (" (legacy rng stream)" if args.legacy_aug_rng else ""))

    print(f"running {args.seeds} seed(s) on a fixed split\n")

    results = []
    try:
        for i in range(args.seeds):
            seed = args.seed + i
            if args.seeds > 1:
                print(f"seed {seed} ...", flush=True)
            r = train_one_seed(tensors, classes, args, seed, per_frame,
                               torch, nn, verbose=(args.seeds == 1), aug=aug,
                               heartbeat=args.heartbeat if args.seeds > 1 else 0.0)
            results.append(r)
            if args.seeds > 1:
                print(f"  seed {seed} done: dev {r['dev_best']:.3f}  "
                      f"val@best-dev {r['val_at_best_dev']:.3f}  "
                      f"(val@best-val {r['val_at_best_val']:.3f})  {r['epochs']} epochs\n")
    finally:
        if aug is not None and aug["pool"] is not None:
            aug["pool"].close()
            aug["pool"].join()

    print("\n" + "=" * 72)
    print(f"RESULTS - {args.model.upper()}, hand-local={with_hand_local}, body={with_body}")
    print("=" * 72)
    print(f"{'seed':>6}  {'fit':>6}  {'dev':>6}  {'val@best-dev':>13}  {'val@best-val':>13}")
    for r in results:
        print(f"{r['seed']:>6}  {r['fit_acc']:>6.3f}  {r['dev_best']:>6.3f}  "
              f"{r['val_at_best_dev']:>13.3f}  {r['val_at_best_val']:>13.3f}")

    honest_m, honest_s = summarize([r["val_at_best_dev"] for r in results])
    biased_m, biased_s = summarize([r["val_at_best_val"] for r in results])
    n_val = int(val_mask.sum())

    print(f"\n  val@best-dev  {honest_m:.3f} +/- {honest_s:.3f}   <- THE NUMBER TO QUOTE")
    print(f"  val@best-val  {biased_m:.3f} +/- {biased_s:.3f}   <- old protocol, optimistic")
    print(f"  selection bias {biased_m - honest_m:+.3f} "
          f"({(biased_m - honest_m) * n_val:+.1f} clips) - what early-stopping on val bought itself")
    print(f"  seed spread on the honest number: {honest_s * n_val:.1f} clips "
          f"(1 SE on {n_val} clips is ~6)")
    print("\n  reference: RF v3 0.895 | RF v2 0.774-0.798 | v2 GRU 0.685-0.694")
    print("  the old 0.766 from this script is a val@best-val number on a larger")
    print("  fit set - compare it to val@best-val above, not to val@best-dev.")

    # The shipped model is chosen by DEV, never by val - picking the
    # best-val seed would smuggle the same bias back in one level up.
    best = max(results, key=lambda r: r["dev_best"])

    if args.seeds == 1:
        print(f"\nPer-class report on val (seed {best['seed']}):")
        print(classification_report(y[val_mask], best["val_pred"],
                                    labels=range(len(classes)),
                                    target_names=classes, zero_division=0))
        print("WARNING: per-class numbers from a single seed are not quotable. "
              "Run --seeds 5 for the aggregated table.")
    else:
        print_per_class(results, y[val_mask], classes, args.seeds)

    if args.json_out:
        rec, f1, support = per_class_arrays(results, y[val_mask], classes)
        payload = {
            "config": {
                "model": args.model, "timesteps": args.timesteps,
                "hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
                "batch": args.batch, "epochs": args.epochs,
                "patience": args.patience, "seed": args.seed,
                "seeds": args.seeds, "dev_frac": args.dev_frac,
                "holdout_seed": args.holdout_seed,
                "with_hand_local": with_hand_local, "with_body": with_body,
                "augment": bool(aug), "aug_strength": args.aug_strength,
                "aug_stream": args.aug_stream,
                "legacy_aug_rng": bool(args.legacy_aug_rng),
                "workers": n_workers, "torch_threads": args.torch_threads,
                "per_frame": per_frame,
            },
            "split": {
                "fit_rows": int(fit_mask.sum()), "dev_rows": int(dev_mask.sum()),
                "val_rows": int(val_mask.sum()), "dev_signers": dev_people,
            },
            "classes": classes,
            "support": [int(s) for s in support],
            "seeds": [{
                "seed": r["seed"], "fit_acc": r["fit_acc"],
                "dev_best": r["dev_best"],
                "val_at_best_dev": r["val_at_best_dev"],
                "val_at_best_val": r["val_at_best_val"],
                "epochs": r["epochs"],
            } for r in results],
            "summary": {
                "val_at_best_dev_mean": honest_m, "val_at_best_dev_std": honest_s,
                "val_at_best_val_mean": biased_m, "val_at_best_val_std": biased_s,
                "n_val": n_val,
            },
            # Per-seed per-class arrays, so a driver can check whether a
            # per-class result holds across streams - the check that section 4
            # of the ablation doc now requires and that printing alone can't do.
            "per_class": {"recall": rec.tolist(), "f1": f1.tolist()},
            "wall_seconds": time.time() - t_run_start,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nJSON results written to: {args.json_out}")

    if args.save_probs:
        # Row indices are into load_raw_clips() order (sorted .npz files,
        # train+val only). ensemble_v3.py rebuilds that order and refuses to
        # blend unless indices, labels and signers all match - blending two
        # probability matrices whose rows are different clips would produce
        # plausible-looking nonsense.
        probs_config = {
            "model": args.model, "timesteps": args.timesteps,
            "hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
            "batch": args.batch, "epochs": args.epochs,
            "patience": args.patience, "seed": args.seed, "seeds": args.seeds,
            "dev_frac": args.dev_frac, "holdout_seed": args.holdout_seed,
            "with_hand_local": with_hand_local, "with_body": with_body,
            "augment": bool(aug), "aug_strength": args.aug_strength,
            "aug_stream": args.aug_stream,
            "legacy_aug_rng": args.legacy_aug_rng,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_probs)) or ".",
                    exist_ok=True)
        np.savez_compressed(
            args.save_probs,
            classes=np.array(classes),
            seeds=np.array([r["seed"] for r in results]),
            dev_rows=np.flatnonzero(dev_mask),
            val_rows=np.flatnonzero(val_mask),
            y_dev=y[dev_mask], y_val=y[val_mask],
            dev_participants=participants[dev_mask].astype(str),
            val_participants=participants[val_mask].astype(str),
            dev_prob=np.stack([r["dev_prob"] for r in results]),
            val_prob=np.stack([r["val_prob"] for r in results]),
            dev_best=np.array([r["dev_best"] for r in results]),
            val_at_best_dev=np.array([r["val_at_best_dev"] for r in results]),
            fit_acc=np.array([r["fit_acc"] for r in results]),
            epochs=np.array([r["epochs"] for r in results]),
            config=np.array(json.dumps(probs_config)),
        )
        print(f"\nper-seed dev/val probabilities written to: {args.save_probs}")

    if args.save_seeds:
        seeds_config = {
            "model": args.model, "timesteps": args.timesteps,
            "hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
            "batch": args.batch, "epochs": args.epochs,
            "patience": args.patience, "seed": args.seed, "seeds": args.seeds,
            "dev_frac": args.dev_frac, "holdout_seed": args.holdout_seed,
            "with_hand_local": with_hand_local, "with_body": with_body,
            "augment": bool(aug), "aug_strength": args.aug_strength,
            "aug_stream": args.aug_stream,
            "legacy_aug_rng": args.legacy_aug_rng, "per_frame": per_frame,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_seeds)) or ".",
                    exist_ok=True)
        torch.save({
            "format": "signbridge-gru-seeds-v1",
            "config": seeds_config,
            "classes": list(classes),
            "seeds": [int(r["seed"]) for r in results],
            "states": [{k: v.detach().cpu() for k, v in r["state"].items()}
                       for r in results],
            "dev_best": [float(r["dev_best"]) for r in results],
            "val_at_best_dev": [float(r["val_at_best_dev"]) for r in results],
            "epochs": [int(r["epochs"]) for r in results],
            "dev_rows": np.flatnonzero(dev_mask).tolist(),
            "val_rows": np.flatnonzero(val_mask).tolist(),
            "y_dev": y[dev_mask].tolist(), "y_val": y[val_mask].tolist(),
        }, args.save_seeds)
        print(f"\nall {len(results)} seed checkpoints written to: {args.save_seeds}")

    if args.no_save:
        print("\n--no-save: checkpoint not written "
              "(the existing one is left untouched)")
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        out = os.path.join(MODEL_DIR, f"pytorch_v3{variant}_{args.model}.pt")
        torch.save({"state_dict": best["state"], "classes": classes,
                    "timesteps": args.timesteps, "model": args.model,
                    "hidden": args.hidden, "per_frame": per_frame,
                    "with_hand_local": with_hand_local, "with_body": with_body,
                    "seed": best["seed"], "dev_acc": best["dev_best"],
                    "val_acc": best["val_at_best_dev"],
                    "augment": bool(aug), "aug_strength": args.aug_strength,
                    "aug_stream": args.aug_stream,
                    "dev_signers": dev_people, "holdout_seed": args.holdout_seed}, out)
        print(f"Model saved to: {out}  (selected by dev, seed {best['seed']})")
    print("\nNOTE: the test split was never loaded by this script. Keep it for one final number.")


if __name__ == "__main__":
    mp.freeze_support()     # no-op when run as a script; required if ever frozen
    main()
