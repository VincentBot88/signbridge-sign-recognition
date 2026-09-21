"""
SignBridge - feature engineering, v3 (body-relative location added).

WHAT v3 ADDS
------------
v2 fixed the representation's blindness to MOTION (summary statistics
cancelled out oscillating signs; keyframes + path length fixed it,
0.403 -> 0.726). v3 fixes its blindness to LOCATION.

Every v2 feature is measured in a hand-local frame: the hand's own wrist
is the origin and the hand's own size is the unit. That is the right
frame for handshape, and it makes the features invariant to where the
signer is standing - but sign location is not a nuisance parameter in
ASL, it is part of the sign. HUNGRY and THANK-YOU both start near the
chin; they differ in where the hand travels relative to the body. With no
chin, no shoulders and no torso in the feature vector, that distinction
is not merely hard to learn, it is absent from the input. (Measured:
HUNGRY<->THANKYOU accounted for 29% of their combined val clips.)

v3 keeps the whole v2 block unchanged and adds a second block measured in
a BODY frame - origin at the shoulder midpoint, unit = shoulder width -
covering, per hand:

  - where three anchor points of the hand (wrist, index fingertip,
    middle-finger MCP) are over the course of the sign, in shoulder-widths
    from the base of the neck
  - how far those anchors travel and their peak speed IN THAT FRAME
    (v2's path length is hand-local, so a hand sweeping across the torso
    with a static handshape registers almost no motion)
  - the signed offset of the wrist from the nose and from the mouth at
    each keyframe - the direct encoding of "at the chin", "above the
    head", "at the chest"
  - distance from the neck and from the same-side shoulder

plus a shared block for the two elbows, which separates signs made with
the arm raised from ones made with it tucked even when the hand path is
similar.

THE TWO BLOCKS ARE COMPLEMENTARY, NOT REDUNDANT
-----------------------------------------------
Hand-local answers "what is the hand doing"; body-relative answers "where
is it doing it". Keeping both means this change cannot lose anything v2
already had - the worst case is that the RandomForest ignores the new
columns. Because the v2 block is produced by importing v2's own
build_clip_features() rather than a copy of it, `--no-body` reproduces the
v2 feature set exactly, which is the control for measuring whether the
body block helped.

    python build_feature_vectors_v3.py --no-body       # control (= v2)
    python build_feature_vectors_v3.py                 # both blocks
    python build_feature_vectors_v3.py --no-hand-local # body block alone

The third one is the interesting diagnostic: if body-only lands anywhere
near the v2 number using ~1/4 the features, that is strong evidence the
location signal is real rather than noise the trees happened to fit.

WHY THIS NEEDS v3 LANDMARKS
---------------------------
data/landmarks/ stores hands already normalized and with each hand's
frames independently indexed, so there is no shared clock on which to ask
"where was the hand relative to the shoulders at this moment", and no
shoulders either. extract_landmarks_v3.py re-extracts raw, frame-aligned
hands + pose into data/landmarks_v3/. See that script's docstring.

Run:
    python extract_landmarks_v3.py
    python mirror_augment_v3.py
    python build_feature_vectors_v3.py
    python train_classifier.py data\\features_v3.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# The v2 block is IMPORTED, not reimplemented, for the same reason
# build_clip_features() is shared with live_recognize.py: two copies of a
# feature definition drift apart silently and invalidate every comparison
# made across them.
from build_feature_vectors_v2 import (
    LABEL_MERGES,
    N_KEYFRAMES,
    N_LANDMARKS,
    build_clip_features as build_hand_local_features,
    make_feature_names as make_hand_local_names,
    resample_to_keyframes,
)
from mp_hand_detector import normalize_landmark_sequence
from mp_body_detector import (
    aspect_correct, body_reference, to_body_frame,
    B_NOSE, B_MOUTH_L, B_MOUTH_R, B_SHOULDER_L, B_SHOULDER_R, B_ELBOW_L, B_ELBOW_R,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks_v3")

# Three points are enough to fix the hand's placement and orientation in
# body space; all 21 would add ~400 columns of information the hand-local
# block already carries in a better frame.
BODY_ANCHORS = (0, 8, 9)
ANCHOR_NAMES = ("wrist", "idxtip", "midmcp")

# A clip whose pose was detected on fewer than this fraction of frames is
# treated as having no usable body reference: interpolating an origin
# across a near-total dropout invents a torso that was never seen.
MIN_POSE_COVERAGE = 0.2

AXES = ("x", "y")


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

def make_body_names():
    names = []
    for side in ("left", "right"):
        for k in range(N_KEYFRAMES):
            for anchor in ANCHOR_NAMES:
                for axis in AXES:
                    names.append(f"{side}_body_kf{k}_{anchor}_{axis}")
        for anchor in ANCHOR_NAMES:
            names.append(f"{side}_body_pathlen_{anchor}")
        for anchor in ANCHOR_NAMES:
            names.append(f"{side}_body_peakspeed_{anchor}")
        for ref in ("nose", "mouth"):
            for k in range(N_KEYFRAMES):
                for axis in AXES:
                    names.append(f"{side}_body_kf{k}_d{ref}_{axis}")
        for k in range(N_KEYFRAMES):
            names.append(f"{side}_body_kf{k}_dist_shoulder")
        for k in range(N_KEYFRAMES):
            names.append(f"{side}_body_kf{k}_dist_neck")
        names.append(f"{side}_body_present")

    for k in range(N_KEYFRAMES):
        for elbow in ("elbow_l", "elbow_r"):
            for axis in AXES:
                names.append(f"body_kf{k}_{elbow}_{axis}")
    names.append("body_present")
    return names


N_BODY_PER_HAND = (
    N_KEYFRAMES * len(BODY_ANCHORS) * len(AXES)   # anchor keyframes
    + len(BODY_ANCHORS)                            # path length
    + len(BODY_ANCHORS)                            # peak speed
    + 2 * N_KEYFRAMES * len(AXES)                  # signed offsets to nose, mouth
    + N_KEYFRAMES                                  # distance to same-side shoulder
    + N_KEYFRAMES                                  # distance to neck
    + 1                                            # presence
)
N_BODY_SHARED = N_KEYFRAMES * 2 * len(AXES) + 1    # two elbows + presence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compact(seq):
    """
    (T, L, 3) frame-aligned with NaN gaps -> (n_detected, L, 3), or None.

    This reproduces v2's storage convention exactly: "the frames this hand
    was visible in", in order, with no gaps.
    """
    if seq is None or seq.shape[0] == 0:
        return None
    valid = np.isfinite(seq[:, 0, 0])
    if not valid.any():
        return None
    return seq[valid]


def interp_landmarks(seq_xy, valid):
    """Fill NaN frames of a (T, L, 2) array by linear interpolation."""
    out = np.array(seq_xy, dtype=np.float32, copy=True)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    t = np.arange(out.shape[0])
    for lm in range(out.shape[1]):
        for ax in range(out.shape[2]):
            out[:, lm, ax] = np.interp(t, idx, out[idx, lm, ax])
    return out


def masked_path_stats(seq, valid):
    """
    Path length and peak speed over a (T, L, 2) sequence, counting only
    ADJACENT pairs of frames that were both actually detected.

    A hand that vanishes mid-clip and reappears elsewhere would otherwise
    contribute one enormous fake step to its own path length - the frames
    either side are adjacent in the compacted array but seconds apart in
    the video.
    """
    both = valid[:-1] & valid[1:]
    if not both.any():
        return (np.zeros(seq.shape[1], dtype=np.float32),
                np.zeros(seq.shape[1], dtype=np.float32))
    steps = np.linalg.norm(np.diff(seq, axis=0), axis=2)   # (T-1, L)
    steps = steps[both]
    return steps.sum(axis=0).astype(np.float32), steps.max(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Body block
# ---------------------------------------------------------------------------

def body_hand_block(hand_body, pose_body, origin_ok, side):
    """
    Body-frame features for one hand.

    hand_body: (T, 21, 2) hand landmarks ALREADY in the body frame (same
               origin and scale as pose_body, so the two are directly
               comparable), NaN on frames the hand was not detected
    pose_body: (T, 11, 2) pose landmarks in the same body frame
    origin_ok: bool - whether a usable body reference exists for this clip
    side:      "left" or "right", picks the same-side shoulder

    Returns (vector of length N_BODY_PER_HAND - 1, presence flag); the
    caller appends the flag.
    """
    zeros = np.zeros(N_BODY_PER_HAND - 1, dtype=np.float32)
    if hand_body is None or not origin_ok:
        return zeros, 0

    hand_valid = np.isfinite(hand_body[:, 0, 0])
    if hand_valid.sum() < 2:
        return zeros, 0

    anchors = hand_body[:, BODY_ANCHORS, :]             # (T, 3, 2)

    path_len, peak_speed = masked_path_stats(anchors, hand_valid)

    # Everything below is measured only on frames the hand was seen in,
    # then resampled to a fixed number of keyframes - same convention as
    # v2, so keyframe k means the same point of the sign in both blocks.
    a_seen = anchors[hand_valid]                        # (n, 3, 2)
    kf_anchors = resample_to_keyframes(a_seen, N_KEYFRAMES)   # (K, 3, 2)

    wrist_seen = anchors[hand_valid][:, 0, :]           # (n, 2)
    nose_seen = pose_body[hand_valid][:, B_NOSE, :]
    mouth_seen = (pose_body[hand_valid][:, B_MOUTH_L, :]
                  + pose_body[hand_valid][:, B_MOUTH_R, :]) / 2.0
    shoulder_idx = B_SHOULDER_L if side == "left" else B_SHOULDER_R
    shoulder_seen = pose_body[hand_valid][:, shoulder_idx, :]

    d_nose = resample_to_keyframes((wrist_seen - nose_seen)[:, None, :], N_KEYFRAMES)[:, 0, :]
    d_mouth = resample_to_keyframes((wrist_seen - mouth_seen)[:, None, :], N_KEYFRAMES)[:, 0, :]

    dist_shoulder = np.linalg.norm(wrist_seen - shoulder_seen, axis=1)
    dist_neck = np.linalg.norm(wrist_seen, axis=1)      # neck == origin == (0,0)
    kf_dist_shoulder = resample_to_keyframes(dist_shoulder[:, None, None], N_KEYFRAMES)[:, 0, 0]
    kf_dist_neck = resample_to_keyframes(dist_neck[:, None, None], N_KEYFRAMES)[:, 0, 0]

    vec = np.concatenate([
        kf_anchors.reshape(-1),        # K * 3 anchors * 2 axes
        path_len,
        peak_speed,
        d_nose.reshape(-1),
        d_mouth.reshape(-1),
        kf_dist_shoulder,
        kf_dist_neck,
    ]).astype(np.float32)

    assert vec.size == N_BODY_PER_HAND - 1, (vec.size, N_BODY_PER_HAND)
    return vec, 1


def body_shared_block(pose_body, pose_valid, origin_ok):
    """Elbow trajectories in the body frame, shared across both hands."""
    zeros = np.zeros(N_BODY_SHARED - 1, dtype=np.float32)
    if not origin_ok or pose_valid.sum() < 2:
        return zeros, 0
    elbows = pose_body[pose_valid][:, [B_ELBOW_L, B_ELBOW_R], :]   # (n, 2, 2)
    kf = resample_to_keyframes(elbows, N_KEYFRAMES)
    return kf.reshape(-1).astype(np.float32), 1


def build_body_features(left_raw, right_raw, pose_raw, frame_w, frame_h):
    """
    Full body block for one clip, in make_body_names() order.

    left_raw / right_raw: (T, 21, 3) raw, NaN-gapped
    pose_raw:             (T, 11, 3) raw, NaN-gapped
    """
    n_body = 2 * N_BODY_PER_HAND + N_BODY_SHARED
    if pose_raw is None or pose_raw.shape[0] == 0:
        return np.zeros(n_body, dtype=np.float32)

    pose_xy = aspect_correct(pose_raw[:, :, :2], frame_w, frame_h)
    origin, scale, pose_valid = body_reference(pose_xy)

    origin_ok = (pose_valid.sum() >= 2
                 and pose_valid.mean() >= MIN_POSE_COVERAGE
                 and scale > 0)

    if not origin_ok:
        return np.zeros(n_body, dtype=np.float32)

    # Interpolate pose gaps so the reference points exist on every frame a
    # hand was seen in, then move everything into the body frame together.
    pose_xy = interp_landmarks(pose_xy, pose_valid)
    pose_body = to_body_frame(pose_xy, origin, scale)

    blocks = []
    for side, raw in (("left", left_raw), ("right", right_raw)):
        if raw is None or raw.shape[0] == 0:
            hand_body = None
        else:
            hand_xy = aspect_correct(raw[:, :, :2], frame_w, frame_h)
            hand_body = to_body_frame(hand_xy, origin, scale)
            # to_body_frame propagates NaN, so missing frames stay missing.
        vec, present = body_hand_block(hand_body, pose_body, origin_ok, side)
        blocks.append(np.concatenate([vec, [present]]))

    shared, shared_present = body_shared_block(pose_body, pose_valid, origin_ok)
    blocks.append(np.concatenate([shared, [shared_present]]))

    return np.concatenate(blocks).astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_clip_features_v3(left_raw, right_raw, pose_raw, frame_w, frame_h,
                           with_hand_local=True, with_body=True):
    """
    SINGLE SOURCE OF TRUTH for v3 feature construction - the batch builder
    below and (eventually) live_recognize.py must both call this, so the
    features the model trains on and the features it is served at runtime
    cannot drift apart.
    """
    parts = []
    if with_hand_local:
        left_local = compact(left_raw)
        right_local = compact(right_raw)
        left_local = normalize_landmark_sequence(left_local) if left_local is not None else None
        right_local = normalize_landmark_sequence(right_local) if right_local is not None else None
        parts.append(build_hand_local_features(left_local, right_local))
    if with_body:
        parts.append(build_body_features(left_raw, right_raw, pose_raw, frame_w, frame_h))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-body", action="store_true",
                    help="v2 feature set only - the control for measuring the body block")
    ap.add_argument("--no-hand-local", action="store_true",
                    help="body block only - diagnostic for whether location alone carries signal")
    ap.add_argument("--out", default=None, help="output csv path")
    args = ap.parse_args()

    with_hand_local = not args.no_hand_local
    with_body = not args.no_body
    if not (with_hand_local or with_body):
        print("ERROR: --no-body and --no-hand-local together leave no features.")
        return

    suffix = "" if (with_hand_local and with_body) else \
             ("_handlocal" if with_hand_local else "_bodyonly")
    out_path = args.out or os.path.join(BASE_DIR, "data", f"features_v3{suffix}.csv")
    if not os.path.isabs(out_path):
        out_path = os.path.join(BASE_DIR, out_path)

    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))
    print(f"Found {len(files)} v3 landmark files in {LANDMARKS_DIR}")
    if not files:
        print("ERROR: run extract_landmarks_v3.py first.")
        return

    names = []
    if with_hand_local:
        names += make_hand_local_names()
    if with_body:
        names += make_body_names()
    print(f"Building {len(names)} features per clip "
          f"(hand-local: {with_hand_local}, body: {with_body})")

    rows = []
    n_body_present = 0
    for i, path in enumerate(files, 1):
        d = np.load(path, allow_pickle=True)

        left = d["left_hand"]
        right = d["right_hand"]
        pose = d["pose"] if "pose" in d else None
        frame_w = int(d["frame_w"]) if "frame_w" in d else 0
        frame_h = int(d["frame_h"]) if "frame_h" in d else 0

        feat = build_clip_features_v3(left, right, pose, frame_w, frame_h,
                                      with_hand_local=with_hand_local,
                                      with_body=with_body)

        raw_label = str(d["label"])
        row = {
            "label": LABEL_MERGES.get(raw_label, raw_label),
            "split": str(d["split"]),
            "participant": str(d["participant"]),
        }
        row.update(zip(names, feat))
        rows.append(row)

        if with_body and row.get("body_present", 0) == 1:
            n_body_present += 1

        if i % 400 == 0:
            print(f"  {i}/{len(files)} processed...")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows x {len(names)} feature columns to:\n  {out_path}")
    if with_body:
        print(f"Body reference usable on {n_body_present}/{len(df)} clips "
              f"({n_body_present/max(len(df),1):.1%}) - the rest fall back to "
              f"hand-local features with the body block zeroed.")
    print(f"Classes: {df['label'].nunique()}")
    print("\nClips per split:")
    print(df["split"].value_counts().to_string())
    print(f"\nNext:  python train_classifier.py {os.path.relpath(out_path, BASE_DIR)}")


if __name__ == "__main__":
    main()
