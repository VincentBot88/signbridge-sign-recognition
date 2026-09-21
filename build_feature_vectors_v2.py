"""
SignBridge - feature engineering, v2 (keyframe trajectory representation).

WHY THIS EXISTS
---------------
v1 (build_feature_vectors.py) compressed each clip into 5 summary
statistics: start pose, end pose, displacement, mean velocity, peak
speed. That representation turned out to be the accuracy ceiling - every
model/data experiment stalled at ~0.40 val because the features
themselves had already destroyed the information.

Measured on real val clips, comparing total distance the hand actually
travels vs. its net start-to-end displacement:

    STOP          33.5x   (travels 7.87 units, nets 0.24)
    SORRY         21.4x
    INTERPRETER   19.2x
    HURT           7.2x
    WHERE          5.6x

For an oscillating sign (WHERE = index finger wiggling side to side,
STOP, SORRY = circular motion), start ~ end, displacement ~ 0, and mean
velocity ~ 0 because the back-and-forth cancels out in an average. Four
of v1's five feature families are blind to those signs by construction.

WHAT V2 DOES INSTEAD
--------------------
Per hand:
  - KEYFRAMES: resample the landmark sequence to exactly N_KEYFRAMES
    evenly-spaced frames (linear interpolation) and use those landmark
    positions directly. This preserves the SHAPE of the trajectory over
    time - including oscillation - instead of averaging it away.
    Clips of different lengths all land on the same number of keyframes,
    which is what made fixed-length summarization necessary in the first
    place.
  - PATH LENGTH per landmark: total distance actually travelled,
    summed frame to frame. Direction-agnostic, so a wiggle registers as
    real movement instead of cancelling to zero. This is the single
    feature v1 was most obviously missing.
  - PEAK SPEED per landmark: largest single-frame movement (kept from v1,
    it was the one motion feature that survived cancellation).
  - presence flag.

Keyframes use x,y only by default: MediaPipe's z is an unreliable
monocular depth estimate, and dropping z from v1 was measured to cost
nothing (0.403 -> 0.403). Path length and peak speed still use full 3D.

Writes to data/features_v2.csv, leaving v1's data/features.csv intact so
the two can be compared directly.

Run:
    python build_feature_vectors_v2.py
    python train_classifier.py data/features_v2.csv
"""

import glob
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features_v2.csv")

N_LANDMARKS = 21
N_KEYFRAMES = 10          # how many evenly-spaced frames to sample per clip
KEYFRAME_AXES = ("x", "y")  # drop z from keyframes (noisy monocular depth)

# Labels merged at feature-build time, so a vocabulary change doesn't require
# re-running the ~25-minute landmark extraction.
#
# HURT + PAIN: measured as genuinely inseparable, not merely hard. Trained as a
# standalone two-way problem they scored 0.125 val accuracy - below the 0.50
# coin-flip baseline, i.e. no usable signal at all. That matches ASL: both are
# commonly signed with index fingers jabbing toward each other, with location
# indicating where it hurts. Keeping them apart asks the classifier to make a
# distinction the video doesn't contain. Merging measured +3.2 points val
# accuracy and leaves 27 words, still above the spec's 20-sign minimum.
LABEL_MERGES = {
    "HURT": "HURT_PAIN",
    "PAIN": "HURT_PAIN",
}

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
N_FEATURES_PER_HAND = (
    N_KEYFRAMES * N_LANDMARKS * len(KEYFRAME_AXES)  # keyframe positions
    + N_LANDMARKS                                    # path length
    + N_LANDMARKS                                    # peak speed
)


def make_feature_names():
    names = []
    for side in ("left", "right"):
        for k in range(N_KEYFRAMES):
            for lm in range(N_LANDMARKS):
                for axis in KEYFRAME_AXES:
                    names.append(f"{side}_kf{k}_lm{lm}_{axis}")
        for lm in range(N_LANDMARKS):
            names.append(f"{side}_pathlen_lm{lm}")
        for lm in range(N_LANDMARKS):
            names.append(f"{side}_peakspeed_lm{lm}")
        names.append(f"{side}_present")
    return names


def resample_to_keyframes(seq, n_keyframes=N_KEYFRAMES):
    """
    seq: (n_frames, 21, 3) -> (n_keyframes, 21, 3)

    Linear interpolation along the frame axis. A 55-frame clip and a
    71-frame clip both come out as n_keyframes frames, so the trajectory
    shape is comparable across clips of different durations.
    """
    n_frames = seq.shape[0]
    if n_frames == 1:
        return np.repeat(seq, n_keyframes, axis=0)

    old_idx = np.linspace(0, n_frames - 1, num=n_frames)
    new_idx = np.linspace(0, n_frames - 1, num=n_keyframes)
    out = np.empty((n_keyframes, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for lm in range(seq.shape[1]):
        for axis in range(seq.shape[2]):
            out[:, lm, axis] = np.interp(new_idx, old_idx, seq[:, lm, axis])
    return out


def summarize_hand_sequence(seq):
    """Returns (feature_vector of length N_FEATURES_PER_HAND, presence_flag)."""
    if seq is None or seq.shape[0] == 0:
        return np.zeros(N_FEATURES_PER_HAND, dtype=np.float32), 0

    keyframes = resample_to_keyframes(seq)
    axis_cols = [_AXIS_IDX[a] for a in KEYFRAME_AXES]
    kf_flat = keyframes[:, :, axis_cols].flatten()   # N_KEYFRAMES * 21 * len(axes)

    if seq.shape[0] > 1:
        steps = np.diff(seq, axis=0)                      # (n_frames-1, 21, 3)
        step_mag = np.linalg.norm(steps, axis=2)          # (n_frames-1, 21)
        path_len = step_mag.sum(axis=0)                   # 21 - total distance travelled
        peak_speed = step_mag.max(axis=0)                 # 21 - largest single step
    else:
        path_len = np.zeros(N_LANDMARKS, dtype=np.float32)
        peak_speed = np.zeros(N_LANDMARKS, dtype=np.float32)

    vec = np.concatenate([kf_flat, path_len, peak_speed]).astype(np.float32)
    return vec, 1


def build_clip_features(left_seq, right_seq):
    """
    Build the full feature vector for one clip, in the exact column order
    make_feature_names() describes: [left block, left_present, right block,
    right_present].

    THIS IS THE SINGLE SOURCE OF TRUTH for feature construction. Both the
    batch builder below and live_recognize.py call it, so the features the
    model is trained on and the features it's served at runtime cannot
    silently drift apart.

    left_seq / right_seq: (n_frames, 21, 3) normalized landmark sequences,
                          or None / empty if that hand wasn't detected.
    """
    left_vec, left_present = summarize_hand_sequence(left_seq)
    right_vec, right_present = summarize_hand_sequence(right_seq)
    return np.concatenate([left_vec, [left_present], right_vec, [right_present]])


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))
    print(f"Found {len(files)} landmark files")
    if not files:
        print(f"ERROR: no .npz files in {LANDMARKS_DIR}. Run extract_clip_landmarks.py first.")
        return

    feature_names = make_feature_names()
    print(f"Building {len(feature_names)} features per clip "
          f"({N_KEYFRAMES} keyframes x {N_LANDMARKS} landmarks x {len(KEYFRAME_AXES)} axes, "
          f"plus path length + peak speed per landmark, per hand)")

    rows = []
    for i, path in enumerate(files, 1):
        d = np.load(path, allow_pickle=True)

        left = d["left_hand"]
        right = d["right_hand"]
        left = left if left.shape[0] > 0 else None
        right = right if right.shape[0] > 0 else None

        feat = build_clip_features(left, right)

        raw_label = str(d["label"])
        row = {
            "label": LABEL_MERGES.get(raw_label, raw_label),
            "split": str(d["split"]),
            "participant": str(d["participant"]),
        }
        row.update(zip(feature_names, feat))
        rows.append(row)

        if i % 400 == 0:
            print(f"  {i}/{len(files)} processed...")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)

    print(f"\nWrote {len(df)} rows x {len(feature_names)} feature columns to:\n  {FEATURES_PATH}")
    if LABEL_MERGES:
        print(f"Applied label merges: {LABEL_MERGES} -> {df['label'].nunique()} classes")
    print("\nClips per split:")
    print(df["split"].value_counts().to_string())
    print("\nNext:  python train_classifier.py data/features_v2.csv")
    print("Compare the val accuracy against v1's 0.403.")


if __name__ == "__main__":
    main()
