"""
SignBridge - feature engineering.

Reads every per-clip landmark file in data/landmarks/ (produced by
extract_clip_landmarks.py - variable-length normalized landmark
sequences, one per hand) and compresses each clip down into ONE
fixed-length row of numbers: this is what actually gets fed into the
RandomForest classifier next.

Per hand, per clip, we compute:
    - start position   (21 landmarks x [x,y,z] = 63 numbers) - the hand
      shape/position in the first frame it was seen
    - end position      (63 numbers) - the hand shape/position in the
      last frame it was seen
    - displacement       (63 numbers) - end minus start, i.e. net travel
      of each landmark over the whole clip. Explicit, because a decision
      tree can't easily reconstruct "difference of two features" from
      start/end alone in one split.
    - mean velocity      (63 numbers) - average frame-to-frame change,
      directional (captures overall drift direction/speed)
    - max speed          (21 numbers) - peak frame-to-frame movement
      magnitude per landmark, direction-agnostic (catches a sharp flick
      even if it doesn't show up much in the average)

  => 63*4 + 21 = 273 numbers per hand, plus 1 "hand present" flag
  => 274 * 2 hands = 548 features per clip total.

If a hand wasn't detected at all in a clip (e.g. a one-handed sign),
that hand's 273 numbers are all zero and its presence flag is 0 - the
classifier can learn "when left_present == 0, ignore left_* features."

Output: data/features.csv - one row per clip, columns = label, split,
participant, then the 548 feature columns.

Run:
    python build_feature_vectors.py
"""

import glob
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")

N_LANDMARKS = 21
N_FEATURES_PER_HAND = N_LANDMARKS * 3 * 4 + N_LANDMARKS  # start,end,disp,velmean (x,y,z) + maxspeed


def make_feature_names():
    names = []
    for side in ("left", "right"):
        for stat in ("start", "end", "disp", "velmean"):
            for lm in range(N_LANDMARKS):
                for axis in ("x", "y", "z"):
                    names.append(f"{side}_{stat}_lm{lm}_{axis}")
        for lm in range(N_LANDMARKS):
            names.append(f"{side}_maxspeed_lm{lm}")
        names.append(f"{side}_present")
    return names


def summarize_hand_sequence(seq):
    """
    seq: (n_frames, 21, 3) normalized landmark sequence for one hand,
         or None if that hand wasn't detected anywhere in this clip.

    Returns (feature_vector of length N_FEATURES_PER_HAND, presence_flag).
    """
    if seq is None or seq.shape[0] == 0:
        return np.zeros(N_FEATURES_PER_HAND, dtype=np.float32), 0

    start = seq[0].flatten()             # 63
    end = seq[-1].flatten()              # 63
    disp = (seq[-1] - seq[0]).flatten()  # 63

    if seq.shape[0] > 1:
        velocity = np.diff(seq, axis=0)            # (n_frames-1, 21, 3)
        mean_vel = velocity.mean(axis=0).flatten()  # 63
        speed = np.linalg.norm(velocity, axis=2)    # (n_frames-1, 21)
        max_speed = speed.max(axis=0)               # 21
    else:
        # only one frame ever detected - no velocity info available
        mean_vel = np.zeros(63, dtype=np.float32)
        max_speed = np.zeros(21, dtype=np.float32)

    vec = np.concatenate([start, end, disp, mean_vel, max_speed]).astype(np.float32)
    return vec, 1


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))
    print(f"Found {len(files)} landmark files")
    if not files:
        print(f"ERROR: no .npz files in {LANDMARKS_DIR}. Run extract_clip_landmarks.py first.")
        return

    feature_names = make_feature_names()
    rows = []

    for i, path in enumerate(files, 1):
        d = np.load(path, allow_pickle=True)

        left = d["left_hand"]
        right = d["right_hand"]
        left = left if left.shape[0] > 0 else None
        right = right if right.shape[0] > 0 else None

        left_vec, left_present = summarize_hand_sequence(left)
        right_vec, right_present = summarize_hand_sequence(right)

        feat = np.concatenate([left_vec, [left_present], right_vec, [right_present]])

        row = {
            "label": str(d["label"]),
            "split": str(d["split"]),
            "participant": str(d["participant"]),
        }
        row.update(zip(feature_names, feat))
        rows.append(row)

        if i % 200 == 0:
            print(f"  {i}/{len(files)} processed...")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)

    print(f"\nWrote {len(df)} rows x {len(feature_names)} feature columns to:\n  {FEATURES_PATH}")
    print("\nClips per label:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nClips per split:")
    print(df["split"].value_counts().to_string())
    print("\nHow often each hand was detected at all:")
    print(f"  left present:  {df['left_present'].sum()} / {len(df)}")
    print(f"  right present: {df['right_present'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
