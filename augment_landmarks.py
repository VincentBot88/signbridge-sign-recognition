"""
SignBridge - data augmentation for the training set.

We only have ~17 real examples per word on average, which is thin for a
548-feature classifier (confirmed empirically: neither feature trimming
nor RandomForest regularization settings improved val accuracy - the
bottleneck is data quantity, not model configuration). This script
synthetically multiplies the TRAINING clips (never val/test - augmenting
your evaluation data would give you a fake, inflated accuracy number)
by generating slightly-varied copies of each real landmark sequence:

  - small random jitter added to every landmark coordinate, simulating
    natural MediaPipe detection noise / hand micro-tremor
  - small random time-stretching (resampling the frame sequence to be a
    bit longer or shorter), simulating natural variation in signing speed

Neither of these changes what sign is being performed - they're just
plausible variations of the same real recording.

Run:
    python augment_landmarks.py

This reads every train-split .npz in data/landmarks/ and writes
N_AUGMENTATIONS new .npz files per clip (suffixed _aug1, _aug2, ...)
into the same folder, still labeled split="train". After running this,
rerun build_feature_vectors.py - it will automatically pick up the new
files and your train set will be several times larger. val/test files
are left completely untouched.
"""

import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")

N_AUGMENTATIONS = 3          # how many synthetic copies to make per real training clip
JITTER_STD = 0.02            # std-dev of positional noise, in normalized hand-width units
TIME_STRETCH_RANGE = (0.85, 1.15)  # resample each sequence to 85%-115% of its original length

rng = np.random.default_rng(42)


def jitter(seq):
    if seq.shape[0] == 0:
        return seq
    noise = rng.normal(0, JITTER_STD, size=seq.shape).astype(np.float32)
    return seq + noise


def time_stretch(seq):
    n_frames = seq.shape[0]
    if n_frames < 2:
        return seq  # nothing to stretch with 0 or 1 frames
    factor = rng.uniform(*TIME_STRETCH_RANGE)
    new_n = max(2, int(round(n_frames * factor)))
    # linear interpolation along the frame axis for each (landmark, coord) independently
    old_idx = np.linspace(0, n_frames - 1, num=n_frames)
    new_idx = np.linspace(0, n_frames - 1, num=new_n)
    stretched = np.empty((new_n, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for lm in range(seq.shape[1]):
        for axis in range(seq.shape[2]):
            stretched[:, lm, axis] = np.interp(new_idx, old_idx, seq[:, lm, axis])
    return stretched


def augment_one(left, right):
    aug_left = jitter(time_stretch(left)) if left.shape[0] > 0 else left
    aug_right = jitter(time_stretch(right)) if right.shape[0] > 0 else right
    return aug_left, aug_right


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))
    train_files = []
    for f in files:
        if "_aug" in os.path.basename(f):
            continue  # don't augment already-augmented files
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) == "train":
            train_files.append(f)

    print(f"Found {len(train_files)} real training clips to augment "
          f"({N_AUGMENTATIONS} copies each -> {len(train_files) * N_AUGMENTATIONS} new files)")

    created = 0
    for i, path in enumerate(train_files, 1):
        d = np.load(path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]
        label, split, participant = str(d["label"]), str(d["split"]), str(d["participant"])

        base_name = os.path.basename(path)[:-4]  # strip .npz
        for aug_i in range(1, N_AUGMENTATIONS + 1):
            aug_left, aug_right = augment_one(left, right)
            out_path = os.path.join(LANDMARKS_DIR, f"{base_name}_aug{aug_i}.npz")
            np.savez_compressed(
                out_path,
                left_hand=aug_left,
                right_hand=aug_right,
                label=label,
                split=split,  # stays "train" - augmented clips are still training data
                participant=participant,
            )
            created += 1

        if i % 100 == 0:
            print(f"  {i}/{len(train_files)} clips augmented...")

    print(f"\nDone. Created {created} new augmented training files in {LANDMARKS_DIR}")
    print("Next: rerun build_feature_vectors.py, then train_classifier.py, and compare")
    print("val accuracy to the 0.403 baseline.")


if __name__ == "__main__":
    main()
