"""
SignBridge - mirror augmentation (handedness coverage).

WHY
---
Measured on the live webcam: right-handed signing is recognized
immediately and confidently; the same signs performed left-handed are
noticeably weaker. That's an accessibility gap, not a scoring quirk - a
left-dominant Deaf user would get materially worse service from the
kiosk than a right-dominant one.

The cause is structural. The feature vector is
[left block, left_present, right block, right_present]. A sign done
right-handed fills the right block and zeros the left; done left-handed
it fills the left block and zeros the right. Those two vectors share
almost no values, so the classifier only handles the handedness it saw
in training - and ASL Citizen skews right-dominant.

In ASL, handedness belongs to the signer, not the sign: a left-handed
signer mirroring a sign is producing correct ASL, not a variant. So
mirroring training clips generates genuinely valid new examples rather
than synthetic noise - which is why this should help where the earlier
jitter/time-stretch augmentation did nothing (that added no information
the model didn't already have, and under v2's keyframe resampling the
time-stretch is a literal no-op).

THE TRANSFORM
-------------
The stored sequences are already normalized (wrist of frame 0 at the
origin, scaled by hand size). Mirroring the source image is exactly
equivalent to negating the x column of those normalized coordinates:

    mirroring the image sends raw_x -> (1 - raw_x), so
    translated_x = (1 - raw_x) - (1 - origin_x) = -(raw_x - origin_x)

and the scale reference is a vector norm, which is unchanged by the sign
flip. So: negate x, then swap the left and right hand sequences (a
mirrored right hand is a left hand, and MediaPipe would label it so).

Applied to TRAIN clips only - never val/test, since evaluating on
mirrored copies of your own training data would inflate the number.
Skips already-augmented files.

Run:
    python mirror_augment.py
    python build_feature_vectors_v2.py
    python train_classifier.py data/features_v2.csv
"""

import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")


def flip_x(seq):
    """Mirror a normalized (n_frames, 21, 3) sequence about the vertical axis."""
    if seq is None or seq.shape[0] == 0:
        return seq
    out = seq.copy()
    out[:, :, 0] *= -1.0
    return out


def mirror_hands(left, right):
    """
    Mirror a clip: a mirrored RIGHT hand becomes the LEFT hand and vice
    versa, with x negated on both.
    """
    return flip_x(right), flip_x(left)


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))

    train_files = []
    for f in files:
        base = os.path.basename(f)
        if "_aug" in base or "_mirror" in base:
            continue  # don't mirror synthetic copies
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) == "train":
            train_files.append(f)

    print(f"Found {len(train_files)} real training clips to mirror")

    created = 0
    for i, path in enumerate(train_files, 1):
        d = np.load(path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]

        m_left, m_right = mirror_hands(left, right)

        out_path = os.path.join(LANDMARKS_DIR, os.path.basename(path)[:-4] + "_mirror.npz")
        np.savez_compressed(
            out_path,
            left_hand=m_left,
            right_hand=m_right,
            label=str(d["label"]),
            split=str(d["split"]),          # stays "train"
            participant=str(d["participant"]) + "_mirror",
        )
        created += 1
        if i % 100 == 0:
            print(f"  {i}/{len(train_files)} mirrored...")

    print(f"\nDone. Created {created} mirrored training clips.")
    print("Next:")
    print("  python build_feature_vectors_v2.py")
    print("  python train_classifier.py data/features_v2.csv")
    print("Then re-test left-handed signing in live_recognize.py.")


if __name__ == "__main__":
    main()
