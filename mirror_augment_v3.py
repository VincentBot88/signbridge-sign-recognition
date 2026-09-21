"""
SignBridge - mirror augmentation for the v3 (raw, frame-aligned) landmarks.

Same rationale as mirror_augment.py: in ASL, handedness belongs to the
signer rather than to the sign, so a mirrored training clip is a valid
example of the same sign rather than synthetic noise - and ASL Citizen
skews right-dominant, which is an accessibility gap for a kiosk, not just
a scoring quirk.

WHAT IS DIFFERENT HERE
----------------------
The v2 version mirrored ALREADY-NORMALIZED hand coordinates, where
mirroring reduces to negating x (the derivation in that file's docstring:
with the origin subtracted, (1 - raw_x) - (1 - origin_x) = -(raw_x - origin_x)).

v3 stores raw MediaPipe coordinates, so the mirror is the literal image
flip it is named after:

    x -> 1 - x        (MediaPipe x is normalized to image width)
    y, z unchanged

and three things get swapped rather than two:

  - the left and right HAND sequences (a mirrored right hand is a left hand)
  - the left and right POSE landmark pairs - shoulders, elbows, ears,
    mouth corners, pose wrists. Missing this would leave the body
    reference un-mirrored while the hands were mirrored, producing clips
    where the right hand signs from the left shoulder: worse than no
    augmentation, because it teaches the exact relationship the body block
    exists to learn.

Applied to TRAIN clips only - never val/test, since evaluating on
mirrored copies of training data would inflate the number.

Run:
    python mirror_augment_v3.py
    python build_feature_vectors_v3.py
"""

import glob
import os

import numpy as np

from mp_body_detector import POSE_MIRROR_PAIRS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks_v3")


def flip_x_raw(seq):
    """Mirror raw MediaPipe-normalized coordinates about the image centre."""
    if seq is None or seq.shape[0] == 0:
        return seq
    out = np.array(seq, dtype=np.float32, copy=True)
    out[:, :, 0] = 1.0 - out[:, :, 0]
    return out


def mirror_pose(pose):
    """Flip x, then swap every left/right landmark pair."""
    if pose is None or pose.shape[0] == 0:
        return pose
    out = flip_x_raw(pose)
    for a, b in POSE_MIRROR_PAIRS:
        out[:, [a, b], :] = out[:, [b, a], :]
    return out


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))

    train_files = []
    for f in files:
        if "_mirror" in os.path.basename(f) or "_aug" in os.path.basename(f):
            continue
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) == "train":
            train_files.append(f)

    print(f"Found {len(train_files)} real training clips to mirror")

    created = 0
    for i, path in enumerate(train_files, 1):
        d = np.load(path, allow_pickle=True)

        # A mirrored right hand IS the left hand, so the arrays swap places
        # as well as being flipped.
        m_left = flip_x_raw(d["right_hand"])
        m_right = flip_x_raw(d["left_hand"])
        m_pose = mirror_pose(d["pose"]) if "pose" in d else None

        out_path = os.path.join(LANDMARKS_DIR, os.path.basename(path)[:-4] + "_mirror.npz")
        payload = dict(
            left_hand=m_left,
            right_hand=m_right,
            frame_w=int(d["frame_w"]),
            frame_h=int(d["frame_h"]),
            n_frames=int(d["n_frames"]),
            label=str(d["label"]),
            split=str(d["split"]),                     # stays "train"
            participant=str(d["participant"]) + "_mirror",
        )
        if m_pose is not None:
            payload["pose"] = m_pose
        np.savez_compressed(out_path, **payload)

        created += 1
        if i % 200 == 0:
            print(f"  {i}/{len(train_files)} mirrored...")

    print(f"\nDone. Created {created} mirrored training clips in {LANDMARKS_DIR}")
    print("Next:  python build_feature_vectors_v3.py")


if __name__ == "__main__":
    main()
