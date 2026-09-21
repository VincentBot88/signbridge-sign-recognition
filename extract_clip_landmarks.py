"""
SignBridge - batch landmark extraction.

Reads every clip listed in data/clips_manifest.csv (produced by
select_vocabulary_clips.py), runs the shared MediaPipe hand detector on
every frame, normalizes the landmarks, and saves one .npz file per clip
containing the per-frame normalized landmark sequence for up to 2 hands.

This produces RAW per-frame landmark sequences, not yet the fixed-length
feature vectors the RandomForest classifier will train on - that's the
next step (feature engineering: start/end position, velocity, etc.),
built on top of this output. Keeping these as separate steps means we can
re-derive different feature schemes later without re-running MediaPipe
over every clip again.

Run:
    python extract_clip_landmarks.py

This will take a while the first time - expect roughly 1-3 frames/sec per
clip on a laptop CPU, so ~1000 clips at ~2.5s/~70 frames each could take
somewhere in the ballpark of 20-40 minutes. Progress is printed as it goes.
"""

import csv
import os
import time

import cv2
import numpy as np
import mediapipe as mp

from mp_hand_detector import create_detector, landmarks_to_array, normalize_landmark_sequence, to_mp_image, RunningMode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "clips_manifest.csv")
CLIPS_DIR = os.path.join(BASE_DIR, "data", "clips")
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")


def extract_video_landmarks(detector, video_path):
    """
    Run the hand detector over every frame of one clip (IMAGE mode - each
    frame is detected independently, which is correct for a batch of many
    short, unrelated clips).

    Returns a dict:
        {
          "Left":  (n_frames_with_left_hand, 21, 3) normalized array or None,
          "Right": (n_frames_with_right_hand, 21, 3) normalized array or None,
        }
    Frames where a given hand isn't detected are simply skipped for that
    hand - the sequence is indexed by "frames the hand was visible in",
    not by wall-clock frame number.

    Normalization is applied ONCE per hand, across its whole sequence
    (see normalize_landmark_sequence) - not per individual frame - so
    that whole-hand travel through space during the sign is preserved
    instead of being reset away on every frame.
    """
    cap = cv2.VideoCapture(video_path)
    raw_sequences = {"Left": [], "Right": []}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = to_mp_image(rgb_frame)

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                raw = landmarks_to_array(hand_landmarks)
                raw_sequences[label].append(raw)

    cap.release()

    normalized = {}
    for label, frames in raw_sequences.items():
        if frames:
            raw_stack = np.stack(frames)  # (n_frames, 21, 3)
            normalized[label] = normalize_landmark_sequence(raw_stack)
        else:
            normalized[label] = None

    return normalized


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} not found. Run select_vocabulary_clips.py first.")
        return

    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    detector = create_detector(running_mode=RunningMode.IMAGE, num_hands=2)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} clips...")
    start = time.time()
    n_ok, n_no_hand, n_missing_file, n_already_done = 0, 0, 0, 0

    for i, row in enumerate(rows, 1):
        video_path = os.path.join(CLIPS_DIR, row["split"], row["label"], row["video_file"])

        base_name = os.path.splitext(row["video_file"])[0]
        out_name = f"{row['split']}__{row['label']}__{row['participant']}__{base_name}.npz"
        out_path = os.path.join(LANDMARKS_DIR, out_name)

        if os.path.exists(out_path):
            n_already_done += 1
            continue

        if not os.path.exists(video_path):
            n_missing_file += 1
            continue

        sequences = extract_video_landmarks(detector, video_path)

        if sequences["Left"] is None and sequences["Right"] is None:
            n_no_hand += 1
            continue

        np.savez_compressed(
            out_path,
            left_hand=sequences["Left"] if sequences["Left"] is not None else np.empty((0, 21, 3), dtype=np.float32),
            right_hand=sequences["Right"] if sequences["Right"] is not None else np.empty((0, 21, 3), dtype=np.float32),
            label=row["label"],
            split=row["split"],
            participant=row["participant"],
        )
        n_ok += 1

        if i % 50 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta_min = (len(rows) - i) / rate / 60 if rate > 0 else 0
            print(f"  {i}/{len(rows)} processed ({elapsed:.0f}s elapsed, ~{eta_min:.1f} min remaining)...")

    detector.close()
    print(f"\nDone in {time.time()-start:.0f}s.")
    print(f"  newly extracted:        {n_ok}")
    print(f"  already done (skipped): {n_already_done}")
    print(f"  no hand detected:       {n_no_hand}")
    print(f"  missing video file:     {n_missing_file}")
    print(f"\nLandmark files saved to: {LANDMARKS_DIR}")


if __name__ == "__main__":
    main()
