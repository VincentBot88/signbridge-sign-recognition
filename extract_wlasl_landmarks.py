"""
SignBridge - batch landmark extraction for the downloaded WLASL clips.

Mirrors extract_clip_landmarks.py exactly - same detector, same
normalize_landmark_sequence, same unflipped-frame convention - by
importing extract_video_landmarks() directly from that script instead of
re-implementing it, so WLASL landmarks can't silently drift out of parity
with the ASL Citizen ones (same "single source of truth" reasoning as
build_clip_features() in build_feature_vectors_v2.py).

Reads data/wlasl_clips_manifest.csv (written by download_wlasl_clips.py)
instead of data/clips_manifest.csv, and writes into the SAME
data/landmarks/ folder as the ASL Citizen clips. build_feature_vectors_v2.py
and mirror_augment.py both already scan that whole folder generically -
label/split/participant come from each .npz's own stored fields, not from
a manifest - so nothing else in the pipeline needs to change for these to
flow through automatically (features get rebuilt, and mirror_augment.py
will pick these up as ordinary train clips to mirror too).

Every WLASL clip is forced to split="train" here, regardless of anything
in the source WLASL metadata - val/test must stay ASL-Citizen-only so
evaluation still reflects your real deployment conditions (your camera,
your lighting), not a mix that includes WLASL's YouTube/dictionary-site
footage.

IMPORTANT - before/after comparison:
Before running this the first time, back up your current features file so
you still have the ASL-Citizen-only number to compare against:

    copy data\\features_v2.csv data\\features_v2_aslcitizen_only.csv

Then, after this script + build_feature_vectors_v2.py have both run:

    python train_classifier.py data\\features_v2_aslcitizen_only.csv
    python train_classifier.py data\\features_v2.csv

...and compare the two val accuracies (and per-class recall, especially on
BATHROOM/WHO/NO/YES/HELP - the words WLASL adds the most to).

Run:
    python extract_wlasl_landmarks.py
"""

import csv
import os
import time

import numpy as np

from mp_hand_detector import create_detector, RunningMode
from extract_clip_landmarks import extract_video_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "wlasl_clips_manifest.csv")
CLIPS_DIR = os.path.join(BASE_DIR, "data", "wlasl_clips")
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} not found. Run download_wlasl_clips.py first.")
        return

    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    detector = create_detector(running_mode=RunningMode.IMAGE, num_hands=2)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} WLASL clips...")
    start = time.time()
    n_ok, n_no_hand, n_missing_file, n_already_done = 0, 0, 0, 0

    for i, row in enumerate(rows, 1):
        video_path = os.path.join(CLIPS_DIR, row["label"], row["video_file"])

        base_name = os.path.splitext(row["video_file"])[0]
        out_name = f"train__{row['label']}__{row['participant']}__wlasl_{base_name}.npz"
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
            split="train",                 # forced - WLASL clips never enter val/test
            participant=row["participant"],
        )
        n_ok += 1

        if i % 25 == 0:
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
    print("\nNext:")
    print("  python build_feature_vectors_v2.py")
    print("  python mirror_augment.py   (now also mirrors these WLASL train clips)")
    print("  python train_classifier.py data\\features_v2.csv")
    print("  (compare against data\\features_v2_aslcitizen_only.csv - see docstring)")


if __name__ == "__main__":
    main()
