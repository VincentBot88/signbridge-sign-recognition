"""
SignBridge - batch landmark extraction, v3 (hands + body, raw and frame-aligned).

WHAT CHANGED FROM v2 EXTRACTION, AND WHY
----------------------------------------
Three changes, all of which are prerequisites for body-relative features.
None of them can be done in the feature builder alone, which is why the
clips have to be re-processed.

1. POSE IS EXTRACTED TOO.
   A second Tasks-API detector (PoseLandmarker, see mp_body_detector.py)
   runs on every frame alongside HandLandmarker, giving shoulders, elbows,
   nose and mouth - the reference points the hands-only representation
   never had.

2. LANDMARKS ARE STORED RAW, NOT NORMALIZED.
   extract_clip_landmarks.py applied normalize_landmark_sequence() before
   saving, which baked one particular choice of coordinate frame into the
   stored data. That contradicted the stated reason for splitting
   extraction from feature building ("so we can re-derive different
   feature schemes later without re-running MediaPipe over every clip") -
   and a different coordinate frame is exactly what this change needs.
   v3 saves exactly what MediaPipe returned; every normalization decision
   now lives in build_feature_vectors_v3.py and is re-runnable in seconds.

3. FRAMES ARE ALIGNED ACROSS HANDS AND POSE.
   v2 stored each hand as "the frames that hand was visible in", with the
   two hands having independent lengths and no frame numbers. That makes
   it impossible to ask "where was the right hand relative to the
   shoulders at this moment" - there is no shared clock. v3 stores every
   array on the video's own frame axis, length T, with NaN on frames where
   that landmark set was not detected. Nothing is dropped, nothing is
   silently re-indexed.

The stored arrays are therefore:
    left_hand   (T, 21, 3)  raw MediaPipe coords, NaN where not detected
    right_hand  (T, 21, 3)  same
    pose        (T, 11, 3)  the mp_body_detector.POSE_KEEP subset
plus frame_w / frame_h (needed for aspect correction), n_frames, and the
usual label / split / participant.

Writes to data/landmarks_v3/ - the v2 data/landmarks/ folder is left
completely untouched, so the current 0.78 model and its inputs stay
reproducible while this is evaluated.

COST
----
Two detectors per frame instead of one, so budget roughly double the v2
extraction time - on the order of an hour for the full ~1,800 clips on a
laptop CPU. It is resumable (already-written .npz files are skipped), so
it can be stopped and restarted freely.

Measured on the pilot: ~7.3 s/clip for ASL Citizen, ~5.3 s/clip for WLASL,
so the full 1,022 + ~380 clips is on the order of 2.5 hours. Extracting
only train+val cuts that to roughly 1h15m, because the 416 test clips are
41% of the ASL Citizen set and must not be looked at until the single
final evaluation anyway. Extract them later, right before running
evaluate_on_test.py.

Run:
    python extract_landmarks_v3.py --splits train,val   # what you want while iterating
    python extract_landmarks_v3.py                      # everything, incl. test
    python extract_landmarks_v3.py --source citizen
    python extract_landmarks_v3.py --limit 60           # quick pilot
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from mp_hand_detector import create_detector, landmarks_to_array, to_mp_image, RunningMode
from mp_body_detector import create_pose_detector, pose_to_array, N_POSE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks_v3")

# Both datasets, described declaratively so the extraction loop itself is
# written once. WLASL is forced to split="train" for the same reason as in
# extract_wlasl_landmarks.py: val/test must stay ASL-Citizen-only so the
# evaluation reflects deployment-like footage, not YouTube dictionary clips.
SOURCES = {
    "citizen": dict(
        manifest=os.path.join(BASE_DIR, "data", "clips_manifest.csv"),
        clips_dir=os.path.join(BASE_DIR, "data", "clips"),
        # ASL Citizen clips live in clips/<split>/<label>/<file>
        path_parts=lambda row: (row["split"], row["label"], row["video_file"]),
        out_name=lambda row: f"{row['split']}__{row['label']}__{row['participant']}__"
                             f"{os.path.splitext(row['video_file'])[0]}.npz",
        split=lambda row: row["split"],
    ),
    "wlasl": dict(
        manifest=os.path.join(BASE_DIR, "data", "wlasl_clips_manifest.csv"),
        clips_dir=os.path.join(BASE_DIR, "data", "wlasl_clips"),
        path_parts=lambda row: (row["label"], row["video_file"]),
        out_name=lambda row: f"train__{row['label']}__{row['participant']}__wlasl_"
                             f"{os.path.splitext(row['video_file'])[0]}.npz",
        split=lambda row: "train",
    ),
}


def extract_clip(hand_detector, pose_detector, video_path):
    """
    Run both detectors over every frame of one clip (IMAGE mode - each
    frame independent, correct for a batch of many short unrelated clips).

    Returns (left, right, pose, frame_w, frame_h) where the three arrays
    are frame-aligned with NaN on frames the detector found nothing, or
    None if the video could not be opened / had no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    left_frames, right_frames, pose_frames = [], [], []
    frame_w = frame_h = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = to_mp_image(rgb_frame)

        left = np.full((21, 3), np.nan, dtype=np.float32)
        right = np.full((21, 3), np.nan, dtype=np.float32)

        hand_result = hand_detector.detect(mp_image)
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                arr = landmarks_to_array(hand_landmarks)
                if handedness[0].category_name == "Left":
                    left = arr
                else:
                    right = arr

        pose = np.full((N_POSE, 3), np.nan, dtype=np.float32)
        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            pose = pose_to_array(pose_result.pose_landmarks[0])

        left_frames.append(left)
        right_frames.append(right)
        pose_frames.append(pose)

    cap.release()

    if not left_frames:
        return None

    return (np.stack(left_frames), np.stack(right_frames), np.stack(pose_frames),
            frame_w, frame_h)


def resolve_renamed(video_path):
    """
    Fall back to a prefix match when the manifest's filename no longer
    matches what is on disk.

    24 WLASL clips were renamed by hand during the variant-split
    experiment to carry their variant label - 62964_21.avi became
    "62964_21-WHAT 1.avi" - and the manifest was never updated. Those
    clips ARE in the v2 training set (their .npz predates the rename), so
    without this fallback v3 would silently train on 24 fewer clips than
    v2 did, all of them in DEAF / DRINK / HOW / WHAT, and the v2 control
    would no longer be comparing like with like.

    Only an exact stem match or "<stem>-<something>" counts, so 62964_2
    cannot swallow 62964_21.
    """
    folder = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.isdir(folder):
        return video_path, False
    matches = [f for f in os.listdir(folder)
               if os.path.splitext(f)[0] == stem
               or os.path.splitext(f)[0].startswith(stem + "-")]
    if len(matches) == 1:
        return os.path.join(folder, matches[0]), True
    return video_path, False


def frac_detected(seq):
    """Fraction of frames in which this landmark set was detected at all."""
    if seq.shape[0] == 0:
        return 0.0
    return float(np.isfinite(seq[:, 0, 0]).mean())


def process_source(name, spec, hand_detector, pose_detector, limit=None, splits=None):
    if not os.path.exists(spec["manifest"]):
        print(f"[{name}] manifest not found, skipping: {spec['manifest']}")
        return

    with open(spec["manifest"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if splits:
        before = len(rows)
        rows = [r for r in rows if spec["split"](r) in splits]
        print(f"[{name}] {len(rows)}/{before} clips are in splits {sorted(splits)}")

    if limit:
        rows = rows[:limit]

    print(f"\n[{name}] processing {len(rows)} clips...")
    start = time.time()
    n_ok = n_no_hand = n_missing = n_done = n_unreadable = n_renamed = 0
    pose_fracs = []

    for i, row in enumerate(rows, 1):
        out_path = os.path.join(LANDMARKS_DIR, spec["out_name"](row))
        if os.path.exists(out_path):
            n_done += 1
            continue

        video_path = os.path.join(spec["clips_dir"], *spec["path_parts"](row))
        if not os.path.exists(video_path):
            video_path, was_renamed = resolve_renamed(video_path)
            if not os.path.exists(video_path):
                n_missing += 1
                continue
            n_renamed += was_renamed

        result = extract_clip(hand_detector, pose_detector, video_path)
        if result is None:
            n_unreadable += 1
            continue

        left, right, pose, frame_w, frame_h = result

        # Same rule as v2: a clip with no hand on any frame is not usable.
        # A clip with no POSE is still kept - the body block will simply be
        # absent for it, and the hand-local block still works.
        if frac_detected(left) == 0.0 and frac_detected(right) == 0.0:
            n_no_hand += 1
            continue

        pose_fracs.append(frac_detected(pose))

        np.savez_compressed(
            out_path,
            left_hand=left,
            right_hand=right,
            pose=pose,
            frame_w=frame_w,
            frame_h=frame_h,
            n_frames=left.shape[0],
            label=row["label"],
            split=spec["split"](row),
            participant=row["participant"],
        )
        n_ok += 1

        if i % 25 == 0:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed else 0
            eta = (len(rows) - i) / rate / 60 if rate else 0
            print(f"  {i}/{len(rows)} ({elapsed:.0f}s elapsed, ~{eta:.1f} min remaining)...")

    print(f"[{name}] done in {time.time()-start:.0f}s: "
          f"{n_ok} new, {n_done} already done, {n_no_hand} no hand, "
          f"{n_missing} missing file, {n_unreadable} unreadable"
          + (f", {n_renamed} found under a renamed filename" if n_renamed else ""))

    # This number decides whether the whole idea is viable for this source.
    # If pose is rarely detected - e.g. on tightly-cropped dictionary
    # footage that shows only hands and a chin - the body block will be
    # mostly absent for those clips and their contribution will be limited
    # to the hand-local features they already had.
    if pose_fracs:
        pose_fracs = np.array(pose_fracs)
        print(f"[{name}] POSE COVERAGE: mean {pose_fracs.mean():.1%} of frames per clip; "
              f"{(pose_fracs > 0.8).mean():.1%} of clips above 80%; "
              f"{(pose_fracs == 0).mean():.1%} of clips with no pose at all")


def main():
    global LANDMARKS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("citizen", "wlasl", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N clips of each source (pilot run)")
    ap.add_argument("--splits", default="all",
                    help="comma-separated splits to extract, e.g. 'train,val'. "
                         "The test split is 416 of the 1,022 ASL Citizen clips - 41%% of "
                         "the extraction time - and must not be looked at until the one "
                         "final evaluation, so there is no reason to extract it while "
                         "iterating. Default 'all'.")
    ap.add_argument("--out-dir", default=None,
                    help="write .npz files here instead of data/landmarks_v3. Use "
                         "data/landmarks_v3_test for the test split, so it never "
                         "sits beside the training data: "
                         "--source citizen --splits test --out-dir data/landmarks_v3_test")
    args = ap.parse_args()

    if args.out_dir:
        LANDMARKS_DIR = os.path.abspath(args.out_dir)

    splits = None if args.splits == "all" else {s.strip() for s in args.splits.split(",")}
    if splits and "test" in splits and not args.out_dir:
        print("NOTE: extracting the test split into data/landmarks_v3 alongside the "
              "training data. Prefer --out-dir data/landmarks_v3_test.")

    os.makedirs(LANDMARKS_DIR, exist_ok=True)

    hand_detector = create_detector(running_mode=RunningMode.IMAGE, num_hands=2)
    pose_detector = create_pose_detector(running_mode=RunningMode.IMAGE)

    names = ("citizen", "wlasl") if args.source == "both" else (args.source,)
    for name in names:
        process_source(name, SOURCES[name], hand_detector, pose_detector, args.limit, splits)

    hand_detector.close()
    pose_detector.close()

    print(f"\nLandmark files written to: {LANDMARKS_DIR}")
    print("\nNext:")
    print("  python mirror_augment_v3.py")
    print("  python build_feature_vectors_v3.py")
    print("  python train_classifier.py data\\features_v3.csv")


if __name__ == "__main__":
    main()
