"""
SignBridge - data QA: find clips that probably aren't what their label says.

WHY THIS EXISTS
---------------
Spot-checking the WLASL clips turned up two clear defects:

  - data/wlasl_clips/HELLO/70017_1.avi contains no signing at all. Its WLASL
    metadata asks for frames 4842-4923 of a YouTube lesson video. Frame
    indexing into a re-encoded YouTube video is fragile: WLASL declares
    fps=25, and if the copy yt-dlp fetched is 30fps, frame 4842 lands ~20%
    off - a different moment entirely.
  - data/wlasl_clips/BATHROOM/58740_3.avi is visibly corrupt AND its frame
    range is "1 -> -1", i.e. the WHOLE source video. For a YouTube lesson
    video that's minutes of footage - many signs, a talking head, hands
    down - all carrying one word's label.

Counting across the 238 recovered clips: 133 come from dictionary sites
(already single-sign clips, usually fine), but 65 are YouTube-whole-video
and 40 are YouTube-frame-range. So ~44% depend on frame indexing being
right. Mislabeled clips don't just fail to help - they actively teach the
classifier wrong things, which is the most likely reason adding 238 clips
moved val accuracy by 2 samples (0.782 -> 0.798).

Hand-reviewing ~1,260 clips is hours of watching and isn't repeatable.
This scores every clip automatically so you only eyeball the flagged ones.

THE SIGNALS
-----------
  hand_rate   fraction of the clip's frames where MediaPipe found a hand.
              THE strongest signal, and you already have it: the .npz files
              store one row per frame a hand was visible in, so
              len(sequence) / total_frames is exactly this. A real isolated
              sign is mostly hands-visible; a lesson video or a wrong
              segment is mostly not.
  duration_s  an isolated citation-form sign runs ~1-4s. 90s means a whole
              lesson video; <0.4s means a truncated extract.
  flat_frac   fraction of sampled frames dominated by a single flat colour.
              Catches the decode corruption (the half-green frames) that
              the downloader's first-frame-only validity check misses.
  centroid_z  distance from the ASL Citizen centroid for the SAME word, in
              standard deviations. High = this clip doesn't look like how
              that word is normally signed. Note this flags genuine ASL
              VARIANTS as well as errors - WLASL aggregates many signers
              and regional variants under one gloss, and a variant is not
              corruption. Judge these by eye, don't auto-drop them.

Runs over BOTH sources so "is ASL Citizen clean too?" is answered with
data rather than assumed. Expectation: ASL Citizen is a curated
isolated-sign corpus, one sign per clip by construction, so it should come
back clean - but that's a claim worth verifying, not trusting.

Run:
    python qa_clips.py

Writes data/qa_report.csv, worst first. Review the top of that list, delete
what's genuinely wrong, then rerun build_feature_vectors_v2.py and compare
again - if the WLASL result changes once the junk is gone, THAT is the
real answer to whether WLASL helps.
"""

import csv
import glob
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
ASL_CLIPS_DIR = os.path.join(BASE_DIR, "data", "clips")
WLASL_CLIPS_DIR = os.path.join(BASE_DIR, "data", "wlasl_clips")
REPORT_PATH = os.path.join(BASE_DIR, "data", "qa_report.csv")

# Thresholds - tuned to flag generously, since a flag just means "look at
# this", not "delete this".
MIN_HAND_RATE = 0.50     # below this, most of the clip has no hand in it
MIN_DURATION_S = 0.40
MAX_DURATION_S = 10.0    # an isolated sign is 1-4s; 10s+ is another animal
MAX_FLAT_FRAC = 0.25     # >25% of sampled frames dominated by one flat colour
FLAT_PIXEL_SHARE = 0.40  # a frame is "flat" if one quantised colour covers this much of it
SAMPLE_EVERY = 5         # check every Nth frame for flatness (speed)


def parse_npz_name(path):
    """
    ASL Citizen:  <split>__<label>__<participant>__<basename>.npz
    WLASL:        train__<label>__<participant>__wlasl_<basename>.npz
    Returns (source, label, participant, video_basename, split) or None.
    """
    base = os.path.basename(path)[:-4]
    if "_mirror" in base or "_aug" in base:
        return None                       # synthetic copies, nothing to QA
    parts = base.split("__", 3)
    if len(parts) != 4:
        return None
    split, label, participant, rest = parts
    if rest.startswith("wlasl_"):
        return "wlasl", label, participant, rest[len("wlasl_"):], "train"
    return "aslcitizen", label, participant, rest, split


def find_video(source, label, video_basename, split):
    if source == "wlasl":
        for ext in (".avi", ".mp4"):
            p = os.path.join(WLASL_CLIPS_DIR, label, video_basename + ext)
            if os.path.exists(p):
                return p
        return None
    for ext in (".mp4", ".avi"):
        p = os.path.join(ASL_CLIPS_DIR, split, label, video_basename + ext)
        if os.path.exists(p):
            return p
    return None


def scan_video(path):
    """Returns (total_frames, fps, flat_fraction). Decodes the whole clip."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0.0, 1.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 25.0

    total = 0
    checked = 0
    flat = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total % SAMPLE_EVERY == 0:
            # Quantise hard and ask how much of the frame is one single
            # colour. A healthy video frame is varied; a decode failure
            # tends to fill large regions with one flat value (the
            # half-green frames seen in some WLASL clips).
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_NEAREST)
            q = (small // 32).reshape(-1, 3)
            _, counts = np.unique(q, axis=0, return_counts=True)
            if counts.max() / q.shape[0] >= FLAT_PIXEL_SHARE:
                flat += 1
            checked += 1
        total += 1

    cap.release()
    return total, fps, (flat / checked if checked else 1.0)


def main():
    files = sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz")))
    if not files:
        print(f"ERROR: no .npz files in {LANDMARKS_DIR}")
        return

    rows = []
    print(f"Scanning {len(files)} landmark files (skipping mirrored/augmented copies)...")

    for i, npz_path in enumerate(files, 1):
        parsed = parse_npz_name(npz_path)
        if parsed is None:
            continue
        source, label, participant, basename, split = parsed

        d = np.load(npz_path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]
        hand_frames = max(left.shape[0], right.shape[0])

        video_path = find_video(source, label, basename, split)
        if video_path is None:
            rows.append(dict(source=source, label=label, participant=participant,
                             clip=basename, total_frames=0, hand_frames=hand_frames,
                             hand_rate=0.0, duration_s=0.0, flat_frac=1.0,
                             centroid_z=0.0, flags="VIDEO_MISSING", npz=os.path.basename(npz_path)))
            continue

        total, fps, flat_frac = scan_video(video_path)
        hand_rate = hand_frames / total if total else 0.0
        duration = total / fps if fps else 0.0

        flags = []
        if total == 0:
            flags.append("UNREADABLE")
        if hand_rate < MIN_HAND_RATE:
            flags.append("LOW_HAND_RATE")
        if duration > MAX_DURATION_S:
            flags.append("TOO_LONG")
        if 0 < duration < MIN_DURATION_S:
            flags.append("TOO_SHORT")
        if flat_frac > MAX_FLAT_FRAC:
            flags.append("CORRUPT_FRAMES")

        rows.append(dict(source=source, label=label, participant=participant,
                         clip=basename, total_frames=total, hand_frames=hand_frames,
                         hand_rate=round(hand_rate, 3), duration_s=round(duration, 2),
                         flat_frac=round(flat_frac, 3), centroid_z=0.0,
                         flags=",".join(flags), npz=os.path.basename(npz_path)))

        if i % 200 == 0:
            print(f"  {i}/{len(files)}...")

    # --- how far is each clip from how that word normally looks? ---
    # Centroid is built from ASL CITIZEN clips only, so it represents the
    # curated reference for the word, and WLASL clips get measured against
    # it rather than against their own noise.
    try:
        from build_feature_vectors_v2 import build_clip_features
        vecs = {}
        for r in rows:
            p = os.path.join(LANDMARKS_DIR, r["npz"])
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            l, rt = d["left_hand"], d["right_hand"]
            vecs[r["npz"]] = build_clip_features(l if l.shape[0] else None,
                                                 rt if rt.shape[0] else None)
        by_label = {}
        for r in rows:
            if r["source"] == "aslcitizen" and r["npz"] in vecs:
                by_label.setdefault(r["label"], []).append(vecs[r["npz"]])
        for r in rows:
            ref = by_label.get(r["label"])
            if not ref or r["npz"] not in vecs:
                continue
            ref = np.array(ref)
            centroid = ref.mean(axis=0)
            ref_d = np.linalg.norm(ref - centroid, axis=1)
            mu, sd = ref_d.mean(), ref_d.std() or 1.0
            z = (np.linalg.norm(vecs[r["npz"]] - centroid) - mu) / sd
            r["centroid_z"] = round(float(z), 2)
            if z > 3.0:
                r["flags"] = (r["flags"] + ",FAR_FROM_WORD" if r["flags"] else "FAR_FROM_WORD")
    except Exception as e:
        print(f"(skipped centroid analysis: {e})")

    def severity(r):
        return (len(r["flags"].split(",")) if r["flags"] else 0, -r["hand_rate"])
    rows.sort(key=severity, reverse=True)

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- summary ---
    flagged = [r for r in rows if r["flags"]]
    print(f"\n{'':<14}{'clips':>7}{'flagged':>9}{'  %':>6}")
    print("-" * 38)
    for src in ("aslcitizen", "wlasl"):
        sub = [r for r in rows if r["source"] == src]
        fl = [r for r in sub if r["flags"]]
        if sub:
            print(f"{src:<14}{len(sub):>7}{len(fl):>9}{len(fl)/len(sub)*100:>5.0f}%")

    from collections import Counter
    print("\nFlag counts:")
    c = Counter(f for r in flagged for f in r["flags"].split(",") if f)
    for k, v in c.most_common():
        print(f"  {v:>5}  {k}")

    print(f"\nWorst 15 (review these first):")
    print(f"  {'source':<11}{'word':<12}{'clip':<18}{'hand':>6}{'dur':>7}  flags")
    for r in rows[:15]:
        print(f"  {r['source']:<11}{r['label']:<12}{r['clip'][:17]:<18}"
              f"{r['hand_rate']:>6.2f}{r['duration_s']:>7.1f}  {r['flags']}")

    print(f"\nFull report: {REPORT_PATH}")
    print("\nAfter deleting bad clips, remove their .npz files too (the 'npz' column")
    print("names them), then rerun build_feature_vectors_v2.py and train_classifier.py.")


if __name__ == "__main__":
    main()
