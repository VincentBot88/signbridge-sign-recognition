"""
SignBridge - download & extract WLASL clips for the 28-word vocabulary.

Downloads source videos for the WLASL instances listed in
wlasl_instances_28word.csv (pulled directly from WLASL_v0.3.json,
dxli94/WLASL - not estimated), trims each to its labeled frame range,
crops to its labeled bbox (only when the bbox is meaningfully smaller than
the actual downloaded frame - WLASL's bbox field isn't a real crop for
every source, so this is checked rather than assumed), and writes one
short clip per instance to data/wlasl_clips/<LABEL>/<video_id>_<instance_id>.avi
(MJPG/AVI, not mp4 - see extract_clip()'s docstring for why).

These are ADDITIONAL data only - added on top of the existing ASL Citizen
clips, never a replacement. Every clip this produces ends up split="train"
downstream (see extract_wlasl_landmarks.py) - never val/test - so
evaluation stays on your original ASL Citizen signers only, matching the
same rule already applied to mirror_augment.py and the old jitter
augmentation.

WLASL's own README notes many source URLs (mostly YouTube) go dead over
time - failures here are expected, not a bug. Every attempt is logged to
data/wlasl_download_log.csv, and this script is safe to re-run: already-
downloaded clips are skipped, so a partial/interrupted run just resumes.

Requires (install into the same venv as the rest of the pipeline):
    pip install yt-dlp requests

BEFORE RUNNING: make sure wlasl_instances_28word.csv is in this same
folder (next to this script).

Run:
    python download_wlasl_clips.py
"""

import csv
import math
import os
import subprocess
import sys
import time

# Must be set BEFORE cv2 is imported. ffmpeg's own C-level logger writes
# straight to stderr and is not controlled by cv2's log level, so without
# this every corrupt/dead-link download prints a "moov atom not found"
# (or similar) line even though the code below detects and handles it
# cleanly. -8 = AV_LOG_QUIET.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import requests

# Cut the very noisy per-frame ffmpeg warnings OpenCV prints for anything
# it doesn't like (corrupt downloads, unsupported streams, etc). Every real
# failure is still fully captured in data/wlasl_download_log.csv - this
# only quiets stderr spam for cases we already handle and log cleanly.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_CSV = os.path.join(BASE_DIR, "wlasl_instances_28word.csv")
RAW_CACHE_DIR = os.path.join(BASE_DIR, "data", "wlasl_raw_cache")
CLIPS_DIR = os.path.join(BASE_DIR, "data", "wlasl_clips")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "wlasl_clips_manifest.csv")
LOG_PATH = os.path.join(BASE_DIR, "data", "wlasl_download_log.csv")

# Flash video - not decodable with cv2/ffmpeg here. A handful of older ASL
# dictionary sites (e.g. some aslpro entries) may still point at .swf; skip
# and log rather than fail confusingly deep inside OpenCV.
SKIP_EXTENSIONS = (".swf",)


def is_valid_video(path):
    """True if path opens and has at least one readable frame.

    Used both to sanity-check a freshly downloaded raw source video (some
    downloads land as an HTML error page, a partial file from an
    interrupted transfer, or another non-video response saved with a .mp4
    extension - all of which cv2 will happily "open" but then fail to
    decode, often as a "moov atom not found" error) and to make sure a
    CACHED raw file from an earlier run wasn't one of those broken ones -
    otherwise a corrupt cached download stays cached forever and every
    later run keeps failing on it for the same reason.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def is_youtube(url):
    return "youtube.com" in url or "youtu.be" in url


def download_youtube(url, video_id):
    """Download a YouTube source video via yt-dlp, cached by video_id."""
    existing = [f for f in os.listdir(RAW_CACHE_DIR) if f.startswith(f"yt_{video_id}.")]
    if existing:
        cached_path = os.path.join(RAW_CACHE_DIR, existing[0])
        if is_valid_video(cached_path):
            return cached_path
        os.remove(cached_path)  # corrupt/partial from an earlier run - redownload

    out_template = os.path.join(RAW_CACHE_DIR, f"yt_{video_id}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        # Native resolution on purpose: WLASL's bbox coordinates are in the
        # ORIGINAL video's pixel space, so downscaling here would put every
        # crop in the wrong coordinate system (extract_clip falls back to the
        # full frame when that happens, which works but loses the crop).
        "-f", "mp4/best",
        "-o", out_template,
        "--no-playlist",
        "--quiet", "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[:300]}")

    matches = [f for f in os.listdir(RAW_CACHE_DIR) if f.startswith(f"yt_{video_id}.")]
    if not matches:
        raise RuntimeError("yt-dlp reported success but produced no output file")
    out_path = os.path.join(RAW_CACHE_DIR, matches[0])
    if not is_valid_video(out_path):
        os.remove(out_path)
        raise RuntimeError("downloaded file is not a valid/decodable video")
    return out_path


def download_direct(url, video_id):
    """Download a direct video URL (non-YouTube ASL dictionary site), cached by video_id."""
    ext = os.path.splitext(url.split("?")[0])[1].lower() or ".mp4"
    if ext in SKIP_EXTENSIONS:
        raise RuntimeError(f"unsupported format {ext} (Flash/swf - not decodable here)")

    out_path = os.path.join(RAW_CACHE_DIR, f"direct_{video_id}{ext}")
    if os.path.exists(out_path):
        if is_valid_video(out_path):
            return out_path
        os.remove(out_path)  # corrupt/partial from an earlier run - redownload

    resp = requests.get(url, stream=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    os.replace(tmp_path, out_path)

    if not is_valid_video(out_path):
        # Common cause: the "video" URL actually served an HTML error page
        # (dead link, moved content, requires a referer/cookie the site
        # checks) rather than real video bytes - looks like a normal
        # download but fails to decode as one.
        os.remove(out_path)
        raise RuntimeError("downloaded file is not a valid/decodable video (dead link or non-video response?)")
    return out_path


def extract_clip(raw_video_path, frame_start, frame_end, bbox, out_path):
    """
    Read [frame_start, frame_end] (frame_end == -1 means read to EOF) from
    raw_video_path, crop to bbox IF the bbox is meaningfully smaller than
    the actual frame, and write the result as a short clip.

    Reads sequentially from frame 0 and discards frames before frame_start,
    rather than seeking with CAP_PROP_POS_FRAMES - some codecs/containers
    seek inaccurately, and silently grabbing the wrong frame range would
    mislabel training data, which matters more here than the extra decode
    time costs on these short source clips.

    Output uses MJPG/.avi rather than mp4v/.mp4: mp4v's OpenCV write path
    depends on the ffmpeg plugin being fully functional for the target
    build/platform, and on some Windows opencv-python installs it silently
    fails per-frame ("Failed to write frame") without raising a Python
    exception. MJPG-in-AVI is natively supported by OpenCV's own writer on
    every platform this pipeline runs on, and the output only needs to be
    re-readable by cv2.VideoCapture for landmark extraction - not a
    distributable video - so the format choice doesn't matter otherwise.

    Returns the number of frames written AND verified readable back.
    """
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        raise RuntimeError("could not open downloaded video (corrupt/unsupported)")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = 25

    x1, y1, x2, y2 = bbox
    writer = None
    out_size = None      # (w, h) taken from the first frame actually written
    use_crop = False
    idx = 0
    n_written = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_end != -1 and idx > frame_end:
            break

        if idx >= frame_start:
            if writer is None:
                # Decide everything from the FIRST REAL DECODED FRAME, never
                # from CAP_PROP_FRAME_WIDTH/HEIGHT. Container metadata lies
                # often enough, and a writer opened at one size silently
                # rejects every frame of another size (OpenCV's write() does
                # not raise) - which is exactly how ~50 clips previously came
                # out as empty 257-byte files counted as successes.
                fh, fw = frame.shape[:2]

                # WLASL's bbox is in its ORIGINAL source video's pixel space.
                # When the file we actually downloaded is a different
                # resolution, those coordinates don't apply - e.g. a bbox of
                # (207,1,930,720) against a 854x480 download. Rather than
                # guess a rescale (a wrong crop can cut the signer's hands
                # out of frame entirely, which silently poisons the training
                # data), only crop when the box genuinely fits, and fall back
                # to the full frame otherwise. MediaPipe finds hands in full
                # frames fine; the crop is an optimization, not a necessity.
                use_crop = (
                    0 <= x1 < x2 <= fw and 0 <= y1 < y2 <= fh
                    and (x2 - x1) >= 2 and (y2 - y1) >= 2
                )

                first = frame[y1:y2, x1:x2] if use_crop else frame
                h, w = first.shape[:2]
                w -= w % 2   # some codecs reject odd dimensions
                h -= h % 2
                if w < 2 or h < 2:
                    cap.release()
                    raise RuntimeError(f"degenerate output size {w}x{h}")
                out_size = (w, h)

                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, out_size)
                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"VideoWriter failed to open at {w}x{h} @ {fps:.0f}fps")

            piece = frame[y1:y2, x1:x2] if use_crop else frame
            if (piece.shape[1], piece.shape[0]) != out_size:
                piece = cv2.resize(piece, out_size)
            writer.write(piece)
            n_written += 1
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if n_written == 0:
        return 0

    # Trust but verify: confirm the file we just wrote is actually
    # readable, not just that we called .write() n_written times (OpenCV's
    # writer.write() doesn't raise or return a usable status on failure -
    # it can silently drop frames while this loop keeps counting them).
    cap2 = cv2.VideoCapture(out_path)
    n_readable = 0
    if cap2.isOpened():
        while True:
            ok, _ = cap2.read()
            if not ok:
                break
            n_readable += 1
    cap2.release()

    if n_readable == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"wrote {n_written} frames but 0 are readable back - writer failure")

    return n_readable


def main():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    if not os.path.exists(INSTANCES_CSV):
        print(f"ERROR: {INSTANCES_CSV} not found. Place it next to this script.")
        return

    with open(INSTANCES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} WLASL instances to fetch across {len(set(r['label'] for r in rows))} words")

    manifest_rows = []
    log_rows = []
    n_ok = n_fail = n_skipped = 0
    start = time.time()

    for i, row in enumerate(rows, 1):
        label = row["label"]
        video_id = row["video_id"]
        instance_id = row["instance_id"]
        out_dir = os.path.join(CLIPS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{video_id}_{instance_id}.avi"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            n_skipped += 1
            manifest_rows.append({
                "label": label, "gloss": row["wlasl_gloss"],
                "participant": f"WLASL_{row['signer_id']}",
                "video_file": out_name, "split": "train",
            })
            continue

        try:
            url = row["url"]
            if is_youtube(url):
                raw_path = download_youtube(url, video_id)
            else:
                raw_path = download_direct(url, video_id)

            bbox = (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"]))
            n_frames = extract_clip(raw_path, int(row["frame_start"]), int(row["frame_end"]), bbox, out_path)

            if n_frames == 0:
                raise RuntimeError("0 frames written (frame_start past end of video?)")

            manifest_rows.append({
                "label": label, "gloss": row["wlasl_gloss"],
                "participant": f"WLASL_{row['signer_id']}",
                "video_file": out_name, "split": "train",
            })
            log_rows.append({"video_id": video_id, "instance_id": instance_id, "label": label,
                              "status": "ok", "detail": f"{n_frames} frames"})
            n_ok += 1
        except Exception as e:
            log_rows.append({"video_id": video_id, "instance_id": instance_id, "label": label,
                              "status": "FAILED", "detail": str(e)[:200]})
            n_fail += 1

        if i % 25 == 0:
            elapsed = time.time() - start
            print(f"  {i}/{len(rows)} ({elapsed:.0f}s elapsed) - ok={n_ok} failed={n_fail} skipped={n_skipped}")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "gloss", "participant", "video_file", "split"])
        w.writeheader()
        w.writerows(manifest_rows)

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "instance_id", "label", "status", "detail"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nDone in {time.time()-start:.0f}s.")
    print(f"  downloaded ok:                   {n_ok}")
    print(f"  already existed (skipped):       {n_skipped}")
    print(f"  failed (dead link/unsupported):  {n_fail}")
    print(f"\nManifest: {MANIFEST_PATH} ({len(manifest_rows)} usable clips)")
    print(f"Failure log: {LOG_PATH}")
    print("\nSpot-check a handful of the output clips in data/wlasl_clips/ before trusting them -")
    print("frame-range/crop metadata quality varies across WLASL's source sites.")
    print("\nNext: python extract_wlasl_landmarks.py")


if __name__ == "__main__":
    main()
