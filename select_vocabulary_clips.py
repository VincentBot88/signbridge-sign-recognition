"""
SignBridge - select & extract only the vocabulary clips we need from the
ASL Citizen zip, instead of unpacking the full ~43GB archive.

Reads train.csv / val.csv / test.csv, filters to our chosen 28-word
vocabulary (merging sign variants like EAT1/EAT2 into one label), and
extracts just those matching .mp4 files from ASL_Citizen.zip into:

    data/clips/<split>/<label>/<original_filename>.mp4

Also writes data/clips_manifest.csv listing every extracted clip with its
label, split, and signer id - that manifest is what extract_clip_landmarks.py
reads next.

BEFORE RUNNING: edit ZIP_PATH and CSV_DIR below to match where you saved
things on your machine.
"""

import csv
import os
import shutil
import zipfile

# ---- EDIT THESE TWO PATHS FOR YOUR MACHINE ----
ZIP_PATH = r"C:\Users\aweso\Downloads\ASL_Citizen.zip"
CSV_DIR = r"C:\Users\aweso\SignBridge\csvs"
# ------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clips")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clips_manifest.csv")

# base label -> ASL Citizen gloss variant(s) merged into that label.
# (Decision: variants of the same word are merged into one class so the
# kiosk accepts either natural way of signing it.)
VOCABULARY = {
    "HELLO": ["HELLO"],
    "BYE": ["BYE"],
    "THANKYOU": ["THANKYOU"],
    "PLEASE": ["PLEASE"],
    "SORRY": ["SORRY"],
    "YES": ["YES"],
    "NO": ["NO"],
    "HELP": ["HELP"],
    "STOP": ["STOP"],
    "WAIT": ["WAIT"],
    "MORE": ["MORE"],
    "WATER": ["WATER"],
    "BATHROOM": ["BATHROOM"],
    "EAT": ["EAT1", "EAT2"],
    "DRINK": ["DRINK1", "DRINK2"],
    "HUNGRY": ["HUNGRY"],
    "HURT": ["HURT"],
    "PAIN": ["PAIN"],
    "SICK": ["SICK"],
    "NAME": ["NAME"],
    "WHERE": ["WHERE"],
    "WHAT": ["WHAT1", "WHAT2"],
    "WHO": ["WHO"],
    "HOW": ["HOW1", "HOW2"],
    "WHY": ["WHY"],
    "UNDERSTAND": ["UNDERSTAND"],
    "DEAF": ["DEAF1", "DEAF2"],
    "INTERPRETER": ["INTERPRETER"],
}

GLOSS_TO_LABEL = {g: label for label, glosses in VOCABULARY.items() for g in glosses}


def load_split_rows(csv_path, split_name):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row["Gloss"]
            if gloss in GLOSS_TO_LABEL:
                rows.append({
                    "label": GLOSS_TO_LABEL[gloss],
                    "gloss": gloss,
                    "participant": row["Participant ID"],
                    "video_file": row["Video file"],
                    "split": split_name,
                })
    return rows


def main():
    all_rows = []
    for split in ("train", "val", "test"):
        csv_path = os.path.join(CSV_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.")
            continue
        rows = load_split_rows(csv_path, split)
        print(f"{split}: {len(rows)} matching clips")
        all_rows.extend(rows)

    print(f"\nTotal clips to extract: {len(all_rows)}")

    if not os.path.exists(ZIP_PATH):
        print(f"\nERROR: zip not found at {ZIP_PATH}")
        print("Edit ZIP_PATH at the top of this script and rerun.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nOpening {ZIP_PATH} (reading the file index of a ~43GB archive "
          f"can take a few seconds, this is normal)...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        sample_video_names = [n for n in names if n.lower().endswith(".mp4")][:5]
        print("Sample video paths found inside the zip:", sample_video_names)

        # Map basename -> full path inside the zip, so we can look up each
        # wanted clip by filename regardless of its internal folder prefix.
        name_by_basename = {}
        for n in names:
            if n.lower().endswith(".mp4"):
                name_by_basename[os.path.basename(n)] = n

        extracted, skipped_existing, missing = 0, 0, []
        for i, row in enumerate(all_rows, 1):
            target_name = row["video_file"]
            match = name_by_basename.get(target_name)
            if match is None:
                missing.append(target_name)
                continue

            dest_dir = os.path.join(OUTPUT_DIR, row["split"], row["label"])
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, target_name)

            if os.path.exists(dest_path):
                skipped_existing += 1
                extracted += 1
                continue

            with zf.open(match) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

            if i % 100 == 0:
                print(f"  {i}/{len(all_rows)} done...")

    print(f"\nDone. {extracted} clips available ({skipped_existing} already existed "
          f"from a previous run), {len(missing)} not found in the zip.")
    if missing:
        print("First few missing filenames (check the sample paths printed above "
              "match the expected naming):", missing[:5])

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "gloss", "participant", "video_file", "split"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
