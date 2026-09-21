"""
SignBridge - does splitting a merged-variant word back into its variants help?

THE QUESTION
------------
Five words in the vocabulary merge two distinct ASL Citizen glosses into one
label (select_vocabulary_clips.py): WHAT1+WHAT2, DEAF1/2, DRINK1/2, EAT1/2,
HOW1/2. Those are genuinely different signs for the same English word. Does
asking the classifier to put two disjoint motion patterns under one label
cost accuracy?

WHAT RUNNING IT ON *WHAT* ALREADY SHOWED
----------------------------------------
    A  merged, balanced weights            0.798    WHAT 2/7 right,  4 claimed
    B  split,  balanced weights            0.790    WHAT 7/7 right, 16 claimed
    C  split,  parent-inherited weights    0.774    WHAT 2/7 right,  5 claimed

B's perfect recall looked like the split working. It wasn't. `class_weight=
"balanced"` gives every CLASS equal total mass in the loss, so splitting one
word into three sub-classes handed it three shares - and because sub-class
probabilities are summed back at output, that inflated mass landed entirely
on WHAT (129.7 vs 46.4 for any other word, 2.8x). The model simply claimed
WHAT four times as often, and catching more true WHATs by guessing WHAT more
is not the same as learning it.

Configuration C proves it: give each sub-class its parent's weight - so the
sub-classes together carry exactly one class's worth of mass, as they did
when merged - and WHAT drops straight back to 2/7. The split contributes
nothing; all of B's apparent gain was the weighting artifact.

Lesson worth keeping: recall alone is a trap. Always check how often a class
was predicted at all. 16 claims for 7 true instances is a liberal guess, not
a model that has learned the class.

WHY C IS THE HONEST COMPARISON
------------------------------
To give parent p's sub-classes a combined mass of n/n_merged_classes split in
proportion to their sizes, each sub-class needs weight
n/(n_merged_classes * count_p) - which is exactly the weight the merged parent
had. So each sub-class simply inherits its parent's weight.

Run:
    python variant_split_test.py                          # WHAT only
    python variant_split_test.py --words WHAT DEAF DRINK EAT HOW --each
        ^ tests each word INDEPENDENTLY (one split at a time) so any accuracy
          change is attributable to that word alone, and prints one summary.
    python variant_split_test.py --words WHAT HOW         # split both at once
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "clips_manifest.csv")
WLASL_CLIPS_DIR = os.path.join(BASE_DIR, "data", "wlasl_clips")

# '62964_21.avi'                      -> no variant
# '62964_21-DEAF.avi'                 -> DEAF1   (see convention note below)
# '62964_21-WHAT 2.avi' / '_WHAT2'    -> WHAT2
_CLIP_NAME = re.compile(r"^(?P<id>\d+_\d+)(?:[-_ ]+(?P<var>.+))?$")


def scan_wlasl_clips():
    """
    Look at what's actually on disk in data/wlasl_clips/<LABEL>/.

    Two things come from this:
      - which WLASL clips still EXIST (clips deleted during manual review leave
        their .npz behind, and training on a landmark file whose video you threw
        away is exactly the kind of silent staleness that invalidates a
        comparison)
      - any variant labels added by renaming. WLASL ships every variant under
        one gloss with no way to tell them apart, so hand-labelling by filename
        is the only source for these.

    VARIANT NAMING follows ASL Citizen's own convention, which is what the
    clips in data/clips/ already use:

        '<id>-DEAF.avi'     -> DEAF1     (bare word IS variant 1)
        '<id>-DEAF 2.avi'   -> DEAF2

    Verified against the real manifest: '7322377994548119-DEAF.mp4' carries
    gloss DEAF1, and '5658430416463056-HOW 2.mp4' carries HOW2. Requiring a
    trailing digit would silently drop every variant-1 label as "unknown",
    which is exactly what happened the first time round.

    Anything after the id that isn't the folder's own word, optionally followed
    by a number, is left as None rather than guessed at.

    Returns {clip_id: variant_or_None}.
    """
    found = {}
    if not os.path.isdir(WLASL_CLIPS_DIR):
        return found
    for label in os.listdir(WLASL_CLIPS_DIR):
        d = os.path.join(WLASL_CLIPS_DIR, label)
        if not os.path.isdir(d):
            continue
        word = label.upper()
        for fn in os.listdir(d):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in (".avi", ".mp4"):
                continue
            m = _CLIP_NAME.match(stem.strip())
            if not m:
                continue
            variant = None
            raw = m.group("var")
            if raw:
                v = re.sub(r"[ _\-]+", "", raw).upper()     # 'DEAF 2' -> 'DEAF2'
                if v.startswith(word):
                    rest = v[len(word):]
                    if rest == "":
                        variant = f"{word}1"                # bare word = variant 1
                    elif rest.isdigit():
                        variant = f"{word}{rest}"
                    # anything else: unrecognised, leave as None
            found[m.group("id")] = variant
    return found


def parse_npz_name(path):
    """<split>__<label>__<participant>__<basename>[_mirror].npz"""
    base = os.path.basename(path)[:-4]
    if base.endswith("_mirror"):
        base = base[: -len("_mirror")]
    parts = base.split("__", 3)
    return parts if len(parts) == 4 else None


def balanced_weights(y, classes):
    """sklearn's 'balanced': n / (n_classes * count) - equal total mass per class."""
    n = len(y)
    return {c: n / (len(classes) * max((y == c).sum(), 1)) for c in classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=["WHAT"])
    ap.add_argument("--each", action="store_true",
                    help="test each word independently instead of splitting all at once")
    args = ap.parse_args()
    words = [w.upper() for w in args.words]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    from build_feature_vectors_v2 import LABEL_MERGES, build_clip_features

    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"ERROR: {MANIFEST_PATH} not found - needed for the variant labels.")
    gloss_of = {os.path.splitext(r["video_file"])[0]: r["gloss"]
                for r in csv.DictReader(open(MANIFEST_PATH, newline="", encoding="utf-8"))}

    wlasl_on_disk = scan_wlasl_clips()
    n_labelled = sum(1 for v in wlasl_on_disk.values() if v)
    print(f"WLASL clips on disk: {len(wlasl_on_disk)}"
          + (f"  ({n_labelled} carry a hand-added variant label in the filename)"
             if n_labelled else ""))

    # ---- load once; labels get rebuilt per experiment ----
    feats, raw_labels, clip_ids, splits = [], [], [], []
    n_orphaned = 0
    orphan_ids = set()
    for path in sorted(glob.glob(os.path.join(LANDMARKS_DIR, "*.npz"))):
        parsed = parse_npz_name(path)
        if parsed is None:
            continue
        split, raw_label, _participant, basename = parsed

        # WLASL landmark files are named ...__wlasl_<id>.npz
        is_wlasl = basename.startswith("wlasl_")
        clip_id = basename[len("wlasl_"):] if is_wlasl else basename

        # Drop landmark files whose video was deleted during manual review, so
        # EVERY configuration below trains on the identical set of clips.
        if is_wlasl and clip_id not in wlasl_on_disk:
            n_orphaned += 1
            orphan_ids.add(clip_id)
            continue

        d = np.load(path, allow_pickle=True)
        left, right = d["left_hand"], d["right_hand"]
        left = left if left.shape[0] > 0 else None
        right = right if right.shape[0] > 0 else None
        if left is None and right is None:
            continue
        feats.append(build_clip_features(left, right))
        raw_labels.append(raw_label)
        clip_ids.append((clip_id, is_wlasl))
        splits.append(split)

    if n_orphaned:
        print(f"Skipped {n_orphaned} landmark file(s) whose clip was deleted "
              f"(counting mirrors) - excluded from every configuration here.")
        print(f"  NOTE: build_feature_vectors_v2.py does NOT skip them, so your main")
        print(f"  pipeline is still training on clips you deleted. To fix:")
        for cid in sorted(orphan_ids):
            print(f"      Remove-Item data\\landmarks\\*wlasl_{cid}*.npz")
        print(f"      python build_feature_vectors_v2.py")

    X = np.stack(feats)
    raw_labels = np.array(raw_labels)
    splits = np.array(splits)
    tr, va = splits == "train", splits == "val"
    n_val = int(va.sum())

    y_merged = np.array([LABEL_MERGES.get(r, r) for r in raw_labels])
    out_classes = sorted(set(y_merged))
    w_merged = balanced_weights(y_merged[tr], out_classes)

    print(f"Loaded {len(X)} clips  |  train {tr.sum()}  val {n_val}")

    def build_split_labels(split_words):
        """
        Variant label priority:
          1. ASL Citizen's own gloss column  (WHAT1 / WHAT2)
          2. a variant hand-added to a WLASL filename ('62964_21-WHAT 1.avi')
          3. '<WORD>_U' - genuinely unknown, rather than guessed
        """
        y, s2p = [], {}
        n_hand, n_unknown = 0, 0
        for raw, (clip_id, is_wlasl) in zip(raw_labels, clip_ids):
            out = LABEL_MERGES.get(raw, raw)
            if raw in split_words:
                gloss = None if is_wlasl else gloss_of.get(clip_id)
                if gloss:
                    lab = gloss
                elif is_wlasl and wlasl_on_disk.get(clip_id):
                    lab = wlasl_on_disk[clip_id]
                    n_hand += 1
                else:
                    lab = f"{raw}_U"
                    n_unknown += 1
            else:
                lab = out
            s2p[lab] = out
            y.append(lab)
        return np.array(y), s2p, n_unknown, n_hand

    def run(y, weights, s2p):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight=weights, random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[va])
        collapsed = np.zeros((n_val, len(out_classes)))
        for j, sub in enumerate(clf.classes_):
            collapsed[:, out_classes.index(s2p.get(sub, sub))] += proba[:, j]
        pred = np.array(out_classes)[collapsed.argmax(1)]
        return pred, accuracy_score(y_merged[va], pred)

    # baseline, identical for every experiment
    pred_a, acc_a = run(y_merged, w_merged, {c: c for c in out_classes})
    print(f"\nBaseline A (all variants merged): val accuracy {acc_a:.3f}")
    se = (0.25 / n_val) ** 0.5
    print(f"1 standard error on {n_val} samples is {se:.3f} ({round(se*n_val)} clips) "
          f"- smaller differences are noise\n")

    def word_stats(pred, parent):
        mask = y_merged[va] == parent
        tot = int(mask.sum())
        hit = int((pred[mask] == parent).sum())
        claimed = int((pred == parent).sum())
        return hit, tot, claimed

    if args.each:
        print(f"Each word split INDEPENDENTLY, with parent-inherited weights "
              f"(configuration C):\n")
        print(f"  {'word':<10}{'train clips':>26}{'overall val':>13}{'vs A':>8}"
              f"{'word: A':>16}{'word: split':>16}")
        print("  " + "-" * 87)
        results = []
        for w in words:
            parent = LABEL_MERGES.get(w, w)
            y_s, s2p, n_unk, n_hand = build_split_labels({w})
            subs = sorted({lab for lab in set(y_s[tr]) if s2p[lab] == parent})
            if len(subs) < 2:
                print(f"  {parent:<10}{'(only one gloss - nothing to split)':>26}")
                continue
            counts = "/".join(str(int((y_s[tr] == s).sum())) for s in subs)
            sub_classes = sorted(set(y_s[tr]))
            w_corr = {s: w_merged[s2p[s]] for s in sub_classes}
            pred_c, acc_c = run(y_s, w_corr, s2p)

            ha, ta, ca = word_stats(pred_a, parent)
            hc, tc, cc = word_stats(pred_c, parent)
            results.append((parent, acc_c))
            print(f"  {parent:<10}{counts:>26}{acc_c:>13.3f}{(acc_c-acc_a)*n_val:>+8.0f}"
                  f"{f'{ha}/{ta} ({ca} claimed)':>16}{f'{hc}/{tc} ({cc} claimed)':>16}")

        print(f"\n  ('train clips' = sub-class sizes after the split, mirrors included;")
        print(f"   'claimed' = how many val clips the model assigned that word overall -")
        print(f"   recall rising while claims balloon means over-prediction, not learning.)")

        if results:
            best_word, best_acc = max(results, key=lambda t: t[1])
            if best_acc > acc_a + se:
                print(f"\n  => {best_word} is the only split clearing 1 SE above baseline "
                      f"({best_acc:.3f} vs {acc_a:.3f}). Worth a closer look.")
            else:
                print(f"\n  => No split beats the merged baseline by more than noise. "
                      f"Best was {best_word} at {best_acc:.3f} vs {acc_a:.3f}.")
                print(f"     Merging the variants is the right call for all of them.")
        return

    # ---- single experiment: A / B / C on the given word set ----
    split_words = set(words)
    y_s, s2p, n_unk, n_hand = build_split_labels(split_words)
    sub_classes = sorted(set(y_s[tr]))
    w_bal = balanced_weights(y_s[tr], sub_classes)
    w_corr = {s: w_merged[s2p[s]] for s in sub_classes}

    bits = []
    if n_hand:
        bits.append(f"{n_hand} WLASL clips variant-labelled by filename")
    if n_unk:
        bits.append(f"{n_unk} still unknown -> '_U'")
    print(f"Splitting: {', '.join(sorted(split_words))}"
          + (f"   ({'; '.join(bits)})" if bits else ""))
    print(f"\nTotal loss mass per word (weight x clips):")
    print(f"  {'':<12}{'clips':>7}{'B: balanced':>14}{'C: corrected':>15}")
    for w in sorted(split_words):
        parent = LABEL_MERGES.get(w, w)
        subs = [s for s in sub_classes if s2p[s] == parent]
        mb = sum(w_bal[s] * (y_s[tr] == s).sum() for s in subs)
        mc = sum(w_corr[s] * (y_s[tr] == s).sum() for s in subs)
        cnt = sum(int((y_s[tr] == s).sum()) for s in subs)
        print(f"  {parent:<12}{cnt:>7}{mb:>14.1f}{mc:>15.1f}")
    other = next(c for c in out_classes
                 if c not in {LABEL_MERGES.get(w, w) for w in split_words})
    om = w_merged[other] * (y_merged[tr] == other).sum()
    print(f"  {'(any other)':<12}{int((y_merged[tr] == other).sum()):>7}"
          f"{om:>14.1f}{om:>15.1f}   <- what one class should carry")

    pred_b, acc_b = run(y_s, w_bal, s2p)
    pred_c, acc_c = run(y_s, w_corr, s2p)

    print(f"\n{'configuration':<44}{'val accuracy'}")
    print("-" * 60)
    print(f"  A  merged, balanced weights               {acc_a:.3f}")
    print(f"  B  split,  balanced weights               {acc_b:.3f}   ({(acc_b-acc_a)*n_val:+.0f} clips vs A)")
    print(f"  C  split,  parent-inherited weights       {acc_c:.3f}   ({(acc_c-acc_a)*n_val:+.0f} clips vs A)")

    print(f"\nEffect on the split word(s) - recall, and times predicted at all:")
    print(f"  {'word':<12}{'A merged':>22}{'B balanced':>22}{'C corrected':>22}")
    for w in sorted(split_words):
        parent = LABEL_MERGES.get(w, w)
        cells = []
        for pred in (pred_a, pred_b, pred_c):
            h, t, c = word_stats(pred, parent)
            cells.append(f"{h}/{t} right, {c} claimed")
        print(f"  {parent:<12}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

    best = max([("A", acc_a, pred_a), ("B", acc_b, pred_b), ("C", acc_c, pred_c)],
               key=lambda t: t[1])
    print(f"\nFull per-class report, best configuration ({best[0]}):")
    print(classification_report(y_merged[va], best[2], labels=out_classes,
                                target_names=out_classes, zero_division=0))


if __name__ == "__main__":
    main()
