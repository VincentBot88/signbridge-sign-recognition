# SignBridge — Sign Recognition Pipeline

ASL sign recognition for **SignBridge**, an AI communication kiosk for sign, speech and text.
Capstone project HT01 (COE/ELE70A), Toronto Metropolitan University.

This repository is the **sign dataset and recognition** half of the project: hand- and
body-landmark extraction, feature representation, the sign classifier, confidence scoring,
and cross-signer evaluation. Speech, text and the kiosk UI are not in this repo.

Development is on a Windows laptop with a built-in webcam, ahead of a move to a Raspberry Pi 5
for the dedicated kiosk hardware.

---

## Status

**The recognizer is frozen and has had its one final test evaluation.** The deployed model is
an ensemble of a RandomForest on v3 features and five body-only GRUs, packaged as a single file
that the kiosk loads with numpy + scikit-learn only (no PyTorch at run time).

Held-out test split: 416 ASL Citizen clips from 11 signers who appear nowhere in training,
dev or val. It was evaluated once, after every choice was fixed
(`sweeps/final/test_result_v3.json`):

| Model | Test accuracy |
|---|---:|
| **RF + 5-GRU ensemble** | **0.927** (381 / 411, 95% CI 0.898–0.948) |
| Same, counting the 5 unusable clips as wrong | 0.916 |
| 5-GRU average alone | 0.883 |
| RandomForest alone | 0.808 |

- **Abstention:** at the dev-chosen confidence threshold (0.394) the kiosk accepts 91% of signs,
  and 96.3% of the accepted signs are correct. The other 9% trigger a "did you mean…?" prompt.
- **Per signer:** accuracy ranges from 0.85 to 0.97 across the 11 test signers.
- **Weakest classes:** HUNGRY (0.67), WAIT (0.77), WHO (0.79) and YES (0.80). The most common
  confusion is WAIT → WHAT (3 clips).

The sections below cover how the project got here. They are kept because each step changed the
number.

---

## Result: feature representation (RandomForest)

Three generations of the feature representation, same classifier, same signer-disjoint split:

| Gen | Representation | Cols | Val accuracy |
|-----|----------------|-----:|-------------:|
| v1  | Per-clip summary statistics — start, end, displacement, mean velocity, peak speed | ~548 | 0.403 |
| v2  | 10 resampled keyframes + per-landmark path length + peak speed | 926 | 0.726 → 0.798 |
| v3  | v2 hand-local block **+ body-relative block** from pose landmarks | 1,221 | **0.895** |

**v1 → v2.** v1 compressed each clip into averages, and for oscillating signs the back-and-forth
cancels: for STOP the hand travelled 7.87 units of path but netted 0.235 of displacement, so ~97%
of the motion was invisible to the classifier. That is why accuracy refused to move across feature
count, tree depth, regularization and 4× augmentation — the bottleneck was the representation, not
the model. Keyframes plus path length fixed it, and signs that had been stuck at 0.00 recall
(WHERE, WATER, WHO) went to 1.00.

**v2 → v3.** Every v2 feature is measured in a hand-local frame — the hand's own wrist is the
origin, the hand's own size is the unit. That is right for handshape, but in ASL *location is a
phoneme*. With no chin, shoulders or torso in the vector, a sign made at the mouth and the same
motion made at the chest are not merely hard to separate, they are literally the same input:
synthetic clips differing only in height produce v2 vectors that differ by 2.9 × 10⁻⁶. Under v3
the same pair differs by 0.70.

Ablation on the final 625-clip / 1,250-row training set:

| Feature set | Cols | Val accuracy | Correct / 124 |
|---|---:|---:|---:|
| Hand-local only (exact v2 reproduction) | 926 | 0.798 | 99 |
| **Hand-local + body** | **1,221** | **0.895** | **111** |
| Body only | 295 | 0.750 | 93 |

The control reproduces v2 exactly, which confirms the re-extraction did not move the ground under
the comparison. Biggest per-class gains: DEAF 0.43 → 1.00, WHAT 0.29 → 0.86, SICK 0.25 → 0.75.
HUNGRY regressed (1.00 → 0.50) — v3 did **not** fix the HUNGRY/THANKYOU confusion.

### Reading these numbers honestly

- v2's 0.726 is from the first 28-class run; 0.798 is the same representation re-measured on the
  final training set, and it is 0.798 — not 0.726 — that is the like-for-like baseline for v3.
- Every figure is a **single measurement on a 124-clip val split**. One standard error is about
  six clips. The v3 gain is twelve clips, roughly two standard errors — the first change in this
  project to clear the noise band by a comfortable margin, but not proof.
- scikit-learn version changes the number (0.726 vs 0.766 on identical inputs and seed), hence the
  pins in `requirements.txt`. Quote figures from the project machine only.

---

## Result: sequence model and ensemble

### GRU on v3 landmarks (`train_pytorch_v3.py`)

On v2 data the GRU trailed the RandomForest by about 10 points (0.69 vs 0.78). The v3 GRU reads
per-frame landmarks on one shared clock for both hands and pose. Short detector dropouts are
interpolated instead of being dropped. The GRU trains with a fixed evaluation protocol:

- A **signer-disjoint dev slice** (250 rows, 8 signers) is carved out of train. It drives early
  stopping, and val is only reported. Selecting on val had inflated earlier GRU numbers by about
  5 points.
- **5 seeds per configuration**, reported as mean ± spread.
- **Two independent augmentation streams.** A result has to hold on both streams to count.
  Several earlier claims were stable across seeds but disappeared on a second stream.

On-the-fly augmentation (`augment_v3.py`) was the largest single lever. It applies non-uniform
time warp, temporal crop, small rotation, landmark jitter and frame dropout. Translation, uniform
scale and uniform speed changes are left out because the feature pipeline already cancels them.
Results for the body-only GRU, val@best-dev, 5 seeds × 2 streams, patience 200:

| Aug strength | Stream 0 | Stream 1 |
|---:|---:|---:|
| none | 0.706 | 0.706 |
| 1 | 0.787 | 0.790 |
| 2 | 0.839 | 0.840 |
| **3** | **0.890** | **0.885** |
| 4 | 0.863 | 0.881 |

Strength 3 was chosen, which puts the GRU level with the RF on val (≈110 / 124 each).

### Ensemble (`ensemble_v3.py`)

The v2 ensemble lost to the RF alone because the GRU was weaker and overconfident. In v3 both
models are equally accurate and read different inputs. The RF uses engineered hand-local and body
features; the GRU uses the raw body-frame trajectory. Both train on the same 1,000-row fit set.
Each model's probabilities are temperature-calibrated on dev, then averaged with equal weight.
The decision rule was fixed before any result was seen.

| Val (124 clips, mean of 5 pairs) | Stream 0 | Stream 1 |
|---|---:|---:|
| RandomForest | 110.2 | 110.4 |
| GRU | 110.4 | 109.8 |
| **Calibrated average** | **115.0** | **114.0** |
| Raw (uncalibrated) average | 112.6 | 111.6 |

Verdict: **supported**. The ensemble beats both single models on both streams (paired t-test),
and dev agrees. Calibration is worth about 2.5 clips on its own.

### Freezing and deployment

`build_ensemble_bundle.py` freezes the RF (seed 42), the five stream-0 GRU seeds, the
temperatures and the abstention threshold into `models/signbridge_ensemble_v3.joblib`. It
refuses to save unless all of these checks pass:

- The rows match the ensemble experiment's rows.
- The kiosk's feature path is identical to the training path.
- The numpy GRU forward pass matches PyTorch on every dev and val clip.
- Each seed reproduces its recorded score.

`signbridge_ensemble.py` loads that bundle and exposes a single `predict()`. The kiosk therefore
needs no PyTorch, which matters on the Raspberry Pi 5. It also keeps Windows Smart App Control,
which has blocked torch's DLLs on the demo laptop before, out of the demo path.

`evaluate_ensemble_on_test.py` ran the one test evaluation shown under **Status**. It records the
result and the bundle's SHA-256, and it refuses to run a second time without `--force`.

---

## Vocabulary

27 classes, above the spec's 20-sign minimum. All real ASL drawn from ASL Citizen, not invented
gestures:

```
HELLO  BYE  THANKYOU  PLEASE  SORRY  YES  NO  HELP  STOP  WAIT  MORE
WATER  BATHROOM  EAT  DRINK  HUNGRY  HURT_PAIN  SICK  NAME
WHERE  WHAT  WHO  HOW  WHY  UNDERSTAND  DEAF  INTERPRETER
```

28 words were selected; HURT and PAIN are merged into one `HURT_PAIN` class. Trained as a
standalone two-way problem, HURT-vs-PAIN scored 0.125 — below a coin flip — because both are
commonly signed with index fingers jabbing toward each other, location indicating where it hurts.
Keeping them separate asked the classifier for a distinction the video does not contain. Merging
measured +3.2 points. It is applied at feature-build time via `LABEL_MERGES` in
`build_feature_vectors_v2.py`, so no re-extraction is needed.

ASL Citizen's splits are disjoint by signer (35 train / 6 val / 11 test people, zero overlap), so
the provided split is already a genuine cross-user evaluation and satisfies the spec's
"evaluate recognition across different users" requirement without a custom split.

---

## Layout

The three generations share modules and live side by side rather than in separate folders — v3
imports directly from v2, and `train_classifier.py` / `evaluate_on_test.py` are shared by all
three. The `v3` git tag marks the RandomForest-only v3 state (val 0.895), before the sequence
model and ensemble work.

**Shared**

| File | Role |
|---|---|
| `mp_hand_detector.py` | HandLandmarker setup, 21-point topology, per-clip landmark normalization |
| `mp_body_detector.py` | PoseLandmarker setup, body-centred frame construction (v3) |
| `train_classifier.py` | RandomForest trainer; takes a feature CSV, derives the model path from it |
| `evaluate_on_test.py` | Held-out test evaluation for any feature set |

**v1 — summary-statistic features**

| File | Role |
|---|---|
| `hand_landmark_smoke_test.py` | Webcam + live hand skeleton; the base camera pipeline |
| `select_vocabulary_clips.py` | Pulls the vocabulary's clips out of the ASL Citizen zip without a full 42.8 GB unzip |
| `extract_clip_landmarks.py` | HandLandmarker over each clip → one `.npz` per clip |
| `build_feature_vectors.py` | Summary-statistic features → `data/features.csv` |
| `augment_landmarks.py` | Jitter + time-stretch augmentation (measured: no gain) |
| `mirror_augment.py` | Left/right mirroring |

**v2 — keyframe features**

| File | Role |
|---|---|
| `build_feature_vectors_v2.py` | Keyframe + path-length features → `data/features_v2.csv`; holds `LABEL_MERGES` |
| `live_recognize.py` | Webcam inference against the v2 model |
| `download_wlasl_clips.py` | Fetches WLASL clips via yt-dlp, cached by video ID |
| `extract_wlasl_landmarks.py` | Landmark extraction for the WLASL clips |
| `qa_clips.py` | Per-clip QA report → `data/qa_report.csv` |
| `variant_split_test.py` | Sign-variant merge experiment |
| `train_pytorch.py` | GRU / MLP control models (optional, needs torch) |
| `ensemble_test.py` | RandomForest + GRU ensemble probe (optional, needs torch) |

**v3 — body-relative features**

| File | Role |
|---|---|
| `extract_landmarks_v3.py` | Hands **and** pose → `data/landmarks_v3/` |
| `build_feature_vectors_v3.py` | Body-relative features; `--no-body` / `--no-hand-local` drive the ablation |
| `mirror_augment_v3.py` | Mirroring, handedness-aware |
| `test_body_features_v3.py` | Synthetic unit tests — no data or model download needed |
| `live_recognize_v3.py` | Webcam inference: v3 RF by default, the frozen ensemble with `--ensemble` |

**v3 — sequence model and ensemble**

| File | Role |
|---|---|
| `train_pytorch_v3.py` | GRU / MLP on per-frame v3 landmarks; dev-based early stopping, multi-seed, `--augment`, `--save-probs` / `--save-seeds` |
| `augment_v3.py` | On-the-fly augmentation (time warp, crop, rotation, jitter, dropout); runs a self-test when executed directly |
| `sweep_aug_v3.py` | Resumable augmentation-strength × stream sweep, with a collated significance table |
| `verify_speedup_v3.py` | Checks that the parallel augmentation rebuild is bit-identical to the serial one |
| `ensemble_v3.py` | RF + GRU ensemble experiment with a pre-registered decision rule |
| `build_ensemble_bundle.py` | Freezes the ensemble into `models/signbridge_ensemble_v3.joblib` after parity checks |
| `signbridge_ensemble.py` | The runtime recognizer: loads the bundle, numpy-only GRU, `predict()` → label, confidence, accept/confirm |
| `evaluate_ensemble_on_test.py` | The single final test evaluation → `sweeps/final/test_result_v3.json` |
| `sweeps/` | Result JSONs from the augmentation sweep, the ensemble experiment and the final test |

---

## Setup

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.12** specifically — mediapipe supports 3.9–3.12 and will not install on 3.13+.

The two MediaPipe detector models (`hand_landmarker.task`, `pose_landmarker.task`) download
themselves on first run, so there is nothing else to fetch.

PyTorch is only needed to *train* the GRUs (`train_pytorch_v3.py`, `build_ensemble_bundle.py`).
Uncomment the `torch` line in `requirements.txt`, or run `pip install torch`. Running the frozen
ensemble does not need it.

One API gotcha worth knowing: current mediapipe releases have removed the legacy
`mp.solutions.hands` API entirely — `mp.solutions` raises `AttributeError` even on older 0.10.x
builds. All landmark code here uses the modern Tasks API. Batch clip processing uses
`RunningMode.IMAGE`; the webcam loop uses `RunningMode.VIDEO`, whose timestamps must strictly
increase for the life of the detector, which is why a VIDEO-mode detector is never reused across
separate clips.

---

## Running the pipeline

Sanity check, no data needed:

```powershell
python test_body_features_v3.py
```

Full v3 rebuild from clips:

```powershell
python extract_landmarks_v3.py --splits train,val
python mirror_augment_v3.py
python build_feature_vectors_v3.py --no-body        # control, 926 cols
python build_feature_vectors_v3.py                  # full,    1,221 cols
python build_feature_vectors_v3.py --no-hand-local  # body,      295 cols

python train_classifier.py data\features_v3_handlocal.csv
python train_classifier.py data\features_v3.csv
python train_classifier.py data\features_v3_bodyonly.csv
```

Train the GRUs and freeze the ensemble (about 40 min of GRU training, then about 1 min):

```powershell
python train_pytorch_v3.py --no-hand-local --augment --aug-strength 3 `
    --aug-stream 0 --seeds 5 --patience 200 --epochs 2000 --torch-threads 1 `
    --no-save --save-seeds models/gru_v3_s3_seeds.pt
python build_ensemble_bundle.py --gru-seeds models/gru_v3_s3_seeds.pt
```

The final test evaluation has already been run. It refuses to run again while
`sweeps/final/test_result_v3.json` exists:

```powershell
python extract_landmarks_v3.py --source citizen --splits test --out-dir data/landmarks_v3_test
python evaluate_ensemble_on_test.py
```

Live demo:

```powershell
python live_recognize_v3.py              # RandomForest only
python live_recognize_v3.py --ensemble   # frozen ensemble; low-confidence signs show as "WORD?"
```

---

## What is not in this repo

Excluded by `.gitignore`, with how to get each back:

| Excluded | Size | How to regenerate |
|---|---|---|
| `venv/` | — | `pip install -r requirements.txt` |
| `*.task` detector models | ~13 MB | Auto-downloaded on first run |
| `models/*.joblib` | ~197 MB | `train_classifier.py <feature csv>` |
| `models/signbridge_ensemble_v3.joblib`, `*.pt` GRU seeds | — | `train_pytorch_v3.py --save-seeds …` then `build_ensemble_bundle.py` |
| `sweeps/ensemble/*.npz` (saved GRU probabilities) | — | `train_pytorch_v3.py --save-probs …` |
| `data/features*.csv` | ~148 MB | `build_feature_vectors*.py` |
| `data/landmarks*/` | — | `extract_clip_landmarks.py` / `extract_landmarks_v3.py` |
| `data/clips/`, `data/wlasl_clips/` | — | `select_vocabulary_clips.py` from the ASL Citizen zip; `download_wlasl_clips.py` for WLASL |
| `csvs/` | ~4 MB | Ships with the ASL Citizen download |

The small manifests **are** committed — `data/clips_manifest.csv`, `data/qa_report.csv`,
`data/wlasl_clips_manifest.csv`, `data/wlasl_download_log.csv`, `wlasl_instances_28word.csv` and
the coverage CSVs — since they are what pin down which clips the results were measured on.

### Data licensing

Source video is **not redistributed here**. Clips come from
[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (primary — consistent
single-signer clips) and [WLASL](https://dxli94.github.io/WLASL/) (secondary), each under its own
licence and data agreement. Get them from the original sources. Only derived metadata — gloss
names, clip IDs, QA metrics — is committed.

No ASL-fluent volunteers were available for primary recording; the capstone spec permits
"consented or prerecorded data", so the plan is to bootstrap from public datasets and re-record a
subset under real webcam and lighting conditions.

---

## Known gaps

- The test figure comes from ASL Citizen clips: studio-like, one signer per clip, cleanly
  segmented. Live webcam use, with continuous signing, different lighting and no clip boundaries,
  has not been measured and will be lower. The re-recorded subset described above is what would
  measure it.
- HUNGRY is still the weakest class (0.67 on test). On val its errors went to THANKYOU, since
  both begin near the chin; on test they are scattered.
- WAIT → WHAT is the most common test confusion.
- Temperatures, GRU checkpoints and the abstention threshold were all fit on the same dev slice,
  so the threshold is slightly permissive. The test abstention result (96.3% accuracy on accepted
  signs, against a 95% target) suggests this does not matter in practice.
- `N_KEYFRAMES` (RF features) is still a hardcoded 10; 16 and 20 are untested.
- The full kiosk demo path, with `live_recognize_v3.py --ensemble` on the Raspberry Pi 5, still
  needs a cold end-to-end test before presentation day.
- EMERGENCY is missing from ASL Citizen and is a real gap for a communication kiosk; it would need
  to come from WLASL or a self-recorded clip.
