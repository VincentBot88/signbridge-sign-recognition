"""
SignBridge - live webcam sign recognition.

This is the first real test of DOMAIN TRANSFER: the classifier has only
ever seen ASL Citizen clips (their cameras, framing, lighting, signers).
This script runs it against your laptop webcam, in your room, with you
signing. Whether that works is currently the biggest untested unknown in
the project - if accuracy collapses here, it reshapes every downstream
decision about thresholds and rejection classes.

HOW IT WORKS
------------
Keeps a rolling buffer of the last WINDOW_FRAMES frames of hand
landmarks. Every PREDICT_EVERY frames it takes that window, runs it
through the *exact same* feature construction used for training
(build_clip_features, imported from build_feature_vectors_v2 - not a
reimplementation, so train/serve features can't drift), and predicts.

Shows the top-3 predictions with confidences, so you can see not just
what it guesses but how sure it is - which is the raw material for the
confidence thresholding step the spec requires.

SEGMENTATION (deliberately crude, for now)
------------------------------------------
There is no trained "no sign" class yet, so the model will always name
one of the vocabulary words. As a stopgap this script gates on motion:
if the hand has barely moved across the window, it shows IDLE instead of
a prediction. This is a heuristic placeholder, not a solution - knowing
where a sign starts and ends in a continuous stream is the real unsolved
problem, and this makes that visible rather than hiding it.

Run:
    python live_recognize.py

Controls:
    q  -> quit
    SPACE -> freeze/unfreeze the current prediction (useful for reading it)
"""

import os
from collections import deque

import cv2
import joblib
import numpy as np

from mp_hand_detector import (
    create_detector, to_mp_image, landmarks_to_array,
    normalize_landmark_sequence, HAND_CONNECTIONS, RunningMode,
)
from build_feature_vectors_v2 import build_clip_features, make_feature_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_v2.joblib")

WINDOW_FRAMES = 45        # ~1.5s at 30fps - roughly one sign's duration
PREDICT_EVERY = 5         # re-predict every N frames (not every frame - too jittery)
MIN_FRAMES_TO_PREDICT = 12
MOTION_GATE = 0.35        # min total path length in the window to bother predicting


def draw_skeleton(frame, hand_landmarks, mirrored_display=True):
    """
    Draw onto the MIRRORED display frame, while the landmarks themselves came
    from the UNFLIPPED frame (see main() - detection must match training).
    So x is mirrored here for drawing only; the feature pipeline never sees
    these mirrored coordinates.
    """
    h, w, _ = frame.shape
    if mirrored_display:
        pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand_landmarks]
    else:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def window_path_length(seq):
    """Total distance the wrist travelled across the buffered window."""
    if seq is None or seq.shape[0] < 2:
        return 0.0
    wrist = seq[:, 0, :]
    return float(np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum())


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run:  python build_feature_vectors_v2.py  &&  python train_classifier.py data/features_v2.csv")
        return

    bundle = joblib.load(MODEL_PATH)
    clf, feature_columns = bundle["model"], bundle["feature_columns"]

    expected = make_feature_names()
    if list(feature_columns) != expected:
        print("WARNING: the model's feature columns don't match what this script builds.")
        print(f"  model: {len(feature_columns)} columns, this script: {len(expected)}")
        print("  Retrain after any change to build_feature_vectors_v2.py's feature settings.")
        return

    detector = create_detector(running_mode=RunningMode.VIDEO, num_hands=2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (is another app using it?)")
        return

    buffers = {"Left": deque(maxlen=WINDOW_FRAMES), "Right": deque(maxlen=WINDOW_FRAMES)}
    frame_i = 0
    display = {"text": "warming up...", "conf": 0.0, "top3": [], "idle": True}
    frozen = False

    print("Running. Sign at the camera. 'q' quits, SPACE freezes the readout.")
    import time
    t0 = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        # CRITICAL: detect on the UNFLIPPED frame.
        #
        # extract_clip_landmarks.py fed the training videos to MediaPipe
        # without flipping. MediaPipe assigns handedness assuming a mirrored
        # input, and flipping also mirrors the landmark x-coordinates - so
        # flipping here would send your physical right hand into the model's
        # LEFT feature block with mirrored geometry, while every training
        # example put right hands in the RIGHT block unmirrored. That is a
        # train/serve mismatch that would tank accuracy for reasons unrelated
        # to the camera. Detect unflipped; mirror only what's displayed.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_for_video(to_mp_image(rgb), int((time.time() - t0) * 1000))

        frame = cv2.flip(frame, 1)  # display only, AFTER detection

        seen = set()
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                side = handedness[0].category_name
                seen.add(side)
                buffers[side].append(landmarks_to_array(hand_landmarks))
                draw_skeleton(frame, hand_landmarks, mirrored_display=True)

        # a hand that left the frame shouldn't keep contributing stale frames
        for side in ("Left", "Right"):
            if side not in seen and buffers[side]:
                buffers[side].clear()

        frame_i += 1
        if frame_i % PREDICT_EVERY == 0 and not frozen:
            seqs = {}
            for side in ("Left", "Right"):
                if len(buffers[side]) >= MIN_FRAMES_TO_PREDICT:
                    seqs[side] = normalize_landmark_sequence(np.stack(buffers[side]))
                else:
                    seqs[side] = None

            motion = max(window_path_length(seqs["Left"]), window_path_length(seqs["Right"]))

            if seqs["Left"] is None and seqs["Right"] is None:
                display = {"text": "no hand", "conf": 0.0, "top3": [], "idle": True}
            elif motion < MOTION_GATE:
                display = {"text": "IDLE (hand still)", "conf": 0.0, "top3": [], "idle": True}
            else:
                feat = build_clip_features(seqs["Left"], seqs["Right"]).reshape(1, -1)
                proba = clf.predict_proba(feat)[0]
                order = np.argsort(proba)[::-1][:3]
                top3 = [(clf.classes_[i], float(proba[i])) for i in order]
                display = {"text": top3[0][0], "conf": top3[0][1], "top3": top3, "idle": False}

        # ---- overlay ----
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
        colour = (140, 140, 140) if display["idle"] else (0, 230, 0)
        cv2.putText(frame, display["text"], (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)
        if not display["idle"]:
            cv2.putText(frame, f"confidence {display['conf']:.2f}", (14, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            xs = 300
            for name, p in display["top3"][1:]:
                cv2.putText(frame, f"{name} {p:.2f}", (xs, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                xs += 150
        if frozen:
            cv2.putText(frame, "FROZEN", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("SignBridge - live recognition (q quit, SPACE freeze)", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            frozen = not frozen

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
