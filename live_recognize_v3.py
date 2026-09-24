"""
SignBridge - live webcam sign recognition, v3 (body-relative features).

Same job as live_recognize.py, served by the v3 model: the one that reads
where a sign is made on the body, not just what the hand is doing
(val 0.895 vs 0.798).

Kept as a SEPARATE file rather than an edit to live_recognize.py, so the
known-good v2 demo path stays intact as a fallback until this has been
tested cold. Once you trust it, delete the old one - two live scripts is
exactly the kind of duplication that drifts.

WHAT'S DIFFERENT FROM THE v2 LIVE SCRIPT
----------------------------------------
1. A second detector. PoseLandmarker runs alongside HandLandmarker to
   supply the shoulders, nose and mouth that the body frame is built on.

2. The buffers are FRAME-ALIGNED and RAW. v2 buffered only the frames a
   hand was seen in, already normalized, and cleared a hand's buffer when
   it left the frame. v3 appends one entry per camera frame for each of
   left hand / right hand / pose, writing NaN when that detector found
   nothing. That is what lets the features ask "where was the hand
   relative to the shoulders at this moment" - there has to be a shared
   clock - and it also makes the buffer-clearing hack unnecessary: a hand
   that left the frame ages out of the rolling window on its own.

3. Features come from build_clip_features_v3(), the same function the
   training CSVs were built with. Train/serve drift is impossible by
   construction, which matters much more now that there are two feature
   blocks to keep in sync instead of one.

4. Pose runs every POSE_EVERY frames, not every frame. The torso moves
   slowly, and the feature builder already interpolates across pose
   dropouts (it has to - real clips drop frames). Running it on every
   second frame roughly halves the added cost for no measurable loss.
   Set POSE_EVERY = 1 if you want it on every frame.

5. A body-visibility indicator. The model now leans heavily on body
   features, so if your shoulders are out of frame the predictions get
   much worse - and without an indicator that looks like the model being
   broken rather than the camera being framed badly. Stand back far
   enough that the readout says BODY OK.

Run:
    python live_recognize_v3.py              # RandomForest only (known-good path)
    python live_recognize_v3.py --ensemble   # frozen RF + 5-GRU ensemble
                                             # (models/signbridge_ensemble_v3.joblib)

With --ensemble, a sign below the dev-chosen confidence threshold is shown
as "WORD?" in orange: the kiosk should ask the user to confirm it rather than
act on it. The ensemble runs on numpy + scikit-learn only - no PyTorch.

Controls:
    q     -> quit
    SPACE -> freeze/unfreeze the current prediction
"""

import argparse
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np

from mp_hand_detector import (
    create_detector, to_mp_image, landmarks_to_array,
    normalize_landmark_sequence, HAND_CONNECTIONS, RunningMode,
)
from mp_body_detector import (
    create_pose_detector, pose_to_array, N_POSE,
    B_NOSE, B_MOUTH_L, B_MOUTH_R, B_SHOULDER_L, B_SHOULDER_R, B_ELBOW_L, B_ELBOW_R,
)
from build_feature_vectors_v2 import make_feature_names as make_hand_local_names
from build_feature_vectors_v3 import (
    build_clip_features_v3, make_body_names, compact,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_v3.joblib")

WINDOW_FRAMES = 45        # ~1.5s at 30fps - roughly one sign's duration
PREDICT_EVERY = 5         # re-predict every N frames (not every frame - too jittery)
POSE_EVERY = 2            # run PoseLandmarker every N frames (see docstring)
MIN_FRAMES_TO_PREDICT = 12
MOTION_GATE = 0.35        # min wrist path length in the window, in hand-local units

# Body skeleton drawn for feedback: shoulders, upper arms, and a neck-to-nose line.
BODY_CONNECTIONS = [
    (B_SHOULDER_L, B_SHOULDER_R),
    (B_SHOULDER_L, B_ELBOW_L),
    (B_SHOULDER_R, B_ELBOW_R),
]


def draw_skeleton(frame, hand_landmarks):
    """
    Draw onto the MIRRORED display frame, while the landmarks themselves came
    from the UNFLIPPED frame (see main() - detection must match training).
    x is mirrored here for drawing only; the feature pipeline never sees
    these mirrored coordinates.
    """
    h, w, _ = frame.shape
    pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def draw_body(frame, pose):
    """
    Draw the reference points the body frame is built from, so it is obvious
    at a glance whether the thing the model depends on is actually being
    tracked. pose: (N_POSE, 3) raw, or None.
    """
    if pose is None or not np.isfinite(pose[B_SHOULDER_L, 0]):
        return
    h, w, _ = frame.shape
    pts = {}
    for i in range(N_POSE):
        if np.isfinite(pose[i, 0]):
            pts[i] = (int((1.0 - pose[i, 0]) * w), int(pose[i, 1] * h))
    for a, b in BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 180, 0), 2)
    # the origin of the body coordinate frame
    if B_SHOULDER_L in pts and B_SHOULDER_R in pts:
        neck = ((pts[B_SHOULDER_L][0] + pts[B_SHOULDER_R][0]) // 2,
                (pts[B_SHOULDER_L][1] + pts[B_SHOULDER_R][1]) // 2)
        cv2.circle(frame, neck, 5, (255, 180, 0), -1)
        if B_NOSE in pts:
            cv2.line(frame, neck, pts[B_NOSE], (255, 180, 0), 1)
    for i in (B_NOSE, B_MOUTH_L, B_MOUTH_R):
        if i in pts:
            cv2.circle(frame, pts[i], 3, (255, 180, 0), -1)


def window_path_length(seq):
    """Total distance the wrist travelled across the buffered window."""
    if seq is None or seq.shape[0] < 2:
        return 0.0
    wrist = seq[:, 0, :]
    return float(np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum())


def local_normalized(raw_window):
    """
    The hand-local view of a buffered window, used ONLY for the motion gate.

    The gate threshold was tuned in hand-local units against the v2 script,
    so it is computed the same way here rather than being re-derived in
    shoulder-widths and needing a fresh threshold.
    """
    seq = compact(raw_window)
    if seq is None or seq.shape[0] < 2:
        return None
    return normalize_landmark_sequence(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", action="store_true",
                    help="use the frozen RF + 5-GRU ensemble instead of the RF alone")
    args = ap.parse_args()

    ens = clf = None
    if args.ensemble:
        from signbridge_ensemble import SignBridgeEnsemble
        try:
            ens = SignBridgeEnsemble.load()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return
        print(ens.describe())
    else:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: {MODEL_PATH} not found.")
            print("Run:  python build_feature_vectors_v3.py  &&  "
                  "python train_classifier.py data\\features_v3.csv")
            return

        bundle = joblib.load(MODEL_PATH)
        clf, feature_columns = bundle["model"], bundle["feature_columns"]

        expected = make_hand_local_names() + make_body_names()
        if list(feature_columns) != expected:
            print("WARNING: the model's feature columns don't match what this script builds.")
            print(f"  model: {len(feature_columns)} columns, this script: {len(expected)}")
            print("  Rebuild and retrain after any change to the feature settings.")
            return

    hand_detector = create_detector(running_mode=RunningMode.VIDEO, num_hands=2)
    pose_detector = create_pose_detector(running_mode=RunningMode.VIDEO)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (is another app using it?)")
        hand_detector.close()
        pose_detector.close()
        return

    nan_hand = np.full((21, 3), np.nan, dtype=np.float32)
    nan_pose = np.full((N_POSE, 3), np.nan, dtype=np.float32)

    buf_left = deque(maxlen=WINDOW_FRAMES)
    buf_right = deque(maxlen=WINDOW_FRAMES)
    buf_pose = deque(maxlen=WINDOW_FRAMES)

    frame_i = 0
    last_ts = -1
    display = {"text": "warming up...", "conf": 0.0, "top3": [], "idle": True}
    frozen = False
    last_pose = None

    print("Running. Sign at the camera. 'q' quits, SPACE freezes the readout.")
    print("Stand back far enough that the readout says BODY OK - the v3 model")
    print("depends on your shoulders being visible.")
    t0 = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame_h, frame_w = frame.shape[:2]

        # CRITICAL: detect on the UNFLIPPED frame.
        #
        # The training videos were fed to MediaPipe without flipping.
        # MediaPipe assigns handedness assuming a mirrored input, and
        # flipping also mirrors the landmark x-coordinates - so flipping here
        # would send your physical right hand into the model's LEFT feature
        # block with mirrored geometry, while every training example put right
        # hands in the RIGHT block unmirrored. Under v3 this would ALSO mirror
        # the hand against an un-mirrored torso, which is worse still: the
        # body block exists precisely to encode that relationship. Detect
        # unflipped; mirror only what is displayed.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Both detectors are in VIDEO mode, which requires strictly increasing
        # timestamps for the life of the detector. Two frames can land inside
        # the same millisecond, so the clock is clamped upward rather than
        # trusted.
        ts = max(last_ts + 1, int((time.time() - t0) * 1000))
        last_ts = ts

        mp_img = to_mp_image(rgb)
        hand_result = hand_detector.detect_for_video(mp_img, ts)

        left = nan_hand
        right = nan_hand
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks,
                                                  hand_result.handedness):
                arr = landmarks_to_array(hand_landmarks)
                if handedness[0].category_name == "Left":
                    left = arr
                else:
                    right = arr

        pose = nan_pose
        if frame_i % POSE_EVERY == 0:
            pose_result = pose_detector.detect_for_video(mp_img, ts)
            if pose_result.pose_landmarks:
                pose = pose_to_array(pose_result.pose_landmarks[0])
                last_pose = pose

        # One append per camera frame, for all three, always. This lockstep is
        # what gives the three streams a shared clock.
        buf_left.append(left)
        buf_right.append(right)
        buf_pose.append(pose)

        frame = cv2.flip(frame, 1)  # display only, AFTER detection

        draw_body(frame, last_pose)
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                draw_skeleton(frame, hand_landmarks)

        frame_i += 1

        left_win = np.stack(buf_left)
        right_win = np.stack(buf_right)
        pose_win = np.stack(buf_pose)

        n_left = int(np.isfinite(left_win[:, 0, 0]).sum())
        n_right = int(np.isfinite(right_win[:, 0, 0]).sum())
        body_frac = float(np.isfinite(pose_win[:, B_SHOULDER_L, 0]).mean())
        # POSE_EVERY frames of skipping means the best achievable coverage is
        # 1/POSE_EVERY; judge against that, not against 1.0.
        body_ok = body_frac >= 0.5 / POSE_EVERY

        if frame_i % PREDICT_EVERY == 0 and not frozen:
            has_left = n_left >= MIN_FRAMES_TO_PREDICT
            has_right = n_right >= MIN_FRAMES_TO_PREDICT

            motion = max(window_path_length(local_normalized(left_win)) if has_left else 0.0,
                         window_path_length(local_normalized(right_win)) if has_right else 0.0)

            if not (has_left or has_right):
                display = {"text": "no hand", "conf": 0.0, "top3": [], "idle": True}
            elif motion < MOTION_GATE:
                display = {"text": "IDLE (hand still)", "conf": 0.0, "top3": [], "idle": True}
            elif ens is not None:
                out = ens.predict(left_win, right_win, pose_win, frame_w, frame_h,
                                  min_hand_frames=MIN_FRAMES_TO_PREDICT)
                display = {"text": out["label"] if out["accepted"] else out["label"] + "?",
                           "conf": out["confidence"], "top3": out["top"],
                           "idle": False, "accepted": out["accepted"]}
            else:
                feat = build_clip_features_v3(
                    left_win if has_left else None,
                    right_win if has_right else None,
                    pose_win, frame_w, frame_h,
                ).reshape(1, -1)
                proba = clf.predict_proba(feat)[0]
                order = np.argsort(proba)[::-1][:3]
                top3 = [(clf.classes_[i], float(proba[i])) for i in order]
                display = {"text": top3[0][0], "conf": top3[0][1], "top3": top3, "idle": False}

        # ---- overlay ----
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
        if display["idle"]:
            colour = (140, 140, 140)
        elif display.get("accepted", True):
            colour = (0, 230, 0)
        else:
            colour = (0, 165, 255)       # below threshold: ask the user to confirm
        cv2.putText(frame, display["text"], (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)
        if not display["idle"]:
            cv2.putText(frame, f"confidence {display['conf']:.2f}", (14, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            xs = 300
            for name, p in display["top3"][1:]:
                cv2.putText(frame, f"{name} {p:.2f}", (xs, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                xs += 150

        body_text = "BODY OK" if body_ok else "BODY NOT VISIBLE - step back"
        body_colour = (0, 230, 0) if body_ok else (0, 165, 255)
        cv2.putText(frame, body_text, (w - 330, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, body_colour, 2)
        if frozen:
            cv2.putText(frame, "FROZEN", (w - 130, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2)

        cv2.imshow("SignBridge - live recognition v3 (q quit, SPACE freeze)", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            frozen = not frozen

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    pose_detector.close()


if __name__ == "__main__":
    main()
