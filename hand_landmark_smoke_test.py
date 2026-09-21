"""
SignBridge - Step 0 smoke test (uses the shared mp_hand_detector module)

Opens the laptop webcam and draws live MediaPipe hand landmarks on top of
the video feed. Confirms camera + MediaPipe + drawing all work together.

Run:
    python hand_landmark_smoke_test.py

Controls:
    q  -> quit
"""

import cv2

from mp_hand_detector import create_detector, to_mp_image, HAND_CONNECTIONS, RunningMode


def draw_landmarks(frame, hand_landmarks, label, score):
    """hand_landmarks here are the raw (un-normalized) mediapipe landmarks,
    since we want to draw them at their actual position in the frame."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    wrist_x, wrist_y = points[0]
    cv2.putText(
        frame, f"{label} ({score:.2f})",
        (max(wrist_x - 20, 0), max(wrist_y + 30, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
    )


def main():
    # VIDEO mode: correct choice here since this is one continuous stream
    # (a single detector tracking hands frame-to-frame in real time).
    detector = create_detector(running_mode=RunningMode.VIDEO, num_hands=2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Is another app (Zoom, Teams, "
              "Camera app) using it? Close those and try again.")
        return

    print("Webcam window should open now. Hold up a hand. Press 'q' to quit.")

    import time
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Empty camera frame, skipping...")
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = to_mp_image(rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                score = handedness[0].score
                draw_landmarks(frame, hand_landmarks, label, score)

        cv2.imshow("SignBridge - Hand Landmark Smoke Test (press q to quit)", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
