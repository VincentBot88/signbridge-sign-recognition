"""
SignBridge - shared MediaPipe hand-landmark detection utilities.

Used by both the live webcam smoke test (VIDEO mode - one continuous
stream) and the batch clip-processing script (IMAGE mode - many
independent short clips), so the model setup, connection topology, and
normalization logic only exist in one place.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Fixed 21-point MediaPipe hand skeleton connections (wrist=0 ... pinky tip=20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

# Landmark 0 = wrist, 9 = middle finger MCP - used as the scale reference
# (roughly constant regardless of hand size / distance from camera)
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"First run - downloading hand landmark model to:\n  {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded.\n")
        except Exception as e:
            print(f"ERROR: could not download the model automatically ({e}).")
            print(f"Manually download this file and save it as hand_landmarker.task "
                  f"next to this script:\n  {MODEL_URL}")
            raise SystemExit(1)


def create_detector(running_mode, num_hands=2, result_callback=None):
    """
    running_mode: RunningMode.VIDEO for a continuous stream (webcam),
                  RunningMode.IMAGE for independent frames/clips (batch
                  processing many short videos - do NOT use VIDEO mode
                  for that, its timestamps must strictly increase for the
                  life of the detector, which breaks across clip
                  boundaries).
    """
    ensure_model()
    kwargs = dict(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    if running_mode == RunningMode.LIVE_STREAM:
        kwargs["result_callback"] = result_callback
    options = HandLandmarkerOptions(**kwargs)
    return HandLandmarker.create_from_options(options)


def to_mp_image(rgb_frame):
    """rgb_frame: a numpy uint8 array in RGB order (H, W, 3)."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))


def landmarks_to_array(hand_landmarks):
    """Convert one hand's landmark list (21 NormalizedLandmark objects)
    into a (21, 3) numpy array of [x, y, z]."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)


def normalize_landmark_sequence(frames_xyz):
    """
    Make a (n_frames, 21, 3) landmark sequence invariant to WHERE in the
    camera frame the hand started and how big/far it is, WITHOUT erasing
    how the hand moves during the clip.

    Uses a single reference computed once for the whole clip (frame 0's
    wrist position for translation, the median wrist-to-middle-MCP
    distance across the clip for scale) and applies that same reference
    to every frame. This is deliberately NOT per-frame re-centering -
    per-frame normalization would reset the wrist to (0,0,0) on every
    single frame, which silently deletes any whole-hand travel through
    space (a lot of ASL signs are defined by exactly that motion, not
    just finger shape).

    Rotation (hand tilt) is still not normalized - kept simple for now,
    revisit if the classifier struggles with tilted-hand signs.

    frames_xyz: (n_frames, 21, 3) RAW (un-normalized) landmarks for one
                hand across a whole clip, frame 0 = first frame the hand
                was detected in.
    Returns: (n_frames, 21, 3) normalized landmarks, same frame count.
    """
    origin = frames_xyz[0, WRIST_IDX]                       # (3,) - fixed for the whole clip
    translated = frames_xyz - origin                        # broadcasts over all frames

    per_frame_scale = np.linalg.norm(translated[:, MIDDLE_MCP_IDX], axis=1)  # (n_frames,)
    scale_ref = np.median(per_frame_scale)
    if scale_ref < 1e-6:
        scale_ref = 1e-6  # avoid divide-by-zero on a degenerate detection

    return translated / scale_ref
