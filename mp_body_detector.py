"""
SignBridge - shared MediaPipe BODY (pose) landmark utilities.

WHY THIS EXISTS
---------------
The v1/v2 feature vector is hands-only: 2 hands x 21 landmarks, with each
hand normalized against its OWN wrist (see
mp_hand_detector.normalize_landmark_sequence). That representation is
deliberately invariant to where the hand is in the camera frame - which
is exactly right for handshape, and exactly wrong for sign LOCATION.

In ASL, location is a phoneme. HUNGRY and THANK-YOU are different signs
that both start near the chin and differ in where the hand goes from
there. With no chin, no shoulders and no torso anywhere in the feature
vector, the classifier is being asked to separate them using information
that was discarded before it ever saw the data. (The measured
HUNGRY<->THANKYOU confusion - 29% of their combined val clips - is the
symptom.)

This module adds the missing body reference: MediaPipe's PoseLandmarker,
subset to the 11 points that matter for signing, so hand positions can be
expressed in a BODY-CENTRED coordinate frame instead of an
image-centred one.

WHY THE TASKS API AND NOT HOLISTIC
----------------------------------
`mp.solutions.holistic` (the one-call hands+pose+face model) has been
removed from current mediapipe releases along with the rest of the legacy
solutions API - the same gotcha already documented for
`mp.solutions.hands`. So the body landmarks come from a second Tasks-API
detector (PoseLandmarker) run alongside the existing HandLandmarker, with
its own ~5MB model file downloaded on first use.

Running two detectors per frame roughly doubles extraction time. The lite
pose model is used rather than full/heavy for that reason - shoulder,
elbow, nose and mouth positions do not need sub-pixel precision to serve
as a coordinate reference.

THE COORDINATE FRAME
--------------------
    origin = midpoint of the two shoulders, PER FRAME
    scale  = shoulder width, MEDIAN over the clip
    x      = aspect-corrected before either is applied

Per-frame origin, per-clip scale is a deliberate split:

  - The origin must track the signer per frame, because people sway,
    lean and drift during a sign. Note this does NOT delete hand motion
    the way per-frame WRIST centring would (the warning in
    mp_hand_detector) - the hand moves relative to the torso, so
    subtracting the torso keeps every bit of that motion. It only
    removes the signer's own bulk movement, which is noise here.

  - The scale must NOT be per-frame. Apparent shoulder width shrinks
    whenever the signer rotates toward one side, so a per-frame divisor
    would inject a spurious zoom into the trajectory every time they
    turn. Distance to the camera barely changes inside a 2-3 second
    clip, so one robust median per clip is both steadier and sufficient.

Aspect correction: MediaPipe normalizes x by image WIDTH and y by image
HEIGHT, so on any non-square video one unit of x is a different physical
distance from one unit of y. Comparing a hand's horizontal offset to its
vertical offset - or dividing both by a shoulder width - is only
meaningful after putting them in the same units. x is multiplied by W/H
to express both axes in units of image height.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# MediaPipe Pose emits 33 landmarks. Most of them (hips, knees, ankles,
# feet, individual eye corners) are either out of frame in signing footage
# or irrelevant to a seated/standing upper-body sign. These 11 are kept:
#
#   nose + ears  -> head position and head size (a scale-free proxy for
#                   "is this sign at face level")
#   mouth        -> the chin/mouth reference that HUNGRY vs THANK-YOU needs
#   shoulders    -> the coordinate frame itself (origin + scale)
#   elbows       -> arm configuration; distinguishes signs made with the
#                   arm raised vs tucked even when the hand path is similar
#   pose wrists  -> a coarse fallback position for a hand that the
#                   HandLandmarker dropped on that frame
#
# MediaPipe's "left" is the SUBJECT's left, matching HandLandmarker's
# handedness convention, so left-hand features pair with left-shoulder
# features without a flip.
POSE_KEEP = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
POSE_NAMES = (
    "nose", "ear_l", "ear_r", "mouth_l", "mouth_r",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
)
N_POSE = len(POSE_KEEP)

# Indices INTO the kept subset (not into MediaPipe's 33).
B_NOSE, B_EAR_L, B_EAR_R = 0, 1, 2
B_MOUTH_L, B_MOUTH_R = 3, 4
B_SHOULDER_L, B_SHOULDER_R = 5, 6
B_ELBOW_L, B_ELBOW_R = 7, 8
B_WRIST_L, B_WRIST_R = 9, 10

# Left/right pairs within the kept subset, for mirror augmentation.
POSE_MIRROR_PAIRS = ((B_EAR_L, B_EAR_R), (B_MOUTH_L, B_MOUTH_R),
                     (B_SHOULDER_L, B_SHOULDER_R), (B_ELBOW_L, B_ELBOW_R),
                     (B_WRIST_L, B_WRIST_R))

MIN_SHOULDER_WIDTH = 1e-3  # guard against a degenerate/edge-on detection


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"First run - downloading pose landmark model to:\n  {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded.\n")
        except Exception as e:
            print(f"ERROR: could not download the pose model automatically ({e}).")
            print(f"Manually download this file and save it as pose_landmarker.task "
                  f"next to this script:\n  {MODEL_URL}")
            raise SystemExit(1)


def create_pose_detector(running_mode, result_callback=None):
    """
    Same running-mode rules as the hand detector: IMAGE for batches of
    independent clips, VIDEO for one continuous webcam stream.
    """
    ensure_model()
    kwargs = dict(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    if running_mode == RunningMode.LIVE_STREAM:
        kwargs["result_callback"] = result_callback
    return PoseLandmarker.create_from_options(PoseLandmarkerOptions(**kwargs))


def pose_to_array(pose_landmarks):
    """One pose's 33 NormalizedLandmarks -> (N_POSE, 3) for the kept subset."""
    full = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks], dtype=np.float32)
    return full[list(POSE_KEEP)]


# ---------------------------------------------------------------------------
# Body coordinate frame
# ---------------------------------------------------------------------------

def aspect_correct(xy, frame_w, frame_h):
    """
    Put x and y in the same physical units (units of image height).

    xy: (..., 2) array of MediaPipe-normalized coordinates.
    """
    out = np.array(xy, dtype=np.float32, copy=True)
    if frame_h and frame_w:
        out[..., 0] *= float(frame_w) / float(frame_h)
    return out


def body_reference(pose_seq_xy):
    """
    pose_seq_xy: (T, N_POSE, 2) aspect-corrected pose landmarks, NaN where
                 the pose was not detected on that frame.

    Returns (origin, scale, valid):
        origin (T, 2) - per-frame shoulder midpoint, gaps linearly
                        interpolated and end-padded so a brief pose dropout
                        does not punch a hole in the trajectory
        scale  float  - one median shoulder width for the whole clip
        valid  (T,)   - bool, True where the pose was genuinely detected
    """
    ls = pose_seq_xy[:, B_SHOULDER_L, :]
    rs = pose_seq_xy[:, B_SHOULDER_R, :]
    valid = np.isfinite(ls).all(axis=1) & np.isfinite(rs).all(axis=1)

    origin = (ls + rs) / 2.0
    widths = np.linalg.norm(ls - rs, axis=1)

    if valid.sum() == 0:
        return origin, 0.0, valid

    scale = float(np.median(widths[valid]))
    if not np.isfinite(scale) or scale < MIN_SHOULDER_WIDTH:
        scale = MIN_SHOULDER_WIDTH

    origin = _interp_gaps(origin, valid)
    return origin, scale, valid


def _interp_gaps(arr, valid):
    """
    Fill NaN rows of an (T, C) array by linear interpolation over the frame
    axis, holding the first/last valid value at the ends. Used only for the
    body ORIGIN - short pose dropouts are a detector artifact, not a real
    jump of the signer's torso, so interpolating across them is closer to
    the truth than leaving a hole.
    """
    out = np.array(arr, dtype=np.float32, copy=True)
    t = np.arange(out.shape[0])
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    for c in range(out.shape[1]):
        out[:, c] = np.interp(t, idx, out[idx, c])
    return out


def to_body_frame(points_xy, origin, scale):
    """
    points_xy: (T, L, 2) aspect-corrected landmarks
    origin:    (T, 2) per-frame shoulder midpoint
    scale:     float, median shoulder width

    Returns (T, L, 2) in shoulder-widths from the shoulder midpoint:
    (0, 0) is the base of the neck, (1, 0) is one shoulder-width to the
    subject's right, y grows downward (image convention).
    """
    return (points_xy - origin[:, None, :]) / scale
