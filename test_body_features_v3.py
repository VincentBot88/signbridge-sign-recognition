"""
SignBridge - synthetic regression test for the v3 body-relative features.

Runs with no data, no videos and no MediaPipe model download - it builds
fake landmark sequences and checks the feature builder against them. The
key assertion is the one named "v2 block CANNOT tell the two apart": two
clips with identical handshape and identical relative hand motion, one
performed at mouth height and one at chest height, are bit-identical
under the v2 feature set and clearly separated under v3. That is the
HUNGRY/THANK-YOU failure mode reproduced in miniature.

Run:
    python test_body_features_v3.py
"""
import numpy as np

import build_feature_vectors_v2 as v2
import build_feature_vectors_v3 as v3
import mirror_augment_v3 as m3
from mp_body_detector import (N_POSE, B_NOSE, B_EAR_L, B_EAR_R, B_MOUTH_L, B_MOUTH_R,
                              B_SHOULDER_L, B_SHOULDER_R, B_ELBOW_L, B_ELBOW_R,
                              B_WRIST_L, B_WRIST_R)

W, H = 1280, 720
T = 40
rng = np.random.default_rng(0)


def make_pose(T=T, sway=0.0):
    p = np.full((T, N_POSE, 3), np.nan, dtype=np.float32)
    t = np.linspace(0, 1, T)
    dx = sway * np.sin(2 * np.pi * t)
    p[:, B_SHOULDER_L, :2] = np.stack([0.42 + dx, np.full(T, 0.58)], 1)
    p[:, B_SHOULDER_R, :2] = np.stack([0.58 + dx, np.full(T, 0.58)], 1)
    p[:, B_NOSE, :2] = np.stack([0.50 + dx, np.full(T, 0.34)], 1)
    p[:, B_MOUTH_L, :2] = np.stack([0.48 + dx, np.full(T, 0.42)], 1)
    p[:, B_MOUTH_R, :2] = np.stack([0.52 + dx, np.full(T, 0.42)], 1)
    p[:, B_EAR_L, :2] = np.stack([0.44 + dx, np.full(T, 0.32)], 1)
    p[:, B_EAR_R, :2] = np.stack([0.56 + dx, np.full(T, 0.32)], 1)
    p[:, B_ELBOW_L, :2] = np.stack([0.36 + dx, np.full(T, 0.75)], 1)
    p[:, B_ELBOW_R, :2] = np.stack([0.64 + dx, np.full(T, 0.75)], 1)
    p[:, B_WRIST_L, :2] = np.stack([0.40 + dx, np.full(T, 0.66)], 1)
    p[:, B_WRIST_R, :2] = np.stack([0.60 + dx, np.full(T, 0.66)], 1)
    p[:, :, 2] = 0.0
    return p


def make_hand(start_xy, delta_xy, T=T, missing=()):
    """A rigid 21-point hand translated linearly from start by delta."""
    base = np.zeros((21, 3), dtype=np.float32)
    base[:, 0] = np.linspace(0, 0.05, 21)
    base[:, 1] = np.linspace(0, -0.06, 21)
    seq = np.full((T, 21, 3), np.nan, dtype=np.float32)
    t = np.linspace(0, 1, T)[:, None]
    for i in range(T):
        if i in missing:
            continue
        off = np.array(start_xy) + np.array(delta_xy) * t[i]
        seq[i] = base + np.array([off[0], off[1], 0.0], dtype=np.float32)
    return seq


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    assert cond, name


# --- 1. shapes line up with the declared feature names ----------------------
pose = make_pose()
right = make_hand((0.60, 0.42), (0.0, 0.18))          # from mouth, downward
left = None_arr = np.full((T, 21, 3), np.nan, dtype=np.float32)   # no left hand

names_local = v2.make_feature_names()
names_body = v3.make_body_names()

f_all = v3.build_clip_features_v3(left, right, pose, W, H)
f_loc = v3.build_clip_features_v3(left, right, pose, W, H, with_body=False)
f_bod = v3.build_clip_features_v3(left, right, pose, W, H, with_hand_local=False)

check("hand-local length == v2 names", len(f_loc) == len(names_local),
      f"({len(f_loc)} vs {len(names_local)})")
check("body length == body names", len(f_bod) == len(names_body),
      f"({len(f_bod)} vs {len(names_body)})")
check("combined length", len(f_all) == len(names_local) + len(names_body),
      f"({len(f_all)})")
check("no NaN in output", np.isfinite(f_all).all())

# --- 2. hand-local block is bit-identical to what v2 would have produced ----
from mp_hand_detector import normalize_landmark_sequence
r_compact = v3.compact(right)
v2_vec = v2.build_clip_features(None, normalize_landmark_sequence(r_compact))
check("hand-local block == v2 exactly", np.allclose(f_loc, v2_vec, equal_nan=True))

# --- 3. THE POINT: same hand motion at a different body location ------------
# Two clips with identical handshape and identical relative motion, differing
# only in WHERE on the body they happen - the HUNGRY/THANK-YOU failure mode.
at_mouth = make_hand((0.60, 0.42), (0.0, 0.10))
at_chest = make_hand((0.60, 0.62), (0.0, 0.10))

loc_mouth = v3.build_clip_features_v3(left, at_mouth, pose, W, H, with_body=False)
loc_chest = v3.build_clip_features_v3(left, at_chest, pose, W, H, with_body=False)
bod_mouth = v3.build_clip_features_v3(left, at_mouth, pose, W, H, with_hand_local=False)
bod_chest = v3.build_clip_features_v3(left, at_chest, pose, W, H, with_hand_local=False)

check("v2 block CANNOT tell the two apart",
      np.allclose(loc_mouth, loc_chest, atol=1e-5),
      f"(max diff {np.abs(loc_mouth-loc_chest).max():.2e})")
check("v3 body block CAN tell them apart",
      np.abs(bod_mouth - bod_chest).max() > 0.5,
      f"(max diff {np.abs(bod_mouth-bod_chest).max():.3f})")

i_dnose = names_body.index("right_body_kf0_dnose_y")
print(f"      wrist-to-nose dy at kf0:  mouth-height {bod_mouth[i_dnose]:+.3f}  "
      f"chest-height {bod_chest[i_dnose]:+.3f}  (shoulder-widths)")

# --- 4. aspect correction actually does something --------------------------
bod_square = v3.build_clip_features_v3(left, at_mouth, pose, 720, 720, with_hand_local=False)
check("aspect ratio changes body features",
      not np.allclose(bod_mouth, bod_square, atol=1e-6))

# --- 5. invariance to where the signer stands ------------------------------
shift = 0.08
pose_shift = make_pose()
pose_shift[:, :, 0] += shift
hand_shift = make_hand((0.60 + shift, 0.42), (0.0, 0.10))
bod_shift = v3.build_clip_features_v3(left, hand_shift, pose_shift, W, H, with_hand_local=False)
check("body block is invariant to signer position in frame",
      np.allclose(bod_mouth, bod_shift, atol=1e-4),
      f"(max diff {np.abs(bod_mouth-bod_shift).max():.2e})")

# --- 6. tolerant of pose dropouts and hand gaps ----------------------------
pose_gap = make_pose()
pose_gap[10:18] = np.nan
hand_gap = make_hand((0.60, 0.42), (0.0, 0.10), missing=tuple(range(12, 16)))
f_gap = v3.build_clip_features_v3(left, hand_gap, pose_gap, W, H)
check("gaps produce finite features", np.isfinite(f_gap).all())
check("body still marked present through a gap",
      f_gap[len(names_local) + names_body.index("body_present")] == 1)

# total pose loss -> body block zeroed, hand-local block intact
pose_none = np.full((T, N_POSE, 3), np.nan, dtype=np.float32)
f_nopose = v3.build_clip_features_v3(left, right, pose_none, W, H)
check("no pose -> body block all zero",
      np.all(f_nopose[len(names_local):] == 0))
check("no pose -> hand-local block unaffected",
      np.allclose(f_nopose[:len(names_local)], f_loc))

# --- 7. mirroring swaps hands AND the body reference consistently ----------
two_handed_r = make_hand((0.62, 0.45), (0.05, 0.10))
two_handed_l = make_hand((0.38, 0.45), (-0.05, 0.10))
f_orig = v3.build_clip_features_v3(two_handed_l, two_handed_r, pose, W, H,
                                   with_hand_local=False)
m_left = m3.flip_x_raw(two_handed_r)
m_right = m3.flip_x_raw(two_handed_l)
m_pose = m3.mirror_pose(pose)
f_mir = v3.build_clip_features_v3(m_left, m_right, m_pose, W, H, with_hand_local=False)

# The mirrored LEFT block should equal the original RIGHT block with x negated.
n_per = v3.N_BODY_PER_HAND
orig_right = f_orig[n_per:2 * n_per]
mir_left = f_mir[:n_per]
x_cols = np.array([n.endswith("_x") for n in names_body[:n_per]])
sign = np.where(x_cols, -1.0, 1.0)
check("mirror: left block == right block with x negated",
      np.allclose(mir_left, orig_right * sign, atol=1e-4),
      f"(max diff {np.abs(mir_left - orig_right*sign).max():.2e})")

mirrored_shoulders_ok = np.allclose(m_pose[:, B_SHOULDER_L, 0], 1 - pose[:, B_SHOULDER_R, 0])
check("mirror: pose left/right pairs swapped", mirrored_shoulders_ok)

print("\nfeature counts:  hand-local %d + body %d = %d"
      % (len(names_local), len(names_body), len(names_local) + len(names_body)))
print("body block per hand: %d, shared: %d" % (v3.N_BODY_PER_HAND, v3.N_BODY_SHARED))
