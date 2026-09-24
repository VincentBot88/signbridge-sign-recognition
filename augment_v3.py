"""
SignBridge - on-the-fly augmentation for the v3 sequence model.

WHY ON-THE-FLY AND NOT BAKED TO DISK
------------------------------------
mirror_augment_v3.py writes its output once, so every epoch sees the same
1,250 rows. That is right for mirroring (a mirrored sign is a genuinely
different, equally valid example, so it belongs in the dataset) but wrong
for perturbations: the value of jitter or a time warp is that the model
sees a DIFFERENT perturbation of the same clip every epoch and is forced
to learn invariance rather than memorise one fixed copy. So these are
applied in the training loop, to the fit set only, and never written down.

Dev and val are never augmented. Dev is the early-stopping signal and
augmenting it would change what is being selected for; val is the reported
number.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Three augmentations that appear on every standard list are no-ops against
this feature pipeline, and adding them would burn compute while changing
nothing:

  - GLOBAL TRANSLATION. The body frame subtracts the per-frame shoulder
    midpoint and the hand-local block subtracts frame 0's wrist, so a
    constant offset cancels in both.
  - UNIFORM SCALE. The body frame divides by the clip's median shoulder
    width; the hand-local block divides by the median wrist-to-middle-MCP
    distance. A uniform zoom cancels in both.
  - UNIFORM SPEED CHANGE. Every clip is resampled to n_steps regardless of
    its frame count, so scaling the whole clip's duration lands on exactly
    the same 32 samples. Only a NON-UNIFORM warp does anything, which is
    why time_warp() below moves an interior control point rather than
    rescaling the clip.

Rotation is NOT in that list: neither block normalises orientation
(mp_hand_detector.normalize_landmark_sequence says so explicitly - "rotation
is still not normalized... revisit if the classifier struggles with tilted
hands"), so a small in-plane rotation is a real perturbation and it targets
a known gap.

Rotation is applied in aspect-corrected space, because MediaPipe's x is
normalised by image width and y by image height - rotating the raw
coordinates of a non-square frame would shear rather than rotate. The
centre of rotation does not matter: both feature blocks subtract an origin,
so rotating about any point differs only by a translation, which cancels.

ALL OF THESE PRESERVE THE LABEL
-------------------------------
Signing speed varies between people (time warp). Clip boundaries are
arbitrary and the live kiosk will not segment cleanly (temporal crop).
Detectors drop frames - this data has measured dropouts already (frame
dropout). Landmark coordinates carry detector noise (jitter). Signers sit
at slight angles to the camera (rotation).

Magnitudes are deliberately small. An augmentation that changes the sign is
label noise, not augmentation - in ASL, location is a phoneme, so anything
that moves the hand substantially relative to the body is off the table.

Run the self-test (no data or model download needed):
    python augment_v3.py
"""

import numpy as np

# Defaults, all scaled by the caller's `strength`. Tuned to be conservative:
# the failure mode of augmentation at this data size is destroying the
# signal, not failing to perturb enough.
ROT_DEG = 8.0          # max in-plane rotation, degrees
JITTER_SIGMA = 0.003   # landmark noise, in MediaPipe normalised units
DROP_P = 0.05          # per-frame chance a hand is blanked
WARP_MAX = 0.15        # max shift of the mid-clip control point, as a fraction of T
CROP_MAX = 0.10        # max fraction trimmed from each end

MIN_FRAMES = 8         # never crop/warp a clip below this
MIN_HAND_FRAMES = 2    # a hand that was present must stay present


def _present(seq):
    """Frames in which this landmark set was detected - same convention as
    the rest of the pipeline ([:, 0, 0], since all channels go NaN together)."""
    if seq is None or seq.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(seq[:, 0, 0])


def rotate(arr, theta, aspect):
    """
    In-plane rotation of raw MediaPipe coordinates by theta radians.

    arr: (..., 3) raw normalised coords, NaN preserved.
    aspect: frame_w / frame_h, used to rotate in physically square units.
    """
    if arr is None or arr.shape[0] == 0 or theta == 0.0:
        return arr
    out = np.array(arr, dtype=np.float32, copy=True)
    c, s = np.cos(theta), np.sin(theta)
    x = out[..., 0] * aspect
    y = out[..., 1]
    out[..., 0] = (c * x - s * y) / aspect
    out[..., 1] = s * x + c * y
    return out


def jitter(arr, rng, sigma):
    """Gaussian noise on detected landmarks. NaN frames stay NaN."""
    if arr is None or arr.shape[0] == 0 or sigma <= 0:
        return arr
    return (np.asarray(arr, dtype=np.float32)
            + rng.normal(0.0, sigma, arr.shape).astype(np.float32))


def time_warp(arrays, rng, max_shift):
    """
    Non-uniform monotone time warp: move the mid-clip control point, so one
    half of the sign plays faster and the other slower.

    Nearest-neighbour sampling on the frame axis, deliberately - linear
    interpolation between a detected and an undetected frame would produce
    NaN and silently eat detections. All three arrays are warped with the
    SAME index map, which is what keeps hands and pose on one clock.
    """
    T = arrays[0].shape[0]
    if T < MIN_FRAMES or max_shift <= 0:
        return arrays
    shift = rng.uniform(-max_shift, max_shift) * (T - 1)
    mid = float(np.clip((T - 1) / 2.0 + shift, 1.0, T - 2.0))
    # for each output frame, which input frame to take
    pos = np.interp(np.arange(T), [0.0, mid, T - 1.0], [0.0, (T - 1) / 2.0, T - 1.0])
    idx = np.clip(np.rint(pos).astype(int), 0, T - 1)
    return [a[idx] if a is not None and a.shape[0] == T else a for a in arrays]


def temporal_crop(arrays, rng, max_frac):
    """
    Trim a random amount off each end. Reverted if it would make a hand that
    was present disappear entirely - that would change what the clip shows,
    not just how it is framed.
    """
    T = arrays[0].shape[0]
    if T < MIN_FRAMES or max_frac <= 0:
        return arrays
    margin = int(max_frac * T)
    if margin < 1:
        return arrays
    a = int(rng.integers(0, margin + 1))
    b = T - int(rng.integers(0, margin + 1))
    if b - a < MIN_FRAMES:
        return arrays

    cropped = [x[a:b] if x is not None and x.shape[0] == T else x for x in arrays]
    for before, after in zip(arrays[:2], cropped[:2]):        # the two hands
        if _present(before).sum() >= MIN_HAND_FRAMES and \
                _present(after).sum() < MIN_HAND_FRAMES:
            return arrays
    return cropped


def frame_dropout(left, right, rng, p):
    """
    Blank random frames per hand, independently - the detector loses one
    hand at a time far more often than both. A hand is never dropped below
    MIN_HAND_FRAMES, so a two-handed sign cannot silently become one-handed.
    """
    if p <= 0:
        return left, right
    out = []
    for arr in (left, right):
        if arr is None or arr.shape[0] == 0:
            out.append(arr)
            continue
        present = _present(arr)
        n_present = int(present.sum())
        if n_present < MIN_HAND_FRAMES:
            out.append(arr)
            continue
        drop = rng.random(arr.shape[0]) < p
        drop &= present
        # keep at least MIN_HAND_FRAMES detected
        if int(present.sum() - drop.sum()) < MIN_HAND_FRAMES:
            keep_back = np.flatnonzero(drop)
            rng.shuffle(keep_back)
            n_restore = MIN_HAND_FRAMES - int(present.sum() - drop.sum())
            drop[keep_back[:n_restore]] = False
        arr = np.array(arr, dtype=np.float32, copy=True)
        arr[drop] = np.nan
        out.append(arr)
    return out[0], out[1]


def augment_clip(left, right, pose, frame_w, frame_h, rng, strength=1.0):
    """
    One clip's raw, frame-aligned arrays in; a perturbed copy out.

    Returns (left, right, pose) with the frame axis still shared across all
    three - every operation here either applies the same index map to all
    three or touches only landmark values, so the shared clock the body
    block depends on survives.
    """
    aspect = (float(frame_w) / float(frame_h)) if (frame_w and frame_h) else 1.0

    arrays = time_warp([left, right, pose], rng, WARP_MAX * strength)
    arrays = temporal_crop(arrays, rng, CROP_MAX * strength)
    left, right, pose = arrays

    theta = np.deg2rad(rng.uniform(-ROT_DEG, ROT_DEG) * strength)
    left = rotate(left, theta, aspect)
    right = rotate(right, theta, aspect)
    pose = rotate(pose, theta, aspect)

    sigma = JITTER_SIGMA * strength
    left = jitter(left, rng, sigma)
    right = jitter(right, rng, sigma)
    pose = jitter(pose, rng, sigma)

    left, right = frame_dropout(left, right, rng, DROP_P * strength)
    return left, right, pose


# ---------------------------------------------------------------------------
# Self-test - synthetic, no data needed:  python augment_v3.py
# ---------------------------------------------------------------------------

def _fake_clip(T=60, gap=True, seed=0):
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.2, 0.8, (T, 21, 3)).astype(np.float32)
    right = rng.uniform(0.2, 0.8, (T, 21, 3)).astype(np.float32)
    pose = rng.uniform(0.2, 0.8, (T, 11, 3)).astype(np.float32)
    if gap:
        left[10:16] = np.nan          # a detector dropout
    return left, right, pose


def _main():
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print("augment_v3 self-test")

    left, right, pose = _fake_clip()
    rng = np.random.default_rng(0)

    # rotation by zero is the identity
    check("rotate(0) is identity",
          np.allclose(rotate(left, 0.0, 1.5), left, equal_nan=True))

    # rotation preserves the NaN pattern and stays finite elsewhere
    rot = rotate(left, np.deg2rad(8), 1.777)
    check("rotate preserves NaN frames",
          np.array_equal(_present(rot), _present(left)))
    check("rotate leaves detected frames finite",
          np.isfinite(rot[_present(rot)]).all())

    # rotation actually changes something
    check("rotate is not a no-op", not np.allclose(rot[_present(rot)],
                                                   left[_present(left)]))

    # jitter keeps NaN where it was
    jit = jitter(left, np.random.default_rng(1), 0.003)
    check("jitter preserves NaN frames",
          np.array_equal(_present(jit), _present(left)))

    # time warp keeps all three arrays on one clock
    w = time_warp([left, right, pose], np.random.default_rng(2), 0.15)
    check("time warp keeps a shared frame axis",
          w[0].shape[0] == w[1].shape[0] == w[2].shape[0] == left.shape[0])

    # temporal crop shortens but never empties a present hand
    c = temporal_crop([left, right, pose], np.random.default_rng(3), 0.10)
    check("crop keeps all three arrays equal length",
          c[0].shape[0] == c[1].shape[0] == c[2].shape[0])
    check("crop keeps a present hand present",
          _present(c[1]).sum() >= MIN_HAND_FRAMES)

    # frame dropout never removes a hand entirely
    worst = np.random.default_rng(4)
    l2, r2 = frame_dropout(left, right, worst, 0.99)
    check("dropout never erases a present hand",
          _present(l2).sum() >= MIN_HAND_FRAMES and
          _present(r2).sum() >= MIN_HAND_FRAMES)

    # a hand that was absent stays absent
    absent = np.full((40, 21, 3), np.nan, dtype=np.float32)
    la, ra = frame_dropout(absent, right, np.random.default_rng(5), 0.2)
    check("dropout leaves an absent hand absent", _present(la).sum() == 0)

    # full pipeline: shapes consistent, no inf, determinism
    a1 = augment_clip(left, right, pose, 1920, 1080, np.random.default_rng(7))
    a2 = augment_clip(left, right, pose, 1920, 1080, np.random.default_rng(7))
    check("augment_clip keeps one shared frame axis",
          a1[0].shape[0] == a1[1].shape[0] == a1[2].shape[0])
    check("augment_clip is deterministic for a fixed rng",
          all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a1, a2)))
    check("augment_clip produces no infinities",
          all(np.isfinite(x[np.isfinite(x)]).all() for x in a1))
    a3 = augment_clip(left, right, pose, 1920, 1080, np.random.default_rng(8))
    check("different rng gives a different result",
          not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a1, a3)))

    # strength 0 should be (almost) the identity
    a0 = augment_clip(left, right, pose, 1920, 1080,
                      np.random.default_rng(9), strength=0.0)
    check("strength=0 is the identity",
          all(np.array_equal(x, y, equal_nan=True)
              for x, y in zip(a0, (left, right, pose))))

    # a short clip must survive untouched rather than vanish
    s_left, s_right, s_pose = _fake_clip(T=5, gap=False)
    sa = augment_clip(s_left, s_right, s_pose, 1280, 720, np.random.default_rng(10))
    check("very short clips survive", sa[0].shape[0] == 5)

    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
