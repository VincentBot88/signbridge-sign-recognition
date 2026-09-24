"""
SignBridge - the frozen v3 recognizer: RandomForest + 5 GRUs, one predict().

This is the model the kiosk runs. It loads ONE file written by
build_ensemble_bundle.py and turns a sign's raw landmarks into a label, a
confidence, and an accept / ask-to-confirm decision.

THE RECIPE (frozen 2026-09-23, see signbridge-ensemble-v3.md)
-------------------------------------------------------------
  RF    RandomForest on the 1,221 v3 features (hand-local + body),
        random_state 42, trained on the 1,000-row fit set
  GRU   5 body-only GRUs (seeds 42-46, strength-3 augmentation, stream 0),
        the exact models the ensemble experiment measured
  each model's probabilities are temperature-scaled (temperatures fit on dev),
  the 5 GRUs are averaged, and then

        P = 0.5 * RF + 0.5 * mean(GRU_1..GRU_5)

  accepted  iff  max(P) >= threshold   (threshold chosen on dev)

  On val: RF+GRU pair 114.5/124, RF x5 + GRU x5 116-117/124. When the
  kiosk declines its least-confident 20%, 0.7% of what it accepts is wrong.

NO TORCH AT RUN TIME
--------------------
The GRU forward pass is reimplemented here in numpy (it is 32 steps of one
64-unit layer). The bundle stores the weights as numpy arrays, so the kiosk
needs numpy + scikit-learn + mediapipe, not PyTorch:
  * the Raspberry Pi 5 does not have to carry a PyTorch install;
  * Windows Smart App Control, which has blocked torch's DLLs on the demo
    laptop before, is not in the demo path at all.
build_ensemble_bundle.py checks this numpy forward pass against PyTorch on
every dev and val clip before it will save a bundle.

USE
---
    from signbridge_ensemble import SignBridgeEnsemble
    ens = SignBridgeEnsemble.load()                   # models/signbridge_ensemble_v3.joblib
    out = ens.predict(left, right, pose, frame_w, frame_h)
    out["label"], out["confidence"], out["accepted"], out["top"]

left / right: (T, 21, 3) raw MediaPipe hand landmarks, NaN on frames the hand
was not detected (or None for a hand never seen). pose: (T, 11, 3) raw, the
mp_body_detector.POSE_KEEP subset, NaN where missing. Exactly the arrays
extract_landmarks_v3.py stores and live_recognize_v3.py buffers.
"""

import hashlib
import os

import numpy as np

from build_feature_vectors_v3 import build_clip_features_v3
from train_pytorch_v3 import clip_to_sequence_v3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(BASE_DIR, "models", "signbridge_ensemble_v3.joblib")
BUNDLE_FORMAT = "signbridge-ensemble-v3"


# ---------------------------------------------------------------------------
# Shared maths - build_ensemble_bundle.py and evaluate_ensemble_on_test.py use
# these exact functions, so what is calibrated is what is served.
# ---------------------------------------------------------------------------

def temper(P, temp):
    """Temperature-scale probabilities: softmax(log P / T). Same as ensemble_v3."""
    z = np.log(np.clip(P, 1e-7, 1.0)) / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_weights_from_state(state):
    """torch GRUClassifier state_dict (tensors or arrays) -> numpy float32 dict."""
    def arr(v):
        return (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)).astype(np.float32)
    keys = ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0",
            "gru.bias_hh_l0", "fc.weight", "fc.bias")
    missing = [k for k in keys if k not in state]
    if missing:
        raise ValueError(f"not a 1-layer GRUClassifier state dict, missing {missing}")
    return {k: arr(state[k]) for k in keys}


def gru_forward_numpy(w, X):
    """
    Softmax output of train_pytorch_v3's GRUClassifier, in numpy.

    X: (N, T, F) float32. PyTorch's nn.GRU gate order is (reset, update, new):
        r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        h = (1 - z) * n + z * h
    then logits = fc(h_T). Dropout is inactive at inference, as in eval().
    """
    X = np.asarray(X, dtype=np.float32)
    W_ih, W_hh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"]
    b_ih, b_hh = w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
    H = W_hh.shape[1]
    h = np.zeros((X.shape[0], H), dtype=np.float32)
    gi_all = X @ W_ih.T + b_ih                         # (N, T, 3H)
    for t in range(X.shape[1]):
        gi = gi_all[:, t, :]
        gh = h @ W_hh.T + b_hh
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h
    logits = h @ w["fc.weight"].T + w["fc.bias"]
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)


def _as_hand(raw, T, min_frames):
    """None / too-sparse hand -> all-NaN (T, 21, 3). Same as training storage."""
    if raw is None or raw.shape[0] == 0:
        return np.full((T, 21, 3), np.nan, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    if min_frames and int(np.isfinite(raw[:, 0, 0]).sum()) < min_frames:
        return np.full_like(raw, np.nan)
    return raw


def clip_inputs(left, right, pose, frame_w, frame_h, gru_cfg, min_hand_frames=0):
    """
    One clip's raw landmarks -> (RF feature vector, GRU sequence).

    Both come from the functions the models were trained with -
    build_clip_features_v3 for the RF and clip_to_sequence_v3 for the GRU -
    so serving cannot drift from training.
    """
    T = next((a.shape[0] for a in (left, right, pose) if a is not None), 0)
    if T == 0:
        raise ValueError("no frames")
    left = _as_hand(left, T, min_hand_frames)
    right = _as_hand(right, T, min_hand_frames)
    rf_x = build_clip_features_v3(left, right, pose, frame_w, frame_h)
    d = {"left_hand": left, "right_hand": right, "frame_w": frame_w, "frame_h": frame_h}
    if pose is not None:
        d["pose"] = pose
    seq, _, _, _ = clip_to_sequence_v3(d, gru_cfg["timesteps"],
                                       with_hand_local=gru_cfg["with_hand_local"],
                                       with_body=gru_cfg["with_body"])
    return rf_x, seq


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The recognizer
# ---------------------------------------------------------------------------

class SignBridgeEnsemble:
    def __init__(self, bundle, path=None):
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a {BUNDLE_FORMAT} bundle (format={bundle.get('format')!r})")
        self.bundle = bundle
        self.path = path
        self.classes = list(bundle["classes"])
        self.rf = bundle["rf"]["model"]
        self.rf_temp = float(bundle["rf"]["temperature"])
        self.gru_cfg = bundle["gru"]["config"]
        self.gru_weights = bundle["gru"]["weights"]
        self.gru_temps = [float(t) for t in bundle["gru"]["temperatures"]]
        self.w_rf = float(bundle["blend"]["rf"])
        self.w_gru = float(bundle["blend"]["gru"])
        self.threshold = bundle["threshold"]["value"]
        if list(self.rf.classes_) != list(range(len(self.classes))):
            raise ValueError("RF class indices do not match the bundle's class list")

    @classmethod
    def load(cls, path=DEFAULT_BUNDLE):
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run build_ensemble_bundle.py")
        return cls(joblib.load(path), path)

    # ---- batch API (evaluation) --------------------------------------------
    def component_probs(self, rf_X, gru_X):
        """Raw (uncalibrated) RF probs and the list of per-seed GRU probs."""
        P_rf = self.rf.predict_proba(np.asarray(rf_X, dtype=np.float64))
        P_gru = [gru_forward_numpy(w, gru_X) for w in self.gru_weights]
        return P_rf, P_gru

    def blend(self, P_rf, P_gru):
        rf_c = temper(P_rf, self.rf_temp)
        gru_c = np.mean([temper(P, t) for P, t in zip(P_gru, self.gru_temps)], axis=0)
        return self.w_rf * rf_c + self.w_gru * gru_c

    def predict_proba_batch(self, rf_X, gru_X):
        return self.blend(*self.component_probs(rf_X, gru_X))

    # ---- single-clip API (live) ---------------------------------------------
    def predict(self, left, right, pose, frame_w, frame_h, min_hand_frames=0, top_k=3):
        """
        Recognize one sign from its raw landmark window.

        Returns a dict:
          label       most likely sign
          confidence  its blended probability
          accepted    confidence >= the dev-chosen threshold. When False the
                      kiosk should ask the user to confirm rather than act.
          top         [(label, prob), ...] top_k alternatives
          rf_label / gru_label   each half's own vote, for debugging

        min_hand_frames: a hand detected on fewer frames than this is treated
        as absent (live_recognize_v3 uses its MIN_FRAMES_TO_PREDICT here).
        """
        rf_x, seq = clip_inputs(left, right, pose, frame_w, frame_h,
                                self.gru_cfg, min_hand_frames)
        P_rf, P_gru = self.component_probs(rf_x[None, :], seq[None, :, :])
        P = self.blend(P_rf, P_gru)[0]
        order = np.argsort(P)[::-1][:top_k]
        conf = float(P[order[0]])
        gru_mean = np.mean(P_gru, axis=0)[0]
        return {
            "label": self.classes[order[0]],
            "confidence": conf,
            "accepted": bool(self.threshold is None or conf >= self.threshold),
            "top": [(self.classes[i], float(P[i])) for i in order],
            "rf_label": self.classes[int(np.argmax(P_rf[0]))],
            "gru_label": self.classes[int(np.argmax(gru_mean))],
            "probs": P,
        }

    def describe(self):
        b = self.bundle
        th = self.threshold
        return (f"SignBridge ensemble v3: RF (T={self.rf_temp:.2f}) + "
                f"{len(self.gru_weights)} GRUs (T={', '.join(f'{t:.2f}' for t in self.gru_temps)}), "
                f"{self.w_rf:g}/{self.w_gru:g} blend, "
                f"threshold {'none' if th is None else f'{th:.3f}'} "
                f"(dev: {b['threshold'].get('dev_coverage', float('nan')):.0%} accepted at "
                f"{b['threshold'].get('dev_accepted_accuracy', float('nan')):.1%} accuracy)")
