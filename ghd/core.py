"""GHD core: per-layer gradient-health signals, detection and confidence scores.

All state here is per layer and pure-Python/NumPy; the hook (``ghd.hook``)
feeds it the scalar norms/cosines it measures on the live gradients.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


EPS = 1e-8
WINDOW = 20
WARMUP_STEPS = 100

# Controller classes (order matters: it is the MLP head's output order).
MODES = ["healthy", "vanishing", "exploding", "oscillating", "noisy"]
# Everything the detector can report after warm-up.
ALL_MODES = MODES + ["recovering"]
RECOVERY_WINDOW = 10

# Detector thresholds.
VANISH_S2_EARLY, VANISH_S3_EARLY = 0.30, 0.40
VANISH_S2_STRICT, VANISH_S3_STRICT = 0.10, 0.20
VANISH_S3_ONLY = 0.02
SDEPTH_PERSIST, SDEPTH_STRUCTURAL = 0.05, 0.01
OSCIL_S6 = -0.30
NOISY_S5 = 0.82
EXPLODE_FACTOR = 10.0


def sigmoid_clipped(x: float) -> float:
    """sigmoid(clip(x, -3, 3)) -> always within [0.047, 0.953]."""
    x = max(-3.0, min(3.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def confidence_vanish(s_depth: float) -> float:
    """1 at total structural vanishing, 0 once S_depth reaches the 0.01 threshold."""
    return float(np.clip(1.0 - s_depth / SDEPTH_STRUCTURAL, 0.0, 1.0))


def confidence_vanish_from_s2_s3(s2: float, s3: float) -> float:
    """Strength for the non-structural vanishing rules (strict / early / S3-only), where
    S_depth >= 0.01 and ``confidence_vanish`` would be 0. 0.5 at S2=0.10, S3=0.20."""
    return sigmoid_clipped(3 * (VANISH_S3_STRICT - s3) + 3 * (VANISH_S2_STRICT - s2))


def confidence_oscil(s4: float, s6: float) -> float:
    return sigmoid_clipped(3 * (0.60 - s4) + 3 * abs(min(s6, 0.0)))


def confidence_noise(s5: float) -> float:
    return sigmoid_clipped(3 * (s5 - NOISY_S5))


def confidence_explode(ratio: float) -> float:
    """Not in the spec's confidence set; used only as the strength label for
    'exploding' steps. It is 0.5 at the 10x trigger point and grows with the spike."""
    return sigmoid_clipped(3 * (math.log10(max(ratio, EPS)) - 1.0))


@dataclass
class LayerSignals:
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s_depth: float
    median: float  # median of the S1 history *before* this step
    n_prior: int  # number of S1 observations behind ``median``


class LayerState:
    """Rolling history and the three independent persistence counters."""

    def __init__(self) -> None:
        self.s1_hist: deque[float] = deque(maxlen=WINDOW)
        self.cos_hist: deque[float] = deque(maxlen=WINDOW)
        # Separate per mode; never shared.
        self.p_vanish = 0
        self.p_oscil = 0
        self.p_noisy = 0
        self.last_vanish_step: int | None = None

    def compute_signals(self, s1: float, s3: float, s_depth: float, cos: float | None) -> LayerSignals:
        n_prior = len(self.s1_hist)
        median = float(np.median(self.s1_hist)) if n_prior else s1
        s2 = s1 / (median + EPS)

        if cos is None or not math.isfinite(cos) or s1 < 0.01 * median:
            s4 = 0.5  # masked
        else:
            s4 = (max(-1.0, min(1.0, cos)) + 1.0) / 2.0

        self.s1_hist.append(s1)
        self.cos_hist.append(2.0 * s4 - 1.0)

        if len(self.s1_hist) >= 3:
            arr = np.asarray(self.s1_hist)
            s5 = float(np.clip(arr.std() / (arr.mean() + EPS), 0.0, 1.0))
        else:
            s5 = 0.0
        s6 = float(np.mean(self.cos_hist))
        s7 = float(np.clip(1.0 - s5 - max(0.0, -s6), 0.0, 1.0))
        return LayerSignals(s1, s2, s3, s4, s5, s6, s7, s_depth, median, n_prior)

    def update_counters(self, sig: LayerSignals) -> None:
        vanishing = (sig.s2 < VANISH_S2_EARLY and sig.s3 < VANISH_S3_EARLY) or sig.s_depth < SDEPTH_PERSIST
        self.p_vanish = self.p_vanish + 1 if vanishing else 0
        self.p_oscil = self.p_oscil + 1 if sig.s6 < OSCIL_S6 else 0
        self.p_noisy = self.p_noisy + 1 if sig.s5 > NOISY_S5 else 0

    def is_spike(self, sig: LayerSignals) -> bool:
        return sig.n_prior >= 3 and sig.s1 > EXPLODE_FACTOR * sig.median

    def recently_vanishing(self, step: int) -> bool:
        return self.last_vanish_step is not None and step - self.last_vanish_step < RECOVERY_WINDOW

    def detect(self, sig: LayerSignals, step: int, enable_noisy: bool = False) -> tuple[str, str]:
        """Strict-priority rule detector. Returns (mode, rule_name)."""
        if sig.s_depth < SDEPTH_STRUCTURAL and self.p_vanish >= 5:
            return "vanishing", "structural_vanishing"
        if self.is_spike(sig):
            if self.recently_vanishing(step):
                return "recovering", "recovering"
            return "exploding", "exploding"
        if sig.s2 < VANISH_S2_STRICT and sig.s3 < VANISH_S3_STRICT and self.p_vanish >= 5:
            return "vanishing", "vanishing_strict"
        if sig.s2 < VANISH_S2_EARLY and sig.s3 < VANISH_S3_EARLY and self.p_vanish >= 3:
            return "vanishing", "vanishing_early"
        if sig.s3 < VANISH_S3_ONLY and self.p_vanish >= 5:
            return "vanishing", "vanishing_s3"
        if sig.s6 < OSCIL_S6 and self.p_oscil >= 5:
            return "oscillating", "oscillating"
        if enable_noisy and sig.s5 > NOISY_S5 and self.p_noisy >= 5:
            return "noisy", "noisy"
        return "healthy", "healthy"


def rule_strength(mode: str, sig: LayerSignals, rule: str = "") -> float:
    if mode == "vanishing":
        if rule == "structural_vanishing":
            return confidence_vanish(sig.s_depth)
        return confidence_vanish_from_s2_s3(sig.s2, sig.s3)
    if mode == "oscillating":
        return confidence_oscil(sig.s4, sig.s6)
    if mode == "noisy":
        return confidence_noise(sig.s5)
    if mode == "exploding":
        return confidence_explode(sig.s1 / (sig.median + EPS))
    return 0.0
