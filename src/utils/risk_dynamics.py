"""
Lightweight risk dynamics utilities for online smoothing and change detection.

Designed to be CPU-friendly and work with short sequences (e.g., last 30 frames).
Use cases:
- Replace ad-hoc EMA in the pipeline
- Emit escalation onsets with bounded false alarms

APIs are minimal to slot into existing `multimodal.py` without heavy refactors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math


class ExponentialMovingAverage:
    """Simple EMA smoother.

    alpha in (0,1]; higher alpha reacts faster.
    """

    def __init__(self, alpha: float = 0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1]")
        self.alpha = alpha
        self._state: Optional[float] = None

    def reset(self) -> None:
        self._state = None

    def update(self, value: float) -> float:
        if self._state is None:
            self._state = float(value)
        else:
            self._state = self.alpha * float(value) + (1.0 - self.alpha) * self._state
        return self._state


@dataclass
class CUSUMConfig:
    """Configuration for one-sided or two-sided CUSUM change detection.

    - target_mean: expected baseline mean of the signal
    - drift: allowable small deviation before we accumulate evidence (k in CUSUM)
    - threshold: detection threshold (h). Larger → fewer false alarms, slower detection
    - two_sided: detect both increases and decreases if True
    """

    target_mean: float = 0.0
    drift: float = 0.05
    threshold: float = 1.0
    two_sided: bool = True


class OnlineCUSUM:
    """Online CUSUM change detector.

    For a risk stream r_t in [0, 4], choose target_mean near the expected baseline
    (e.g., 1.0–1.5). Set drift small (e.g., 0.05–0.15). Tune threshold for desired
    false-alarm rate (start around 1.0–3.0).
    """

    def __init__(self, config: CUSUMConfig):
        self.config = config
        self._pos = 0.0
        self._neg = 0.0

    def reset(self) -> None:
        self._pos = 0.0
        self._neg = 0.0

    def update(self, value: float) -> Tuple[bool, Optional[str], float, float]:
        """Update with one value.

        Returns:
            detected: whether a change is detected
            direction: 'up'|'down'|None
            s_pos: current positive CUSUM score
            s_neg: current negative CUSUM score
        """
        x = float(value) - self.config.target_mean
        # Positive shift
        self._pos = max(0.0, self._pos + x - self.config.drift)
        # Negative shift
        self._neg = min(0.0, self._neg + x + self.config.drift)

        if self._pos > self.config.threshold:
            self._pos = 0.0  # reset after alarm
            return True, "up", 0.0, self._neg
        if self.config.two_sided and abs(self._neg) > self.config.threshold:
            self._neg = 0.0
            return True, "down", self._pos, 0.0
        return False, None, self._pos, self._neg


class OnlineZScore:
    """Welford online mean/variance with rolling z-score anomaly flag.

    Useful as an OOD/shift heuristic on fused risk features.
    """

    def __init__(self, min_count: int = 20, z_thresh: float = 3.0):
        self.min_count = max(2, int(min_count))
        self.z_thresh = float(z_thresh)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> Tuple[float, bool]:
        x = float(value)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

        if self._n < self.min_count:
            return 0.0, False

        var = self._m2 / (self._n - 1)
        std = math.sqrt(max(var, 1e-12))
        z = 0.0 if std == 0.0 else (x - self._mean) / std
        return z, abs(z) >= self.z_thresh


def discretize_risk(score: float) -> int:
    """Map continuous risk [0,4] to {0,1,2,3} levels with stable bounds."""
    s = max(0.0, min(4.0, float(score)))
    if s < 0.75:
        return 0
    if s < 1.75:
        return 1
    if s < 2.75:
        return 2
    return 3


