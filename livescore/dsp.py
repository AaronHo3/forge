"""
dsp.py — shared output-conditioning DSP.

A soft-knee limiter and a loudness normalizer, used by BOTH the live engine
(mrt_controller.py) and the offline keepsake renderer (keepsake.py), so the live
take and its rendered keepsake are conditioned the same way.
"""

from __future__ import annotations

import numpy as np


def soft_limit(x: np.ndarray, th: float = 0.85) -> np.ndarray:
    """Soft-knee limiter: linear below `th`, tanh-compressed above, so loud
    transients are smoothly tamed instead of clipping harshly. Returns a new
    array when limiting is applied; the input is never mutated."""
    a = np.abs(x)
    over = a > th
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (th + (1.0 - th) * np.tanh((a[over] - th) / (1.0 - th)))
    return x


def normalize(buf: np.ndarray, target_rms: float = 0.16) -> np.ndarray:
    """Scale `buf` so its RMS equals `target_rms`. Silence (near-zero RMS) is
    left untouched to avoid a divide-by-zero blowup."""
    rms = float(np.sqrt(np.mean(buf ** 2)))
    if rms > 1e-5:
        buf = buf * (target_rms / rms)
    return buf
