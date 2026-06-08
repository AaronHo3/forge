"""Unit tests for dsp.py: the shared output-conditioning DSP (limiter + loudness).

Shared by the live engine and the keepsake renderer, so one tested copy keeps
their sound conditioning identical.
"""

import numpy as np
import pytest

import dsp


@pytest.mark.unit
class TestSoftLimit:
    def test_tames_peaks_below_unity(self):
        x = np.array([0.5, 0.9, 2.0, -2.0], dtype=np.float32)
        out = dsp.soft_limit(x, th=0.85)
        assert out[0] == pytest.approx(0.5)        # below the knee, untouched
        assert np.all(np.abs(out) < 1.0 + 1e-6)    # nothing exceeds unity
        assert abs(out[2]) < 2.0                   # the 2.0 peak was compressed

    def test_does_not_mutate_input(self):
        x = np.array([2.0, -2.0], dtype=np.float32)
        original = x.copy()
        dsp.soft_limit(x, th=0.85)
        assert np.array_equal(x, original)

    def test_leaves_quiet_signal_untouched(self):
        x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        assert np.array_equal(dsp.soft_limit(x, th=0.85), x)


@pytest.mark.unit
class TestNormalize:
    def test_hits_target_rms(self):
        buf = np.full(1000, 0.5, dtype=np.float32)
        out = dsp.normalize(buf, target_rms=0.16)
        assert float(np.sqrt(np.mean(out ** 2))) == pytest.approx(0.16, abs=1e-4)

    def test_leaves_silence_untouched(self):
        buf = np.zeros(1000, dtype=np.float32)
        out = dsp.normalize(buf, target_rms=0.16)
        assert np.all(out == 0.0)  # guard against divide-by-zero blowup
