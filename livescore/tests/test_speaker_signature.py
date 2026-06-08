"""Unit tests for speaker_signature: the deterministic voice-identity layer.

The core promise is "same voice -> same song identity, different voice ->
different identity." These pin determinism of the projection and the signature,
and the graceful empty-input behavior.
"""

import numpy as np
import pytest

from speaker_signature import (
    SpeakerSignature, _proj_index, PALETTES, KEYS,
)


@pytest.mark.unit
class TestProjIndex:
    def test_is_deterministic_for_same_seed(self):
        vec = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
        assert _proj_index(vec, 6, seed=13) == _proj_index(vec, 6, seed=13)

    def test_index_in_range(self):
        vec = np.array([4.2, -7.1, 0.3, 9.9, -3.3])
        for n in (1, 6, 12):
            assert 0 <= _proj_index(vec, n, seed=7) < n


@pytest.mark.unit
class TestSpeakerSignature:
    def test_not_ready_before_min_samples(self):
        sig = SpeakerSignature()
        assert sig.ready is False
        assert sig.windows == 0

    def test_empty_signature_is_deterministic_and_well_formed(self):
        # With no audio the profile is all-zeros, which must still yield a valid,
        # stable signature (not crash, not random).
        sig = SpeakerSignature()
        s = sig.signature()
        assert s["palette"] == PALETTES[0]
        assert s["key"] == KEYS[0]
        assert s["tempo"] == "gently flowing"
        assert s["brightness"] == pytest.approx(0.5, abs=1e-6)

    def test_signature_is_cached(self):
        sig = SpeakerSignature()
        assert sig.signature() is sig.signature()  # frozen, same object

    def test_same_audio_gives_same_signature(self):
        t = np.arange(int(0.5 * 48_000)) / 48_000
        tone = (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        a, b = SpeakerSignature(), SpeakerSignature()
        for _ in range(SpeakerSignature.MIN_SAMPLES):
            a.add_audio(tone)
            b.add_audio(tone)
        assert a.ready and b.ready
        assert a.signature() == b.signature()

    def test_describe_contains_key(self):
        sig = SpeakerSignature()
        assert sig.signature()["key"] in sig.describe()
