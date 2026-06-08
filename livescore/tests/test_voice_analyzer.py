"""Integration-ish tests for VoiceAnalyzer._extract on synthetic buffers.

_extract is deterministic given a fixed numpy block (it runs librosa, not a
mic), so we can feed it silence and a tone and assert the feature contract
without any audio hardware.
"""

import numpy as np
import pytest

from voice_analyzer import VoiceAnalyzer, SAMPLE_RATE, BLOCK_SIZE


@pytest.fixture(scope="module")
def analyzer():
    # __init__ opens no stream; start() would. _extract is callable directly.
    return VoiceAnalyzer()


@pytest.mark.unit
def test_silence_is_detected(analyzer):
    block = np.zeros(BLOCK_SIZE, dtype=np.float32)
    f = analyzer._extract(block)
    assert f.is_silent is True
    assert f.energy == pytest.approx(0.0, abs=1e-6)
    # The silent branch returns before pitch/rate/brightness are computed.
    assert f.pitch == 0.0


@pytest.mark.unit
def test_voiced_tone_is_not_silent_and_in_range(analyzer):
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    block = (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    f = analyzer._extract(block)
    assert f.is_silent is False
    assert 0.0 <= f.energy <= 1.0
    assert 0.0 <= f.brightness <= 1.0
    assert 0.0 <= f.speech_rate <= 1.0


@pytest.mark.unit
def test_louder_input_gives_more_energy(analyzer):
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    quiet = (0.05 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    loud = (0.25 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    assert analyzer._extract(loud).energy > analyzer._extract(quiet).energy
