"""Integration-ish tests for VoiceAnalyzer._extract on synthetic buffers.

_extract is deterministic given a fixed numpy block (it runs librosa, not a
mic), so we can feed it silence and a tone and assert the feature contract
without any audio hardware.
"""

import threading

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


@pytest.mark.unit
class TestRawAudioBuffer:
    """The rolling raw-audio buffer feeds Whisper. The mic thread appends while
    the LLM thread snapshots, so it must be concurrency-safe."""

    def test_returns_last_n_seconds_concatenated(self):
        va = VoiceAnalyzer()
        for i in range(20):
            va._append_raw(np.full(BLOCK_SIZE, float(i), dtype=np.float32))
        out = va.get_audio_for_transcription(0.5)   # 0.5s = the last 5 blocks
        assert out.shape[0] == 5 * BLOCK_SIZE
        assert out[0] == pytest.approx(15.0)         # blocks 15..19 retained

    def test_empty_buffer_returns_empty(self):
        assert VoiceAnalyzer().get_audio_for_transcription(4.0).shape[0] == 0

    def test_bounded_to_maxlen(self):
        va = VoiceAnalyzer()
        for _ in range(200):
            va._append_raw(np.zeros(BLOCK_SIZE, dtype=np.float32))
        assert len(va._raw_blocks) <= 50

    def test_concurrent_append_and_snapshot_is_safe(self):
        # Regression: list(deque) during a concurrent append used to raise
        # "deque mutated during iteration" before _raw_lock guarded both sides.
        va = VoiceAnalyzer()
        blk = np.zeros(BLOCK_SIZE, dtype=np.float32)
        errors: list[Exception] = []
        stop = threading.Event()

        def writer():
            try:
                while not stop.is_set():
                    va._append_raw(blk)
            except Exception as e:  # pragma: no cover - only on a real race
                errors.append(e)

        t = threading.Thread(target=writer)
        t.start()
        try:
            # The race is in `list(self._raw_blocks)`, which runs in full
            # regardless of the window, so a tiny 0.5s window keeps each snapshot
            # cheap while still exercising the contended path many times.
            for _ in range(2000):
                va.get_audio_for_transcription(0.5)
        except Exception as e:  # pragma: no cover - only on a real race
            errors.append(e)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert not errors, f"concurrent access raised: {errors}"
