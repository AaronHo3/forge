"""Phase 3: a headless happy-path test of the live generation loop.

With an injected fake backend AND a fake output stream, `_run_audio_loop` runs
with no model and no audio device, so we can assert it boots, generates a chunk,
and reports healthy. This is the other half of the Phase 2 fault test (which only
covered the failure path), unlocked by the MRTBackend seam.
"""

import time

import numpy as np
import pytest

from mrt_controller import PythonMRTController


class _FakeWaveform:
    """Minimal stand-in for magenta_rt's Waveform — the loop only reads .samples."""
    def __init__(self, samples: np.ndarray):
        self.samples = samples


class FakeBackend:
    """Deterministic in-memory stand-in for MagentaRT2SystemMlxfn. No model, no
    GPU: embed_style returns a small vector, generate returns a short stereo sine
    chunk plus a monotonically advancing state."""
    _num_notes = 128

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate
        self.generate_calls = 0
        self.embed_calls = 0

    def embed_style(self, text_or_audio, **kwargs):
        self.embed_calls += 1
        # Deterministic across processes (unlike hash()); the value is not
        # asserted on, only its shape, so a length-derived float is plenty.
        v = (len(str(text_or_audio)) % 100) / 100.0
        return np.full(8, v, dtype=np.float32)

    def generate(self, *, style=None, frames=25, state=None, **kwargs):
        self.generate_calls += 1
        n = max(1, int(frames / 25 * self._sr))
        t = np.arange(n) / self._sr
        sig = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return _FakeWaveform(np.stack([sig, sig], axis=1)), (state or 0) + 1


class _FakeStream:
    """No-op output stream: the callback is never driven, so no audio plays and
    the loop just seeds and paces chunks into its buffer."""
    def __init__(self, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


@pytest.mark.unit
class TestHeadlessLoop:
    def test_loop_boots_and_generates_a_chunk(self):
        backend = FakeBackend()
        c = PythonMRTController(
            backend_factory=lambda: backend,
            output_factory=lambda **kw: _FakeStream(**kw),
        )
        c.start()
        # Wait for the loop to load the (fake) backend and seed the first chunk.
        deadline = time.monotonic() + 5.0
        while c.health()["last_chunk_age_s"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
        h = c.health()
        c.stop()

        assert h["ok"] is True, f"loop faulted: {h['fault']}"
        assert h["last_chunk_age_s"] is not None, "no chunk generated headless"
        assert backend.generate_calls >= 1

    def test_scene_change_re_embeds_poles_headless(self):
        backend = FakeBackend()
        c = PythonMRTController(
            backend_factory=lambda: backend,
            output_factory=lambda **kw: _FakeStream(**kw),
        )
        c.start()
        deadline = time.monotonic() + 5.0
        while c.health()["last_chunk_age_s"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
        embeds_before = backend.embed_calls
        c.set_prompts("sad cello", "muted piano", "A minor")
        # The scene change embeds the two new poles on the audio thread.
        deadline = time.monotonic() + 5.0
        while backend.embed_calls < embeds_before + 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        c.stop()
        assert c.health()["ok"] is True
        assert backend.embed_calls >= embeds_before + 2, "scene change did not re-embed"
