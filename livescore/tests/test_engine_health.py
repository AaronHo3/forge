"""Phase 2 resilience tests: the engine must fail LOUDLY, not die silently.

The whole point: before this, if the generation thread died (model load failed,
audio device missing, a generation error), `_cur_stream` stayed empty and the app
played silence forever while still looking 'running'. Now any such failure sets a
fault that health() reports.

These inject a backend factory so no real model loads and no audio device opens.
The failing case never reaches the audio device — it dies at the (injected)
backend creation, exactly where a real model-load failure would.
"""

import pytest

from mrt_controller import PythonMRTController


@pytest.mark.unit
class TestEngineHealth:
    def test_health_shape_before_start(self):
        c = PythonMRTController()
        h = c.health()
        assert set(h) >= {"ok", "fault", "last_chunk_age_s", "starves", "underruns"}
        assert h["ok"] is True
        assert h["fault"] is None
        assert h["last_chunk_age_s"] is None  # no chunk produced yet

    def test_backend_failure_is_surfaced_not_silent(self):
        def exploding_backend():
            raise RuntimeError("model weights missing")

        c = PythonMRTController(backend_factory=exploding_backend)
        assert c.health()["ok"] is True  # healthy before the thread runs

        c.start()
        # Block until the supervisor signals a fault — deterministic, no polling.
        faulted = c._fault_event.wait(timeout=5.0)
        h = c.health()
        c.stop()

        assert faulted, "engine fault was not signalled within timeout"
        assert h["ok"] is False, "engine fault was not surfaced via health()"
        assert "model weights missing" in h["fault"]
        assert "RuntimeError" in h["fault"]
