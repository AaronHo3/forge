"""
Unit tests for Telemetry.health() — the engine-health → dashboard event path.

We don't start the HTTP server; we capture what would be emitted by replacing
_emit, then assert the JSON the dashboard would receive. This pins the field
mapping (which lives only in Telemetry.health) against MRTController's
health()/perf_stats() dict shapes so the two can't silently drift.
"""

import json
import time

import pytest

from telemetry import Telemetry


def _capture(tmp_path):
    """A Telemetry whose emitted lines are captured instead of served."""
    tel = Telemetry(log_path=str(tmp_path / "t.jsonl"))
    tel._t0 = time.monotonic()        # event() needs a start time; we skip start()
    sent: list[dict] = []
    tel._emit = lambda line: sent.append(json.loads(line))
    return tel, sent


@pytest.mark.unit
def test_health_none_is_noop(tmp_path):
    """MIDI mode (controller.health() -> None) must emit nothing."""
    tel, sent = _capture(tmp_path)
    tel.health(None)
    tel.health(None, {"gen_ms_per_chunk": 700})
    assert sent == []


@pytest.mark.unit
def test_health_ok_maps_fields(tmp_path):
    """A healthy snapshot maps perf + health dicts into one 'health' event."""
    tel, sent = _capture(tmp_path)
    tel.health({"ok": True, "fault": None, "starves": 0, "underruns": 1,
                "last_chunk_age_s": 0.8},
               {"gen_ms_per_chunk": 760.0, "realtime_ok": True})
    assert len(sent) == 1
    e = sent[0]
    assert e["kind"] == "health"
    assert e["ok"] is True and e["fault"] is None
    assert e["gen_ms"] == 760.0 and e["realtime_ok"] is True
    assert e["underruns"] == 1 and e["starves"] == 0
    assert e["last_chunk_age_s"] == 0.8


@pytest.mark.unit
def test_health_fault_surfaces(tmp_path):
    """A faulted engine carries ok=False + the fault string for the red badge."""
    tel, sent = _capture(tmp_path)
    tel.health({"ok": False, "fault": "RuntimeError: model weights missing",
                "starves": 3, "underruns": 1})
    e = sent[0]
    assert e["ok"] is False
    assert e["fault"] == "RuntimeError: model weights missing"
    # No perf dict given → gen_ms None, realtime defaults to True (not "slow").
    assert e["gen_ms"] is None and e["realtime_ok"] is True


@pytest.mark.unit
def test_health_counts_fall_back_to_perf(tmp_path):
    """If the health dict omits the counters, they're taken from perf — pins the
    `health.get(..., perf.get(...))` fallback in the field mapping."""
    tel, sent = _capture(tmp_path)
    tel.health({"ok": True},
               {"gen_ms_per_chunk": 800.0, "realtime_ok": True,
                "starves": 5, "underruns": 2})
    e = sent[0]
    assert e["starves"] == 5 and e["underruns"] == 2


@pytest.mark.unit
def test_health_slow_flag(tmp_path):
    """Behind-realtime generation flows through as realtime_ok False."""
    tel, sent = _capture(tmp_path)
    tel.health({"ok": True, "starves": 2, "underruns": 0},
               {"gen_ms_per_chunk": 1120.0, "realtime_ok": False})
    e = sent[0]
    assert e["ok"] is True and e["realtime_ok"] is False
    assert e["gen_ms"] == 1120.0 and e["starves"] == 2
