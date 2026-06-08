"""Tests for the structured end-of-session report."""

import json

import pytest

from session_summary import build_summary, write_summary


@pytest.mark.unit
def test_summary_captures_core_fields():
    s = build_summary(
        duration_s=120.0,
        scenes=[{"t": 0.0, "a": "dark", "b": "warm", "key": "C major"}],
        perf={"gen_ms_per_chunk": 640.0, "realtime_ok": True,
              "starves": 0, "underruns": 1},
        health={"ok": True, "fault": None},
    )
    assert s["duration_s"] == 120.0
    assert s["scene_count"] == 1
    assert s["engine_ok"] is True
    assert s["fault"] is None
    assert s["generation"]["realtime_ok"] is True
    assert s["generation"]["underruns"] == 1
    assert s["scenes"][0]["key"] == "C major"


@pytest.mark.unit
def test_summary_is_defensive_about_missing_stats():
    s = build_summary(duration_s=5.0, scenes=[], perf=None, health=None)
    assert s["scene_count"] == 0
    assert s["engine_ok"] is True       # defensive default when health is absent
    assert s["generation"]["starves"] == 0
    assert s["generation"]["gen_ms_per_chunk"] is None


@pytest.mark.unit
def test_summary_reports_a_fault():
    s = build_summary(duration_s=1.0, scenes=[], perf=None,
                      health={"ok": False, "fault": "RuntimeError: boom"})
    assert s["engine_ok"] is False
    assert "boom" in s["fault"]


@pytest.mark.unit
def test_write_summary_roundtrips(tmp_path):
    s = build_summary(duration_s=1.0, scenes=[], perf=None, health=None)
    p = tmp_path / "summary.json"
    write_summary(s, str(p))
    assert json.loads(p.read_text()) == s
