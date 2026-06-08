"""
session_summary.py — a structured end-of-session report.

At the end of a live telling we have everything needed to describe how it went:
how long, how many scenes, whether generation kept up in real time, and whether
the engine faulted. build_summary() assembles that into one JSON-able dict (pure,
so it is testable); write_summary() persists it next to the telemetry log for
offline analysis.
"""

from __future__ import annotations

import json
from typing import Any


def build_summary(*, duration_s: float, scenes: list[dict],
                  perf: dict | None, health: dict | None) -> dict[str, Any]:
    """Assemble a session report from the controller's perf/health stats and the
    captured scene timeline. Inputs are defensive (perf/health may be None), so a
    partial or failed session still produces a valid report."""
    perf = perf or {}
    health = health or {}
    return {
        "duration_s": round(float(duration_s), 1),
        "scene_count": len(scenes),
        "engine_ok": health.get("ok", True),
        "fault": health.get("fault"),
        "generation": {
            "gen_ms_per_chunk": perf.get("gen_ms_per_chunk"),
            "realtime_ok": perf.get("realtime_ok"),
            "starves": perf.get("starves", 0),
            "underruns": perf.get("underruns", 0),
        },
        "scenes": [
            {"t": s.get("t"), "a": s.get("a"), "b": s.get("b"), "key": s.get("key")}
            for s in scenes
        ],
    }


def write_summary(summary: dict, path: str) -> str:
    """Write the summary to `path` as pretty JSON. Returns the path."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
