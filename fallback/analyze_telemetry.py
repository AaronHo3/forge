"""
analyze_telemetry.py
--------------------
Read a telemetry JSONL log and report the real range each feature reached.

Use it after a voice session to see, with actual data, whether your voice
drove the parameters across their full range — or sat in a narrow band that
needs the mapping widened.

    python3 analyze_telemetry.py                       # newest telemetry-*.jsonl
    python3 analyze_telemetry.py telemetry-20260605.jsonl
"""

import glob
import json
import os
import sys

import paths


FIELDS = ["energy", "pitch", "rate", "bright", "blend", "chaos"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _newest_log()
    if not path:
        print("No telemetry-*.jsonl files found. Run main.py first.")
        return

    rows = _load(path)
    if not rows:
        print(f"{path} is empty.")
        return

    voiced = [r for r in rows if not r.get("silent")]
    print(f"\nFile: {path}")
    print(f"Samples: {len(rows)}  ({len(voiced)} voiced, "
          f"{len(rows) - len(voiced)} silent)\n")

    print(f"{'feature':8s} {'min':>8s} {'max':>8s} {'mean':>8s} "
          f"{'spread':>8s}   coverage (voiced only)")
    print("─" * 64)
    for f in FIELDS:
        vals = [r[f] for r in voiced if f in r]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        span = hi - lo
        bar = _coverage_bar(lo, hi) if f != "pitch" else ""
        print(f"{f:8s} {lo:8.3f} {hi:8.3f} {mean:8.3f} {span:8.3f}   {bar}")

    print()
    _verdict(voiced)


def _verdict(voiced):
    """Flag features that never used much of their 0–1 range."""
    notes = []
    for f in ("blend", "chaos"):
        vals = [r[f] for r in voiced if f in r]
        if not vals:
            continue
        span = max(vals) - min(vals)
        if span < 0.4:
            notes.append(f"⚠ {f} only swung {span:.2f} of its 0–1 range — "
                         f"the music barely moved. Widen the mapping.")
        else:
            notes.append(f"✓ {f} swung {span:.2f} — good expressive range.")
    for n in notes:
        print("  " + n)
    print()


def _coverage_bar(lo, hi, width=20):
    """ASCII bar showing which slice of 0–1 the feature actually occupied."""
    lo_i = max(0, min(width, round(lo * width)))
    hi_i = max(0, min(width, round(hi * width)))
    return "░" * lo_i + "█" * max(1, hi_i - lo_i) + "░" * (width - hi_i)


def _load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _newest_log():
    files = sorted(glob.glob(os.path.join(paths.TELEMETRY, "telemetry-*.jsonl")))
    return files[-1] if files else None


if __name__ == "__main__":
    main()
