"""
harmony.py — shared musical-key helpers.

Note-name parsing and key -> MRT2 note-conditioning masks, used by BOTH the live
controller (mrt_controller.py) and the offline keepsake renderer (keepsake.py).
Keeping one copy here means the live take and its rendered keepsake can never
drift out of tune with each other.
"""

from __future__ import annotations

import re

# Pitch classes and the two scale shapes the project supports.
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_MAJOR = (0, 2, 4, 5, 7, 9, 11)

# Pitches in this inclusive MIDI range get the soft key constraint; the rest are
# left free. 36-84 is a musical mid-range.
NOTE_LO = 36
NOTE_HI = 84

# A note name: letter A-G, optional accidental (# sharp, b flat), then a
# possibly-multi-digit, possibly-negative octave. \A and \Z anchor a full match.
_NOTE_RE = re.compile(r"\A\s*([A-Ga-g])([#b]?)(-?\d+)\s*\Z")


def note_to_midi(name: str | None) -> int:
    """Convert a note name like 'C3', 'F#4', or 'Db3' to its MIDI number
    (C4 = 60, A4 = 69). Sharps use '#', flats use lowercase 'b'. Octaves may be
    multi-digit or negative. Raises ValueError on an unrecognised name or a note
    outside the valid MIDI range (0-127)."""
    m = _NOTE_RE.match(name or "")
    if not m:
        raise ValueError(f"unrecognised note name: {name!r}")
    pitch, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    semitone = _PC[pitch]   # single source of truth for pitch classes
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = (octave + 1) * 12 + semitone
    if not 0 <= midi <= 127:
        raise ValueError(f"note {name!r} is outside the MIDI range (0-127)")
    return midi


def build_notes(key: str | None, num_notes: int) -> list[int] | None:
    """Build a `num_notes`-long note-conditioning mask for a musical key, so MRT2
    plays in tune. Soft constraint: in-key pitches stay MASKED (-1, free to play,
    giving movement/arpeggios) and only out-of-key pitches in NOTE_LO..NOTE_HI are
    turned OFF (0). Never forces a pitch on (that would hold a static chord).
    Returns None on any problem (caller falls back to no harmony)."""
    if not key:
        return None
    try:
        k = key.strip()
        pc = _PC[k[0].upper()]
        if len(k) > 1 and k[1] in '#b':
            pc += 1 if k[1] == '#' else -1
        pc %= 12
        # A bare key name (e.g. "C") or anything without "maj" defaults to minor.
        scale = {(pc + iv) % 12 for iv in (_MAJOR if 'maj' in k.lower() else _MINOR)}
        notes = [-1] * num_notes
        for midi in range(min(128, num_notes)):
            if NOTE_LO <= midi <= NOTE_HI and (midi % 12) not in scale:
                notes[midi] = 0
        return notes
    except Exception:
        return None
