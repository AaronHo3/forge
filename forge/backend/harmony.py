"""
harmony.py - music theory for the Harmony Sandbox.

The education thesis made concrete: a learner picks a chord (or a whole
progression) and MRT2 arranges a live band around it in real time. They do the
harmonic thinking; the model does the playing they could never execute. This
module is the theory layer:

  - chord_notes()      build the 128-int note-conditioning vector MRT2 wants
                       (chord tones = 1 "play", others = 0 "off")
  - diatonic_chords()  the seven chords of a key, with roman numerals + function
  - PROGRESSIONS       famous progressions as scale degrees, key-agnostic

The note-conditioning contract (per MRT2's generate(notes=...)):
  -1 = masked  (model free to choose)
   0 = off     (do not play this pitch)
   1 = on      (play / emphasize this pitch)
"""

from __future__ import annotations

# Pitch classes 0..11 starting at C.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Chord quality -> intervals from the root (semitones).
QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10),
}
SUFFIX = {"maj": "", "min": "m", "dim": "dim", "aug": "+", "maj7": "maj7",
          "min7": "m7", "7": "7", "sus2": "sus2", "sus4": "sus4",
          "dim7": "dim7", "m7b5": "m7b5"}

# MRT2 voices in this MIDI window; outside it we leave pitches free (bass/air).
NOTE_LO, NOTE_HI = 36, 84

# Diatonic templates (triads) for major and natural-minor keys.
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
_MAJOR_QUALS = ("maj", "min", "min", "maj", "maj", "min", "dim")
_MINOR_QUALS = ("min", "dim", "maj", "min", "min", "maj", "maj")
# Diatonic seventh chords (stacking thirds within the scale). Note V is a DOMINANT
# 7th in major, which is the whole reason the V7 -> I pull is so strong.
_MAJOR_SEVENTHS = ("maj7", "min7", "min7", "maj7", "7", "min7", "m7b5")
_MINOR_SEVENTHS = ("min7", "m7b5", "maj7", "min7", "min7", "maj7", "7")
_MAJOR_ROMANS = ("I", "ii", "iii", "IV", "V", "vi", "vii°")
_MINOR_ROMANS = ("i", "ii°", "III", "iv", "v", "VI", "VII")
_MAJOR_FUNC = ("Tonic", "Subdominant", "Tonic", "Subdominant", "Dominant", "Tonic", "Dominant")
_MINOR_FUNC = ("Tonic", "Subdominant", "Tonic", "Subdominant", "Dominant", "Subdominant", "Subtonic")

FUNCTION_HINT = {
    "Tonic": "Home. Feels resolved and at rest.",
    "Subdominant": "Moving away from home, building gentle motion.",
    "Dominant": "Maximum pull. It wants to resolve back home.",
    "Subtonic": "A colorful step just below home.",
}

# Famous progressions as 1-indexed scale degrees (work in any key/mode).
PROGRESSIONS = {
    "major": [
        {"name": "Pop  I V vi IV", "degrees": [1, 5, 6, 4]},
        {"name": "50s  I vi IV V", "degrees": [1, 6, 4, 5]},
        {"name": "Pachelbel  I V vi iii", "degrees": [1, 5, 6, 3]},
        {"name": "Jazz  ii V I", "degrees": [2, 5, 1]},
    ],
    "minor": [
        {"name": "Andalusian  i VII VI V", "degrees": [1, 7, 6, 5]},
        {"name": "Minor pop  i VI III VII", "degrees": [1, 6, 3, 7]},
        {"name": "Minor ii° v i", "degrees": [2, 5, 1]},
    ],
}


def root_pc(name: str) -> int:
    """'C', 'F#', 'Bb' -> pitch class 0..11."""
    name = name.strip()
    pc = PC[name[0].upper()]
    if len(name) > 1 and name[1] in "#b":
        pc += 1 if name[1] == "#" else -1
    return pc % 12


def chord_name(root: int, quality: str) -> str:
    return NOTE_NAMES[root % 12] + SUFFIX.get(quality, quality)


def chord_notes(root: int, quality: str, num_notes: int, strict: bool = True) -> list[int]:
    """Build the note-conditioning vector for one chord.

    strict=True  -> non-chord pitches are turned OFF (0): unambiguous harmony, the
                    clearest teaching signal.
    strict=False -> non-chord pitches stay masked (-1): the model may add color.
    Pitches outside the voicing window are always free (-1) so bass and air breathe.
    """
    ivs = QUALITIES.get(quality, QUALITIES["maj"])
    pcs = {(root + iv) % 12 for iv in ivs}
    notes = [-1] * num_notes
    for midi in range(num_notes):
        if NOTE_LO <= midi <= NOTE_HI:
            notes[midi] = 1 if (midi % 12) in pcs else (0 if strict else -1)
        else:
            notes[midi] = -1
    return notes


def diatonic_chords(key_root: int, mode: str = "major") -> list[dict]:
    """The seven chords of a key, each with roman numeral, name, and function."""
    if mode == "minor":
        scale, quals, sevs = _MINOR_SCALE, _MINOR_QUALS, _MINOR_SEVENTHS
        romans, funcs = _MINOR_ROMANS, _MINOR_FUNC
    else:
        scale, quals, sevs = _MAJOR_SCALE, _MAJOR_QUALS, _MAJOR_SEVENTHS
        romans, funcs = _MAJOR_ROMANS, _MAJOR_FUNC
    out = []
    for i in range(7):
        root = (key_root + scale[i]) % 12
        q, sev = quals[i], sevs[i]
        out.append({
            "degree": i + 1, "roman": romans[i], "root": root, "quality": q,
            "name": chord_name(root, q), "function": funcs[i],
            "hint": FUNCTION_HINT.get(funcs[i], ""),
            "seventh": sev, "seventh_name": chord_name(root, sev),
        })
    return out


def key_payload(root_name: str, mode: str) -> dict:
    """Everything the sandbox UI needs to render a key: chords + progressions."""
    mode = "minor" if mode == "minor" else "major"
    kr = root_pc(root_name)
    return {
        "root": root_name, "root_pc": kr, "mode": mode,
        "key_label": f"{NOTE_NAMES[kr]} {mode}",
        "chords": diatonic_chords(kr, mode),
        "progressions": PROGRESSIONS[mode],
    }
