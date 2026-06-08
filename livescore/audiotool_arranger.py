"""
audiotool_arranger.py
---------------------
Turns a `keepsake-*.json` (the captured telling) into an `arrangement-*.json`:
an editable, symbolic score — diatonic chords, a bass line, and a sparkle
arpeggio per scene, laid out on a beat timeline.

This is the Python half of the Audiotool integration. It owns ALL the musical
decisions (key, chords, octaves, voicing) and emits plain note events in beats.
The Node sidecar in `audiotool_export/` reads this JSON and writes a real,
editable Audiotool project via the Nexus SDK — no music theory on that side.

Why split here? Nexus *writes* are JS/TS only, but every creative choice in this
project already lives in Python. The arrangement JSON is the entire contract
between the two, and it's fully testable with no network and no Audiotool:

    python3 audiotool_arranger.py keepsake-20260606-080622.json
    # → arrangement-20260606-080622.json   (then: cd audiotool_export && npm start)

The live pass is the sketch; the keepsake WAV is the recording; THIS is the
score the narrator can reopen, remix, and share.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime

import paths

# ── Musical constants (shared vocabulary with keepsake.py / mrt_controller.py) ──
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_MAJOR = (0, 2, 4, 5, 7, 9, 11)

# Diatonic chord roads, as scale-degree indices (0-based) into the 7-note scale.
# Minor: i – VI – III – VII (the cinematic loop). Major: I – V – vi – IV (the pop loop).
_PROGRESSION = {
    "minor": (0, 5, 2, 6),
    "major": (0, 4, 5, 3),
}

# Timeline shape. 90 BPM in 4/4 → one bar = 2.667 s, a calm storytelling pulse.
TEMPO_BPM = 90
BEATS_PER_BAR = 4
SEC_PER_BAR = BEATS_PER_BAR * 60.0 / TEMPO_BPM
MIN_BARS = 2
MAX_BARS = 8

# Register anchors (MIDI) for each role. Kept low/mid/high so they never collide.
_BASS_BASE = 36       # C2 region — foundation
_HARMONY_BASE = 48    # C3 region — chords / pad
_TEXTURE_BASE = 72    # C5 region — sparkle arpeggio
_MELODY_BASE = 60     # C4 region — sung lead (Claude-written, snapped to key)

# Instrument per role. `gakki` is Audiotool's SOUNDFONT PLAYER — a sampler that
# plays REAL recorded instruments (cello, bass, bells, piano), so the score
# actually sounds like its labels instead of a synth. The exporter loads a
# matching soundfont preset onto each gakki; if none is found it falls back to an
# audible synth (see FALLBACK_SYNTH in the Node side) rather than going silent.
_TRACK_PLAN = (
    ("foundation", "gakki", "Bass",     1),
    ("harmony",    "gakki", "Harmony",  3),
    ("texture",    "gakki", "Texture",  6),
)
# Appended only when a melody source (Claude) is active.
_MELODY_TRACK = ("melody", "gakki", "Melody", 4)

MELODY_MODEL = "claude-haiku-4-5-20251001"

# Mood lexicon — reused intent from keepsake.py, used here only to nudge velocity
# and arpeggio density (the key already decides major/minor).
_DARK_WORDS = {"dark", "tense", "ominous", "low", "dread", "uneasy", "grief",
               "sad", "eerie", "war", "pounding", "urgent", "creeping", "sparse",
               "cold", "menacing", "fear", "anxious", "tremolo"}
_BRIGHT_WORDS = {"warm", "bright", "triumphant", "hopeful", "gentle", "peaceful",
                 "soft", "uplifting", "serene", "joyful", "sunlit", "golden",
                 "calm", "tender", "resolute", "shimmer", "fingerpicked"}


# ── Note / region / track value objects (immutable; serialize straight to JSON) ─
@dataclass(frozen=True)
class Note:
    positionBeats: float
    pitch: int
    durationBeats: float
    velocity: float


@dataclass(frozen=True)
class Region:
    trackId: str
    scene: int
    key: str
    displayName: str
    startBeat: float
    durationBeats: float
    colorIndex: int
    notes: list[Note] = field(default_factory=list)


@dataclass(frozen=True)
class Track:
    id: str
    synth: str
    displayName: str
    colorIndex: int
    presetQuery: str = ""   # free-text search into Audiotool's preset library


# ── Key / scale helpers ─────────────────────────────────────────────────────────
def parse_key(key: str | None) -> tuple[int, bool]:
    """'A minor' → (pitch_class=9, is_minor=True). Defaults to A minor."""
    if not key:
        return 9, True
    k = key.strip()
    letter = k[:1].upper()
    if letter not in _PC:
        return 9, True
    pc = _PC[letter]
    if len(k) > 1 and k[1] in "#b":
        pc += 1 if k[1] == "#" else -1
    is_minor = "min" in k.lower()
    return pc % 12, is_minor


def scale_midi(root_pc: int, is_minor: bool, base_midi: int) -> list[int]:
    """Two octaves of the diatonic scale, ascending, anchored at/above base_midi.
    Indexable for triad building (degree d → notes at d, d+2, d+4)."""
    intervals = _MINOR if is_minor else _MAJOR
    root = base_midi + ((root_pc - base_midi) % 12)   # first root at/above base
    return [root + iv + 12 * octv for octv in range(3) for iv in intervals]


def triad(scale: list[int], degree: int) -> list[int]:
    """Diatonic triad stacked on a scale degree (root, third, fifth)."""
    return [scale[degree], scale[degree + 2], scale[degree + 4]]


def scale_pitch_classes(root_pc: int, is_minor: bool) -> set[int]:
    intervals = _MINOR if is_minor else _MAJOR
    return {(root_pc + iv) % 12 for iv in intervals}


def snap_to_scale(pitch: int, scale_pcs: set[int]) -> int:
    """Move a pitch to the nearest member of the scale (ties → down).
    Guarantees Claude's melody is always in-key, no matter what it returns."""
    for delta in (0, -1, 1, -2, 2):
        if (pitch + delta) % 12 in scale_pcs:
            return pitch + delta
    return pitch  # unreachable for a 7-note scale, but safe


def mood_of(scene: dict) -> str:
    words = set(re.findall(r"[a-z]+", f"{scene.get('a','')} {scene.get('b','')}".lower()))
    return "dark" if len(words & _DARK_WORDS) >= len(words & _BRIGHT_WORDS) else "bright"


# Instrument words we look for in the scene labels, grouped by the role they best
# fill. The arranger counts these across the whole telling and picks each track's
# preset search term from the story's own vocabulary — so a cello-heavy tale gets
# cello-ish patches, a piano tale gets piano patches.
_HARMONY_INSTRUMENTS = ("piano", "rhodes", "guitar", "strings", "cello", "organ",
                        "pad", "harp", "keys")
_MELODY_INSTRUMENTS = ("strings", "violin", "cello", "flute", "guitar", "piano",
                       "synth", "harp")
_TEXTURE_INSTRUMENTS = ("bells", "glockenspiel", "marimba", "celesta", "kalimba",
                        "harp", "vibraphone", "music box")


def _top_instrument(scenes: list[dict], vocabulary: tuple[str, ...], default: str) -> str:
    """Most-mentioned instrument from `vocabulary` across all scene labels."""
    blob = " ".join(f"{s.get('a','')} {s.get('b','')}" for s in scenes).lower()
    counts = {w: blob.count(w) for w in vocabulary}
    best = max(counts, key=lambda w: counts[w])
    return best if counts[best] > 0 else default


def _preset_queries(scenes: list[dict]) -> dict[str, str]:
    """One preset search term per role, biased by the story's instruments."""
    return {
        "foundation": "bass",
        "harmony": _top_instrument(scenes, _HARMONY_INSTRUMENTS, "warm pad"),
        "texture": _top_instrument(scenes, _TEXTURE_INSTRUMENTS, "bells"),
        "melody": _top_instrument(scenes, _MELODY_INSTRUMENTS, "strings"),
    }


# ── Per-scene voicing ───────────────────────────────────────────────────────────
def _scene_bars(duration_s: float) -> int:
    return int(min(MAX_BARS, max(MIN_BARS, round(duration_s / SEC_PER_BAR))))


def _bass_notes(prog_scale, progression, bars, vel) -> list[Note]:
    """One sustained root per bar, walking the chord progression."""
    notes = []
    for bar in range(bars):
        root = prog_scale[progression[bar % len(progression)]]
        notes.append(Note(positionBeats=bar * BEATS_PER_BAR, pitch=root,
                          durationBeats=BEATS_PER_BAR, velocity=vel))
    return notes


def _harmony_notes(prog_scale, progression, bars, vel) -> list[Note]:
    """A sustained diatonic triad per bar."""
    notes = []
    for bar in range(bars):
        for pitch in triad(prog_scale, progression[bar % len(progression)]):
            notes.append(Note(positionBeats=bar * BEATS_PER_BAR, pitch=pitch,
                              durationBeats=BEATS_PER_BAR, velocity=vel))
    return notes


def _texture_notes(prog_scale, progression, bars, vel, dense: bool) -> list[Note]:
    """A light ascending arpeggio of the bar's triad. Bright/dense scenes get
    eighth notes (8/bar); dark/sparse scenes get a calmer quarter pattern."""
    notes = []
    step = 0.5 if dense else 1.0
    per_bar = int(BEATS_PER_BAR / step)
    for bar in range(bars):
        tones = triad(prog_scale, progression[bar % len(progression)])
        pattern = tones + [tones[1]] if not dense else tones + [tones[1], tones[2], tones[1]]
        for i in range(per_bar):
            pitch = pattern[i % len(pattern)] + 12   # an octave up = sparkle
            notes.append(Note(positionBeats=bar * BEATS_PER_BAR + i * step,
                              pitch=pitch, durationBeats=step * 0.9, velocity=vel))
    return notes


# ── Claude-written lead melody (optional, offline, snapped to key) ───────────────
_MELODY_SYSTEM = """\
You are a melodic composer writing a single-line lead melody for one scene of a
film score. You are given the key, the number of 4/4 bars, and the scene's mood.

Write an expressive, singable melody — phrases with shape, not every beat filled.
Leave rests for breath. Favor stepwise motion with occasional leaps. Stay mostly
diatonic to the key (out-of-key notes will be snapped in, so don't rely on them).

Output ONLY a JSON list of notes. Each note:
  {"b": <beat position, float, 0..bars*4>, "p": <MIDI pitch int, 60..84>, "d": <duration in beats, float>}
No prose, no markdown. Aim for roughly 1.5–3 notes per bar.

Example (2 bars, A minor):
[{"b":0,"p":69,"d":1},{"b":1,"p":72,"d":0.5},{"b":1.5,"p":71,"d":0.5},{"b":2,"p":69,"d":2},{"b":5,"p":67,"d":1}]
"""


def _validate_melody(raw_notes, bars, scale_pcs, vel) -> list[Note]:
    """Clamp + snap whatever Claude returned into safe, in-key Note objects."""
    span = bars * BEATS_PER_BAR
    out: list[Note] = []
    for item in raw_notes:
        try:
            b = float(item["b"]); p = int(item["p"]); d = float(item["d"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 <= b < span):
            continue
        p = snap_to_scale(max(55, min(86, p)), scale_pcs)
        d = max(0.25, min(float(BEATS_PER_BAR), d))
        out.append(Note(positionBeats=round(b, 3), pitch=p,
                        durationBeats=round(d, 3), velocity=vel))
    return out


def make_claude_melody_fn(palette: str):
    """Return melody_fn(scene, root_pc, is_minor, bars, bright) → [Note] | None,
    backed by Claude. Returns None (caller falls back) if anthropic is unavailable."""
    try:
        import anthropic
    except ImportError:
        print("[melody] anthropic not installed — skipping Claude melodies")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[melody] no ANTHROPIC_API_KEY — skipping Claude melodies")
        return None
    client = anthropic.Anthropic()

    def melody_fn(scene, root_pc, is_minor, bars, bright):
        scale_pcs = scale_pitch_classes(root_pc, is_minor)
        key_name = scene.get("key") or ("minor" if is_minor else "major")
        mood = scene.get("b") if bright else scene.get("a")
        user = (f"Key: {key_name}\nBars: {bars} (4/4)\n"
                f"Scene mood: {mood}\nSpeaker palette (gentle lean): {palette}")
        try:
            resp = client.messages.create(
                model=MELODY_MODEL, max_tokens=400,
                system=[{"type": "text", "text": _MELODY_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user},
                          {"role": "assistant", "content": "["}],
            )
            raw = "[" + resp.content[0].text
            raw = raw[:raw.rfind("]") + 1]
            notes = _validate_melody(json.loads(raw), bars, scale_pcs,
                                     0.7 if bright else 0.6)
            return notes or None
        except Exception as e:
            print(f"[melody] scene melody failed ({e}) — leaving it instrumental")
            return None

    return melody_fn


# ── Top-level: keepsake → arrangement ────────────────────────────────────────────
def build_arrangement(session: dict, melody_fn=None) -> dict:
    """Pure transform: a loaded keepsake dict → an arrangement dict.

    `melody_fn(scene, root_pc, is_minor, bars, bright) -> [Note] | None` is an
    optional lead-melody source (e.g. Claude). When provided, a 4th 'melody'
    track is added; scenes where it returns None simply get an empty region.
    """
    scenes = session.get("scenes") or []
    if not scenes:
        raise ValueError("keepsake has no scenes to arrange")
    home_key = (session.get("signature") or {}).get("key", "")

    plan = list(_TRACK_PLAN) + ([_MELODY_TRACK] if melody_fn else [])
    queries = _preset_queries(scenes)
    tracks = [Track(*p, presetQuery=queries.get(p[0], "")) for p in plan]

    # Per-scene durations come from the gap to the next scene (last scene gets a
    # default), exactly as the WAV renderer derives them.
    durations = []
    for i, sc in enumerate(scenes):
        nxt = scenes[i + 1]["t"] if i + 1 < len(scenes) else sc["t"] + MAX_BARS * SEC_PER_BAR
        durations.append(max(0.0, float(nxt) - float(sc["t"])))

    regions: list[Region] = []
    cursor_beat = 0.0
    for idx, (scene, dur_s) in enumerate(zip(scenes, durations), start=1):
        bars = _scene_bars(dur_s)
        span_beats = bars * BEATS_PER_BAR
        key = scene.get("key") or home_key
        root_pc, is_minor = parse_key(key)
        progression = _PROGRESSION["minor" if is_minor else "major"]
        bright = mood_of(scene) == "bright"
        base_vel = 0.78 if bright else 0.62
        color = (idx - 1) % 8

        role_notes = {
            "foundation": _bass_notes(
                scale_midi(root_pc, is_minor, _BASS_BASE), progression, bars, base_vel),
            "harmony": _harmony_notes(
                scale_midi(root_pc, is_minor, _HARMONY_BASE), progression, bars, base_vel - 0.08),
            "texture": _texture_notes(
                scale_midi(root_pc, is_minor, _TEXTURE_BASE), progression, bars,
                base_vel - 0.18, dense=bright),
        }
        if melody_fn is not None:
            role_notes["melody"] = melody_fn(scene, root_pc, is_minor, bars, bright) or []

        label = scene.get("b") if bright else scene.get("a")
        for track in tracks:
            regions.append(Region(
                trackId=track.id, scene=idx, key=key or "(home)",
                displayName=f"S{idx} · {label or 'scene'}"[:40],
                startBeat=cursor_beat, durationBeats=span_beats, colorIndex=color,
                notes=role_notes.get(track.id, [])))
        cursor_beat += span_beats

    return {
        "source": session.get("_source", "keepsake.json"),
        "title": f"Score the Story — {session.get('created', datetime.now().isoformat())[:16]}",
        "tempoBpm": TEMPO_BPM,
        "beatsPerBar": BEATS_PER_BAR,
        "palette": (session.get("signature") or {}).get("palette", ""),
        "tracks": [asdict(t) for t in tracks],
        "regions": [asdict(r) for r in regions],
    }


def arrange_file(keepsake_path: str, out_path: str | None = None,
                 use_claude: bool = True) -> str:
    with open(keepsake_path) as f:
        session = json.load(f)
    session["_source"] = keepsake_path

    melody_fn = None
    if use_claude:
        palette = (session.get("signature") or {}).get("palette", "")
        melody_fn = make_claude_melody_fn(palette)
        if melody_fn:
            print("  melody: Claude (a sung lead per scene, snapped to key)")
    if melody_fn is None:
        print("  melody: none (deterministic 3-track arrangement)")

    arrangement = build_arrangement(session, melody_fn=melody_fn)
    out_path = out_path or paths.arrangement_for(keepsake_path)
    with open(out_path, "w") as f:
        json.dump(arrangement, f, indent=2)

    n_notes = sum(len(r["notes"]) for r in arrangement["regions"])
    print(f"✓ Arranged {len(session['scenes'])} scenes → {out_path}")
    print(f"  {len(arrangement['tracks'])} tracks · "
          f"{len(arrangement['regions'])} regions · {n_notes} notes · "
          f"{TEMPO_BPM} BPM")
    print(f"\n  Next: push it into an editable Audiotool project:")
    print(f"    cd audiotool_export && npm install && "
          f"AT_PAT=at_pat_… ARRANGEMENT={out_path} npm start")
    return out_path


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--no-claude"]
    no_claude = "--no-claude" in sys.argv
    if not argv:
        print("Usage: python3 audiotool_arranger.py <keepsake-*.json> [out.json] [--no-claude]")
        raise SystemExit(1)
    arrange_file(argv[0], argv[1] if len(argv) > 1 else None, use_claude=not no_claude)
