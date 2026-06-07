"""
prompt_hints.py - curated, MRT2-friendly prompt building blocks.

Shown as clickable chips so players (and workbench users) who don't know what to
type have a starting point. These are phrases that reliably steer MRT2 toward
musical, coherent output - grouped by the dimension they control.

Shared by the workbench and the games (served at GET /api/hints).
"""

from __future__ import annotations

HINTS: dict[str, list[str]] = {
    "Instrument": [
        "felt piano", "Rhodes keys", "fingerpicked nylon guitar", "warm upright bass",
        "vibraphone", "harp", "cello", "music box", "analog synth pad",
        "brushed drums", "marimba", "electric piano",
    ],
    "Mood": [
        "warm and nostalgic", "dark and tense", "dreamy", "melancholy",
        "triumphant", "playful", "hopeful", "mysterious", "calm and spacious",
    ],
    "Genre": [
        "lo-fi hip-hop", "ambient", "cinematic score", "jazz trio", "bossa nova",
        "synthwave", "folk", "chillhop", "neo-soul",
    ],
    "Tempo / feel": [
        "slow and spacious", "mid-tempo groove", "upbeat and driving",
        "downtempo", "gentle swing", "steady pulse",
    ],
    "Production": [
        "tape saturation", "vinyl crackle", "reverb-drenched", "intimate close-mic",
        "warm analog", "lo-fi dusty",
    ],
}
