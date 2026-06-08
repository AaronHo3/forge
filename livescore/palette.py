"""
palette.py — the broad vocabulary of styles MRT2 can render.

MRT2 has no fixed instrument list (it's steered by a 768-d MusicCoCa text/audio
embedding, not a General-MIDI patch number), so "all the instruments it knows"
is really "every text prompt that lands somewhere musical in that space." This
file curates a wide, organised vocabulary across instrument families, genres,
and moods so we can (a) give the live director a rich palette to draw from and
(b) embed + cluster the whole thing to SEE what regions of MRT2's space are
reachable and distinct (see analyze_style_space.py).

Each entry is a short, MRT2-friendly phrase. `family` is only a label for
colouring the cluster map — the model never sees it.
"""

from __future__ import annotations

# ── Instruments, grouped by family ────────────────────────────────────────────
INSTRUMENTS: dict[str, list[str]] = {
    "keys": [
        "grand piano", "felt piano", "upright piano", "honky-tonk piano",
        "Rhodes electric piano", "Wurlitzer electric piano", "clavinet",
        "harpsichord", "celeste", "toy piano", "tack piano", "prepared piano",
    ],
    "guitar": [
        "nylon classical guitar", "fingerpicked acoustic guitar",
        "steel-string acoustic guitar", "clean electric guitar", "jazz guitar",
        "slide guitar", "twelve-string guitar", "surf guitar", "flamenco guitar",
        "ambient guitar swells",
    ],
    "strings": [
        "solo cello", "solo violin", "viola", "warm string ensemble",
        "pizzicato strings", "tremolo strings", "arco double bass",
        "fiddle", "cinematic string swell", "lush legato strings",
    ],
    "woodwind": [
        "flute", "clarinet", "oboe", "bassoon", "alto saxophone",
        "tenor saxophone", "soprano saxophone", "pan flute", "recorder",
        "english horn",
    ],
    "brass": [
        "muted trumpet", "flugelhorn", "french horn", "trombone", "tuba",
        "warm brass section", "soft brass swell", "jazz horn section",
    ],
    "mallet": [
        "vibraphone", "marimba", "glockenspiel", "xylophone", "kalimba",
        "music box", "steel drums", "hammered dulcimer", "celesta bells",
    ],
    "plucked_world": [
        "concert harp", "koto", "sitar", "banjo", "mandolin", "ukulele",
        "oud", "guzheng", "balalaika", "charango", "autoharp",
    ],
    "synth": [
        "warm analog synth pad", "ambient synth pad", "arpeggiated synth",
        "FM electric bells", "saw-wave synth lead", "vintage mellotron",
        "vaporwave synth", "retro synthwave lead", "soft sine pad",
        "granular synth texture", "modular bleeps",
    ],
    "bass": [
        "upright jazz bass", "electric fingered bass", "synth bass",
        "deep sub bass", "fretless bass", "slap bass",
    ],
    "drums_perc": [
        "soft brushed drums", "lo-fi hip hop beat", "boom-bap drums",
        "four-on-the-floor house beat", "breakbeat", "jazz ride cymbal groove",
        "hand percussion", "tabla", "congas and bongos", "trap hi-hats",
        "shuffling shaker groove", "marching snare",
    ],
    "ambient_texture": [
        "warm tape ambience", "vinyl crackle texture", "soft rain field recording",
        "low sustained drone", "reverberant cathedral pad", "windy atmosphere",
        "tape-saturated hum", "shimmering reverb wash",
    ],
    "voice_like": [
        "wordless humming choir", "airy vocal pad", "gospel organ",
        "church pipe organ", "harmonium",
    ],
}

# ── Genres ────────────────────────────────────────────────────────────────────
GENRES: list[str] = [
    "lo-fi hip hop", "ambient", "jazz trio", "bossa nova", "classical chamber",
    "cinematic film score", "synthwave", "downtempo electronic", "neo-soul",
    "indie folk", "delta blues", "gospel", "funk groove", "deep house",
    "drum and bass", "trip hop", "post-rock", "minimal techno", "smooth jazz",
    "baroque", "spaghetti western", "city pop", "dub reggae", "afrobeat",
    "chillhop", "new age", "dark ambient", "orchestral",
]

# ── Moods ─────────────────────────────────────────────────────────────────────
MOODS: list[str] = [
    "warm and hopeful", "dark and tense", "melancholic and tender",
    "triumphant and soaring", "dreamy and floating", "mysterious and uneasy",
    "playful and bright", "serene and still", "ominous and brooding",
    "nostalgic and bittersweet", "energetic and driving", "somber and heavy",
    "romantic and lush", "anxious and restless", "peaceful and meditative",
    "epic and cinematic",
]


def all_styles() -> list[tuple[str, str]]:
    """Flat [(text, family)] over the whole vocabulary. family is the cluster
    colour label: the instrument family, or 'genre' / 'mood'."""
    out: list[tuple[str, str]] = []
    for family, names in INSTRUMENTS.items():
        out.extend((n, family) for n in names)
    out.extend((g, "genre") for g in GENRES)
    out.extend((m, "mood") for m in MOODS)
    return out


def instrument_names() -> list[str]:
    """Just the instrument phrases (no genres/moods) — the live palette."""
    return [n for names in INSTRUMENTS.values() for n in names]


def families() -> list[str]:
    """All colour-label families, in a stable order."""
    return list(INSTRUMENTS.keys()) + ["genre", "mood"]
