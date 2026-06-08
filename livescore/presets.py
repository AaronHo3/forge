"""
presets.py
----------
Per-use-case tuning bundles. The same engine becomes "Meditation mode" or
"D&D mode" by swapping a handful of numbers and a style hint — no code changes.

Each Preset gathers the knobs that were previously scattered across
FeatureMapper, LLMStyleDirector, and PythonMRTController:

  smoothing / drums_threshold / enable_drums → how the voice maps to params
  transcribe_interval / audio_window / cooldown → how fast it reacts to speech
  morph_step                      → how fast the music transitions on a change
  default_a / default_b           → the starting musical poles
  style_hint                      → biases the LLM's genre vocabulary

Add a new mode by adding one entry to PRESETS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str

    # ── FeatureMapper (voice → params) ───────────────────────────────────
    smoothing: float = 0.70        # higher = smoother/slower param glide
    drums_threshold: float = 0.85  # NORMALIZED energy (0–1, relative to the
                                   # speaker's own loudness) above which drums turn
                                   # on. 0.5 ≈ average; higher = needs a bigger
                                   # peak; >1 = never. Tuned per preset below.
    enable_drums: bool = False     # master switch: drums play only when this is
                                   # True AND energy crosses drums_threshold. Off
                                   # for intimate telling; on for dnd/stream/fitness.

    # ── LLMStyleDirector (reaction timing) ───────────────────────────────
    # Listening is ALWAYS continuous. `cooldown` only sets how often Claude is
    # consulted (cost/churn control); scene stability comes from Claude's
    # keep/change judgment + the crossfade length, not from going deaf.
    transcribe_interval: float = 1.5
    audio_window: float = 4.0
    cooldown: float = 4.0          # min seconds between Claude consultations

    # ── PythonMRTController (music transition) ───────────────────────────
    morph_step: float = 0.30       # per-chunk morph; 0.30 ≈ ~3s, 0.12 ≈ ~8s
    default_a: str = "mellow lo-fi chillhop, soft piano arpeggio, gentle beat, instrumental"
    default_b: str = "warm lo-fi hip hop beat, chill, gentle piano, instrumental"
    # Harmony is locked to this key from the FIRST chunk, so the note constraint
    # is never switched on mid-stream (which made the model stutter at the first
    # scene change). "C major" is ideal: it shares its 7 notes with A minor (its
    # relative), so the SAME locked key supports both bright and dark scenes —
    # the mood rides on the style, the harmony stays a stable in-tune bed.
    default_key: str = "C major"

    # ── LLM genre bias (appended to the system prompt) ───────────────────
    style_hint: str = ""


PRESETS: dict[str, Preset] = {

    # Balanced default — dramatic narration, audiobooks, bedtime stories.
    "storytelling": Preset(
        name="storytelling",
        cooldown=6.0,                 # let scenes breathe — fewer, calmer changes
        morph_step=0.20,              # ~5s glide between scenes (smoother flow)
        drums_threshold=0.9,          # HIGH on purpose — intimate telling stays
                                      # drumless except on a real, big emphasis
    ),

    # ⭐ DEMO MODE — intimate, personal storytelling tuned for MRT2's BEST-sounding
    # territory: warm, gentle, soft textures it renders beautifully. Avoids
    # everything it does badly (orchestral, percussion, action). Use this live.
    "intimate": Preset(
        name="intimate",
        smoothing=0.82,               # gentle, unhurried reactions
        drums_threshold=1.1,          # off — intimate stays drumless
        transcribe_interval=1.8,
        audio_window=5.0,
        cooldown=6.0,                 # let scenes breathe — fewer, calmer changes
        morph_step=0.16,              # slow, graceful scene morphs (~6s)
        default_a="soft felt piano, gentle and warm, intimate, instrumental",
        default_b="warm fingerpicked acoustic guitar, tender lo-fi, cozy, instrumental",
        style_hint=("INTIMATE, WARM, PRETTY instrumental music — this is MRT2's "
                    "sweet spot. Use ONLY soft, gentle timbres: felt piano, "
                    "fingerpicked acoustic guitar, warm Rhodes, mellow cello, soft "
                    "synth pads, harp, music box, lo-fi warmth. Always tender and "
                    "evolving. NEVER harsh, loud, percussive, orchestral, brass, or "
                    "fast. Convey emotion through warmth and softness, like a "
                    "personal memory set to music."),
    ),

    # Calm, slow, ambient. Scenes change rarely and morph gently. Drums off.
    "meditation": Preset(
        name="meditation",
        smoothing=0.88,
        drums_threshold=1.1,          # effectively never
        transcribe_interval=2.0,
        audio_window=5.0,
        cooldown=6.0,                 # consults Claude less often; calm by nature
        morph_step=0.10,              # ~10s, very gradual
        default_a="deep ambient drone, soft synth pads, slow and still, instrumental",
        default_b="warm gentle piano, airy strings, peaceful, instrumental",
        style_hint=("Strongly prefer calm, slow, ambient, drone, and minimal "
                    "instrumental textures. Never use percussion, fast tempos, "
                    "or busy arrangements. This is for relaxation and breathing."),
    ),

    # Tabletop RPG / D&D. Cinematic, reacts to combat vs exploration vs dread.
    "dnd": Preset(
        name="dnd",
        smoothing=0.65,
        drums_threshold=0.6,          # drums on dramatic swells (combat/action)
        enable_drums=True,
        cooldown=4.0,
        morph_step=0.30,
        default_a="dark fantasy dungeon ambience, low sustained strings, ominous drone, instrumental",
        default_b="warm adventurous orchestral pads, bright string texture, hopeful swell, instrumental",
        style_hint=("Cinematic fantasy ATMOSPHERE — textures and pads, not melodies. "
                    "Use sustained strings, low drones, soft brass swells, harp "
                    "shimmer, gentle percussion, and ambient orchestral beds. Convey "
                    "battle/tavern/forest/dungeon/magic through mood and timbre, "
                    "never a soaring melodic theme."),
    ),

    # Twitch / podcast underscore. Chill bed that reacts to host energy/hype.
    "stream": Preset(
        name="stream",
        smoothing=0.72,
        drums_threshold=0.62,         # groove comes in when the host gets hyped
        enable_drums=True,
        cooldown=5.0,
        morph_step=0.25,
        default_a="chill lo-fi beat, mellow electric piano, relaxed, instrumental",
        default_b="upbeat electronic groove, bright synth, energetic, instrumental",
        style_hint=("Modern, non-distracting background music: lo-fi beats, chillhop, "
                    "synthwave, downtempo electronic, future funk. Keep it groovy and "
                    "loopable, never cinematic or orchestral."),
    ),

    # Fitness / training. Fast, energetic, voice energy drives intensity hard.
    "fitness": Preset(
        name="fitness",
        smoothing=0.55,               # snappier reaction
        drums_threshold=0.45,         # drums kick in easily (just above average)
        enable_drums=True,
        cooldown=3.0,
        morph_step=0.40,              # ~2.5s, punchy
        default_a="steady warmup electronic pulse, driving bass, instrumental",
        default_b="high energy EDM workout, pounding four-on-the-floor, instrumental",
        style_hint=("High-energy workout music: EDM, big-room house, drum and bass, "
                    "electronic rock, hype trap. Always driving and motivational with "
                    "strong beats. Calmer only for cooldowns."),
    ),
}


def get(name: str) -> Preset:
    """Look up a preset by name, falling back to storytelling."""
    return PRESETS.get(name, PRESETS["storytelling"])
