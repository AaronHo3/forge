"""
config.py — engine constants shared by the live controller AND the offline
keepsake renderer, so a keepsake sounds like the same performance, not a
different mix.

Only constants that MUST match live here. Values that are intentionally
different between live (low-latency) and keepsake (offline-coherent) — chunk
size, top_k, palette anchor, morph rate — stay local to each module, on purpose.
"""

# Physical / format constants.
SAMPLE_RATE = 48_000     # MRT2 requires 48kHz; non-negotiable.
FRAMES_PER_SEC = 25      # MRT2 codec frame rate: 25 frames = 1.0s of audio.

# Audio character that must be identical for a live take and its keepsake.
STYLE_CFG = 2.0          # MusicCoCa style-guidance floor (too high -> harsh).
TEMPERATURE = 1.0        # sampling temperature (lower = more coherent/musical).
DECAY_RMS = 0.015        # decay watchdog: re-seed when the stream fades below...
DECAY_CHUNKS = 5         # ...this many chunks in a row (stops fade-to-silence).
