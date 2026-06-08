"""
speaker_signature.py
--------------------
Derives a stable "musical signature" from a speaker's voice *timbre* — so the
same person always produces a recognizably similar song, and two people saying
the SAME words produce DIFFERENT songs.

The earlier voice features (energy, pitch, brightness) describe *intensity and
content* — they can't tell people apart. Identity lives in TIMBRE: the unique
resonances of a person's vocal tract. We capture it with MFCCs (mel-frequency
cepstral coefficients), the classic voice-timbre fingerprint — no extra model
dependencies, just librosa.

The mean MFCC profile over a few seconds of speech is the speaker's vocal
"color". We map it deterministically to:
  - a PALETTE  (the instrument family the song is built from)
  - a KEY      (the tonal home of the song)
  - a TEMPO    feel

Same voice → same signature. Different voice → different signature. That is the
"uniquely you" layer. (Upgradeable later to a learned speaker embedding such as
resemblyzer/ECAPA for an even purer identity vector.)
"""

import numpy as np

SAMPLE_RATE = 48_000
N_MFCC = 20

# Instrument palettes the signature can land on. Each is a short, MusicCoCa-
# friendly description of a sonic identity. Ordered loosely dark → bright.
PALETTES = [
    "deep warm tones: low cello, double bass, mellow felt piano",
    "warm woody tones: cello, acoustic guitar, soft piano",
    "rich velvet tones: warm strings, vibraphone, mellow Rhodes",
    "intimate tones: fingerpicked nylon guitar, music box, felt piano",
    "airy tones: soft pads, ambient synth, breathy flute, glassy keys",
    "bright crystalline tones: glockenspiel, harp, bright piano, bells",
]

KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]


def _proj_index(vec: np.ndarray, n: int, seed: int) -> int:
    """Deterministically map a feature vector to an index in [0, n).
    A fixed seeded projection makes this stable across runs (unlike hash())."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(vec.shape[0])
    val = float(np.dot(vec, w))
    return int(abs(val) * 100) % n


class SpeakerSignature:
    """Accumulates voiced audio and derives a stable musical signature."""

    MIN_SAMPLES = 3   # MFCC windows needed before the signature is trustworthy

    def __init__(self):
        self._mfcc_sum = None
        self._count = 0
        self._frozen = None

    def add_audio(self, audio_48k):
        """Feed a chunk of voiced (non-silent) mic audio."""
        if audio_48k is None or len(audio_48k) < SAMPLE_RATE // 4:
            return
        import librosa
        audio_16k = librosa.resample(np.asarray(audio_48k, dtype=np.float32),
                                     orig_sr=SAMPLE_RATE, target_sr=16_000)
        mfcc = librosa.feature.mfcc(y=audio_16k, sr=16_000, n_mfcc=N_MFCC)
        prof = mfcc.mean(axis=1)
        self._mfcc_sum = prof if self._mfcc_sum is None else self._mfcc_sum + prof
        self._count += 1
        self._frozen = None   # invalidate cache

    @property
    def ready(self) -> bool:
        return self._count >= self.MIN_SAMPLES

    @property
    def windows(self) -> int:
        """How many voiced windows have been blended into this signature."""
        return self._count

    def signature(self) -> dict:
        """The stable musical signature derived from the accumulated voice."""
        if self._frozen is not None:
            return self._frozen
        prof = (self._mfcc_sum / max(1, self._count)) if self._mfcc_sum is not None \
            else np.zeros(N_MFCC)

        # MFCC[1] is the spectral tilt (overall brightness of the voice).
        tilt = float(prof[1]) if prof.shape[0] > 1 else 0.0
        bright = 1.0 / (1.0 + np.exp(-tilt / 30.0))          # 0..1 brightness

        # Palette + key from projections of the timbre profile, so different
        # voices spread across the space (deterministic per speaker).
        palette = PALETTES[_proj_index(prof[1:8], len(PALETTES), seed=13)]
        key     = KEYS[_proj_index(prof[2:9], len(KEYS), seed=7)]
        tempo   = "relaxed and spacious" if tilt < 0 else "gently flowing"

        self._frozen = {
            "palette": palette,
            "key": key,
            "tempo": tempo,
            "brightness": round(bright, 3),
        }
        return self._frozen

    def describe(self) -> str:
        s = self.signature()
        return f"{s['key']} · {s['tempo']} · {s['palette'][:40]}"
