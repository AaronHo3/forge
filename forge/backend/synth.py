"""
synth.py - precise tone synthesis for ear training.

MRT2 is a generative audio model - it can't reliably play an exact interval or
chord. Ear training needs *precise* pitches: a perfect fifth is a 3:2 frequency
ratio, an octave is 2:1. Those we synthesize deterministically with numpy - instant,
exact, and infinitely varied (random root each time). A soft multi-harmonic tone
with a decay envelope reads as a mellow electric-piano-ish note.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile

SR = 44_100


def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def tone(freq: float, dur: float) -> np.ndarray:
    """One note: fundamental + a couple of harmonics, with a soft attack and an
    exponential decay so it sounds like a struck key rather than a buzzer."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t)
            + 0.45 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.2 * np.sin(2 * np.pi * 3 * freq * t))
    env = np.exp(-3.2 * t / dur)
    a = max(1, int(SR * 0.006))                 # 6ms attack, no click
    env[:a] *= np.linspace(0, 1, a)
    return (wave * env).astype(np.float32)


def melodic(midis: list[int], note_dur: float = 0.75, gap: float = 0.06) -> np.ndarray:
    """Notes one after another."""
    parts = []
    silence = np.zeros(int(SR * gap), dtype=np.float32)
    for m in midis:
        parts.append(tone(midi_to_freq(m), note_dur))
        parts.append(silence)
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def harmonic(midis: list[int], dur: float = 1.6) -> np.ndarray:
    """Notes sounded together (a chord/interval block)."""
    tones = [tone(midi_to_freq(m), dur) for m in midis]
    n = min(len(x) for x in tones)
    mix = sum(x[:n] for x in tones) / len(tones)
    return mix.astype(np.float32)


def join(*segments: np.ndarray, gap: float = 0.18) -> np.ndarray:
    silence = np.zeros(int(SR * gap), dtype=np.float32)
    out = []
    for i, s in enumerate(segments):
        if i:
            out.append(silence)
        out.append(s)
    return np.concatenate(out)


def write_wav(path: str, mono: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = np.clip(mono * (0.9 / peak), -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    stereo = np.stack([pcm, pcm], axis=1)
    wavfile.write(path, SR, stereo)
