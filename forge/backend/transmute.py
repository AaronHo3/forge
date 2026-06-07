"""
transmute.py - "Transmute": reinterpret an existing clip into a new one via SA3
audio-to-audio. The MRT2→SA3 bridge made concrete.

This is an SA3-only creative op (MRT2 has no audio-in editing mode), so it lives
here rather than in the engine-agnostic ForgeCore or the Backend protocol. It owns
its own SA3 engine handle; the source clip can be ANY WAV - MRT2-made, SA3-made,
or (later) user-uploaded - because SA3 only sees audio, not which model made it.
"""

from __future__ import annotations

import os
import uuid

from .engines.sa3 import SA3Backend
from .models import Clip, PromptSpec

_SECONDS_PER_CHUNK = 0.8   # keep the length slider consistent with the rest of Forge


class Transmuter:
    def __init__(self, sa3: SA3Backend, clips_dir: str = "outputs/clips"):
        self._sa3 = sa3
        self._dir = clips_dir
        os.makedirs(clips_dir, exist_ok=True)

    def transmute(
        self, src_wav_path: str, prompt: str, *,
        init_noise_level: float = 0.8, chunks: int = 10,
        created_by: str = "me", timeout: float = 300.0,
    ) -> Clip:
        """Restyle `src_wav_path` toward `prompt`. init_noise_level 0→preserve
        source, 1→ignore it. Returns a new Clip (embedding=None - SA3 has none)."""
        if not src_wav_path or not os.path.exists(src_wav_path):
            raise ValueError("source clip audio not found")
        clip_id = uuid.uuid4().hex[:12]
        out = os.path.join(self._dir, f"{clip_id}.wav")
        self._sa3.render_audio_to_audio(
            src_wav_path, prompt.strip(), out,
            init_noise_level=init_noise_level,
            duration=max(1.0, chunks * _SECONDS_PER_CHUNK), timeout=timeout,
        )
        spec = PromptSpec(text_a=prompt.strip(), chunks=chunks)
        return Clip(id=clip_id, wav_path=out, spec=spec,
                    created_by=created_by, embedding=None,
                    engine=getattr(self._sa3, "name", "sa3"))
