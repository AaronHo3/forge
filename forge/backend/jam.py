"""
jam.py - Live Morph / Jam mode (MRT2's real-time showcase).

MRT2 generates a continuous, evolving stream; the player steers it live (blend,
density, prompt swaps) and the music morphs in ~1s. This is the thing ONLY MRT2
can do - SA3 (diffusion) renders fixed clips and can't be steered mid-stream.

JamState is the shared, thread-safe handle between:
  - the WORKER thread (mrt_worker.run_jam reads live params, appends audio), and
  - the WS handler (updates params from the browser, captures takes).

A captured take becomes a normal Clip, so it flows into the same Library →
Restyle → Audiotool funnel as everything else.
"""

from __future__ import annotations

import threading
import uuid
import os

import numpy as np

from .models import Clip, PromptSpec

SAMPLE_RATE = 48_000
_RECORD_CAP_CHUNKS = 2000   # ~26 min at ~0.8s/chunk - rolling cap so memory is bounded


class JamState:
    """Live, mutable session parameters + a rolling audio record. Reads of single
    attributes are atomic under the GIL; the record list is lock-guarded."""

    def __init__(self, prompt_a: str, prompt_b: str | None = None, key: str | None = None,
                 blend: float = 0.0, density: float = 0.3, drums: bool = False):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b or None
        self.key = key or None
        self.blend = blend
        self.density = density
        self.drums = drums
        self.chord: tuple[int, str] | None = None   # (root_pc, quality) for Harmony Sandbox; overrides key
        self.temperature: float | None = None        # live sampling "chaos" knob (None = worker default)
        self.top_k: int | None = None                # live sampling "focus" knob (None = worker default)
        self.running = True
        self._lock = threading.Lock()
        self._record: list[np.ndarray] = []

    def set_params(self, *, blend=None, density=None, drums=None,
                   temperature=None, top_k=None) -> None:
        if blend is not None:
            self.blend = max(0.0, min(1.0, float(blend)))
        if density is not None:
            self.density = max(0.0, min(1.0, float(density)))
        if drums is not None:
            self.drums = bool(drums)
        if temperature is not None:
            self.temperature = max(0.1, min(2.0, float(temperature)))
        if top_k is not None:
            self.top_k = max(1, min(128, int(top_k)))

    def set_chord(self, chord: tuple[int, str] | None) -> None:
        """Set the live harmony (root pitch class, quality), or None to free it.
        run_jam reads this and conditions MRT2's notes on it. Atomic under the GIL."""
        self.chord = chord

    def set_prompt(self, *, prompt_a=None, prompt_b=None, key=None) -> None:
        # Empty string clears the optional fields; prompt_a is never cleared blank.
        if prompt_a is not None and prompt_a.strip():
            self.prompt_a = prompt_a.strip()
        if prompt_b is not None:
            self.prompt_b = prompt_b.strip() or None
        if key is not None:
            self.key = key.strip() or None

    def append_record(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._record.append(chunk)
            if len(self._record) > _RECORD_CAP_CHUNKS:
                self._record.pop(0)

    def take(self, seconds: float | None = None) -> np.ndarray | None:
        """Snapshot the recorded audio (optionally just the last `seconds`)."""
        with self._lock:
            if not self._record:
                return None
            buf = np.concatenate(self._record, axis=0)
        if seconds:
            n = int(seconds * SAMPLE_RATE)
            buf = buf[-n:]
        return buf


def capture(state: JamState, clips_dir: str, *, seconds: float | None = None,
            created_by: str = "me") -> Clip:
    """Write the recorded jam (or its last `seconds`) to a WAV and return a Clip
    so it enters the normal Library/Restyle/Audiotool pipeline."""
    from .mrt_worker import MRTWorker  # noqa: PLC0415 - reuse the 16-bit PCM writer

    buf = state.take(seconds)
    if buf is None or buf.shape[0] < SAMPLE_RATE // 2:
        raise ValueError("nothing recorded yet - let it play for a moment first")
    os.makedirs(clips_dir, exist_ok=True)
    clip_id = uuid.uuid4().hex[:12]
    wav_path = os.path.join(clips_dir, f"{clip_id}.wav")
    MRTWorker._write_wav(wav_path, buf)
    dur = buf.shape[0] / SAMPLE_RATE
    spec = PromptSpec(
        text_a=state.prompt_a, text_b=state.prompt_b, blend=state.blend,
        key=state.key, density=state.density, drums=state.drums,
        chunks=max(1, round(dur / 0.8)),
    )
    return Clip(id=clip_id, wav_path=wav_path, spec=spec,
                created_by=created_by, engine="mrt2")
