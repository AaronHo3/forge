"""
forge_core.py - the core primitive everything else wraps.

    prompt → MRT2 → clip

ForgeCore turns a PromptSpec into a Clip (rendered WAV + metadata), and offers
the two higher-level moves a musician actually uses:
  - variations():   N takes from one spec (divergent exploration)
  - pregen_deck():  batch-render a curated deck offline so beginner/solo play
                    has ZERO latency at play time (GAME_PLAN.md §7.7)

It does not know about games, rooms, or the web - those sit on top of it.

STATUS: Phase 0 scaffold.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

from .engines.base import Backend
from .models import Clip, PromptSpec


class ForgeCore:
    def __init__(self, backend: Backend, clips_dir: str = "outputs/clips"):
        self._backend = backend
        self._clips_dir = clips_dir
        os.makedirs(clips_dir, exist_ok=True)

    def generate(self, spec: PromptSpec, created_by: str = "system",
                 timeout: float = 300.0) -> Clip:
        """Render one clip, blocking until the WAV is ready. Fine for the
        turn-based web layer (the single engine serializes all jobs anyway).
        The returned clip carries its style embedding (for novelty), if the
        engine provides one."""
        clip_id = uuid.uuid4().hex[:12]
        wav_path = os.path.join(self._clips_dir, f"{clip_id}.wav")
        _, emb = self._backend.render_blocking(spec, wav_path, timeout=timeout)
        return Clip(id=clip_id, wav_path=wav_path, spec=spec,
                    created_by=created_by, embedding=emb,
                    engine=getattr(self._backend, "name", ""))

    def variations(self, spec: PromptSpec, n: int = 4,
                   created_by: str = "system") -> list[Clip]:
        """N takes of the same idea by sweeping DENSITY (sparser → denser). MRT2's
        local build samples deterministically, so changing the input is the only
        way to get different output; density is a musically meaningful axis to vary."""
        lo = max(0.0, spec.density - 0.25)
        hi = min(1.0, spec.density + 0.25)
        out: list[Clip] = []
        for i in range(n):
            d = lo if n == 1 else lo + (hi - lo) * i / (n - 1)
            out.append(self.generate(replace(spec, density=d), created_by=created_by))
        return out

    def pregen_deck(self, specs: list[PromptSpec]) -> list[Clip]:
        """Batch-render a curated deck up front (beginner/solo zero-latency play)."""
        return [self.generate(s) for s in specs]
