"""
models.py - shared data types for Forge.

Mirrors the data model in GAME_PLAN.md §7.4. Frozen dataclasses: state changes
produce new objects, never mutate in place (project immutability rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Skill = Literal["beginner", "advanced"]
Phase = Literal["lobby", "round", "reveal", "done"]


@dataclass(frozen=True)
class PromptSpec:
    """Everything needed to generate one clip from MRT2.

    A single text prompt (text_a) is the simple case. Supplying text_b + blend
    drives MRT2's A↔B style interpolation (the "Collider" morph). The rest map
    onto generate() args proven in the livescore project's mrt_controller.py.
    """
    text_a: str
    text_b: str | None = None
    blend: float = 0.0          # 0.0 = full A, 1.0 = full B
    key: str | None = None      # e.g. "A minor" - optional harmonic anchor
    density: float = 0.3        # → cfg_musiccoca / chaos
    drums: bool = False
    chunks: int = 8             # generate() calls; ~0.8s each at CHUNK_FRAMES=20
    chord: tuple[int, str] | None = None   # (root_pc, quality) → note-condition the render on a chord


@dataclass(frozen=True)
class Clip:
    """A rendered piece of audio plus the spec that made it."""
    id: str
    wav_path: str
    spec: PromptSpec
    created_by: str = "system"
    embedding: tuple[float, ...] | None = None   # for novelty / similarity
    novelty_score: float | None = None
    name: str = ""                               # user-given display name (optional)
    engine: str = ""                             # which engine made it: "mrt2" | "sa3"


@dataclass(frozen=True)
class Submission:
    player_id: str
    guess: str
    clip_id: str | None = None   # the clip their guess generated (telephone)
    score: float = 0.0
    tutor_note: str = ""


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    skill: Skill = "advanced"
    score: float = 0.0


@dataclass(frozen=True)
class Round:
    mode: str
    truth: PromptSpec
    phase: Phase = "round"
    submissions: tuple[Submission, ...] = ()


@dataclass(frozen=True)
class Room:
    code: str
    mode: str
    skill: Skill = "advanced"
    players: tuple[Player, ...] = ()
    rounds: tuple[Round, ...] = ()
    phase: Phase = "lobby"

    def with_player(self, p: Player) -> "Room":
        return replace(self, players=self.players + (p,))
