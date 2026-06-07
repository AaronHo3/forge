"""
telephone.py - "Broken Record", the first game mode (hotseat first).

THE LOOP (GAME_PLAN.md §3):
  1. Player 0 (the composer) writes a prompt → MRT2 → clip 0.
  2. Player 1 hears clip 0 ONLY → guesses the prompt → that guess → clip 1.
  3. Player 2 hears clip 1 → guesses → clip 2 ... around the circle.
  4. Reveal: the whole chain of prompts + clips, so everyone hears the DRIFT.

One pass = one hop per player (N players → N hops). Each clip is also a keepable
stem, so a telephone session doubles as crate-digging.

SCORING (Judge): a guess is scored against the prompt that made the clip the
player HEARD - i.e. the previous player's text. The seed (player 0) isn't scored.
"Chain fidelity" compares the final text back to the seed (how far it drifted).

Hotseat = one device passed around; the server holds game state, the browser
hides text between turns. The same TelephoneGame will back networked rooms later.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..forge_core import ForgeCore
from ..judge import Judge
from ..models import PromptSpec, Skill
from ..storage import Storage


@dataclass
class Hop:
    player: str
    text: str            # the prompt/guess this player wrote
    clip_id: str
    is_seed: bool
    score: float = 0.0   # filled at reveal (guesses only)
    tutor: str = ""


class TelephoneGame:
    name = "telephone"

    def __init__(self, gid: str, players: list[str], forge: ForgeCore, judge: Judge,
                 storage: Storage, *, chunks: int = 8, key: str | None = None,
                 skill: Skill = "advanced"):
        self.id = gid
        self.players = players
        self.chunks = chunks
        self.key = key
        self.skill: Skill = skill
        self.hops: list[Hop] = []
        self._forge = forge
        self._judge = judge
        self._storage = storage

    # ── Progress ─────────────────────────────────────────────────────────────
    @property
    def step(self) -> int:
        return len(self.hops)

    @property
    def total(self) -> int:
        return len(self.players)

    @property
    def complete(self) -> bool:
        return self.step >= self.total

    @property
    def current_player(self) -> str | None:
        return None if self.complete else self.players[self.step]

    def public_state(self) -> dict:
        return {
            "game_id": self.id, "players": self.players,
            "step": self.step, "total": self.total,
            "current_player": self.current_player, "complete": self.complete,
        }

    # ── Play ─────────────────────────────────────────────────────────────────
    def submit(self, text: str) -> dict:
        """The current player commits their prompt/guess; we render the clip the
        NEXT player will hear. Returns state + that clip's URL (text stays hidden)."""
        if self.complete:
            raise ValueError("game already complete")
        text = (text or "").strip()
        if not text:
            raise ValueError("empty prompt")
        player = self.players[self.step]
        is_seed = self.step == 0
        spec = PromptSpec(text_a=text, key=self.key, chunks=self.chunks)
        clip = self._forge.generate(spec, created_by=player)
        self._storage.put_clip(clip)            # so /clips/<id>.wav serves + crate works
        self.hops.append(Hop(player, text, clip.id, is_seed))
        out = self.public_state()
        out["clip_url"] = f"/clips/{clip.id}.wav"   # what the next player hears
        out["clip_id"] = clip.id
        return out

    # ── Reveal ───────────────────────────────────────────────────────────────
    def reveal(self) -> dict:
        scores = {p: 0.0 for p in self.players}
        chain = []
        for i, h in enumerate(self.hops):
            if not h.is_seed:
                v = self._judge.score(h.text, self.hops[i - 1].text, self.skill)
                h.score, h.tutor = v.score, v.tutor_note
                scores[h.player] += v.score
            chain.append({
                "player": h.player, "text": h.text,
                "clip_url": f"/clips/{h.clip_id}.wav", "clip_id": h.clip_id,
                "is_seed": h.is_seed, "score": round(h.score), "tutor": h.tutor,
            })
        fidelity = None
        if len(self.hops) >= 2:
            fidelity = round(
                self._judge.score(self.hops[-1].text, self.hops[0].text, self.skill).score)
        leaderboard = sorted(({"player": p, "score": round(s)}
                              for p, s in scores.items()),
                             key=lambda d: -d["score"])
        return {"chain": chain, "leaderboard": leaderboard, "fidelity": fidelity}


class TelephoneManager:
    """In-memory registry of active telephone games."""

    def __init__(self, forge: ForgeCore, judge: Judge, storage: Storage):
        self._forge = forge
        self._judge = judge
        self._storage = storage
        self._games: dict[str, TelephoneGame] = {}

    def new(self, players: list[str], *, chunks: int = 8, key: str | None = None,
            skill: Skill = "advanced") -> TelephoneGame:
        gid = uuid.uuid4().hex[:8]
        g = TelephoneGame(gid, players, self._forge, self._judge, self._storage,
                          chunks=chunks, key=key, skill=skill)
        self._games[gid] = g
        return g

    def get(self, gid: str) -> TelephoneGame | None:
        return self._games.get(gid)
