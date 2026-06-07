"""
battle.py - Forge Battle (Phase 3): craft-to-a-brief with a novelty award.

Each ROUND everyone gets the SAME constraint brief and crafts a prompt to hit it,
all at once. MRT2 renders every submission, then:

  - MATCH    = how well your prompt fits the brief (AI judge, 0–100)
  - NOVELTY  = 1 − cosine similarity of MusicCoCa style embeddings vs the field
               → bonus points; 🚀 Most Original = highest novelty
  - every clip is keepable (the harvest)

Connection/timer plumbing is in BaseHub; this file is the Battle game + flow.
"""

from __future__ import annotations

import asyncio
import random

from .basehub import BaseGame, BaseHub
from .judge import Judge
from .models import PromptSpec

_BRIEFS = [
    "haunting and sparse, like a forgotten lullaby",
    "triumphant sunrise, building to hope",
    "rainy neon city at 2am",
    "playful and bouncy, a toy parade",
    "vast and cold, drifting through deep space",
    "warm nostalgia, an old home movie",
    "tense chase, heart pounding",
    "dreamy underwater garden",
    "dusty desert highway at high noon",
    "cozy fireside while snow falls outside",
    "mischievous and jazzy, a midnight cat burglar",
    "epic mountain vista, wind and soaring strings",
]
_DEFAULTS = ["warm ambient texture", "mellow piano", "soft synth pad"]
NOVELTY_MAX = 40   # max bonus for the most original sound


class BattleGame(BaseGame):
    def __init__(self, code, host, settings):
        super().__init__(code, host, settings)
        self.round = 0
        self.scores: dict[str, float] = {host: 0.0}
        self.briefs = random.sample(_BRIEFS, len(_BRIEFS))
        self.subs: dict[str, dict] = {}

    def _on_add(self, name):
        self.scores[name] = 0.0

    def _on_remove(self, name):
        self.scores.pop(name, None)

    @property
    def total_rounds(self) -> int:
        r = int(self.settings.get("rounds") or 0)
        return r if r >= 1 else min(3, len(self.briefs))

    @property
    def brief(self) -> str:
        return self.briefs[self.round % len(self.briefs)]

    @property
    def is_last(self) -> bool:
        return self.round + 1 >= self.total_rounds

    def begin_round(self):
        self.phase = "craft"
        self.subs = {}

    def score_round(self, judge: Judge) -> dict:
        skill = self.settings.get("skill", "advanced")
        names = list(self.subs)
        results, best_nov, most_original = [], -1.0, None
        for p in names:
            s = self.subs[p]
            match = judge.score(s["text"], self.brief, skill).score
            others = [self.subs[q]["embedding"] for q in names if q != p]
            nov = Judge.novelty(s["embedding"], others)
            bonus = round(NOVELTY_MAX * nov)
            total = round(match) + bonus
            self.scores[p] += total
            results.append({"player": p, "text": s["text"],
                            "clip_url": f"/clips/{s['clip_id']}.wav", "clip_id": s["clip_id"],
                            "match": round(match), "novelty_bonus": bonus, "total": total})
            if nov > best_nov:
                best_nov, most_original = nov, p
        for r in results:
            r["most_original"] = (r["player"] == most_original)
        results.sort(key=lambda d: -d["total"])
        leaderboard = sorted(({"player": p, "score": round(s)} for p, s in self.scores.items()),
                             key=lambda d: -d["score"])
        return {"type": "reveal", "brief": self.brief, "results": results,
                "leaderboard": leaderboard, "most_original": most_original,
                "round": self.round + 1, "rounds": self.total_rounds, "is_last": self.is_last}


class BattleHub(BaseHub):
    def _parse_settings(self, raw: dict) -> dict:
        s = {**self._common_settings(raw), "rounds": int(raw.get("rounds") or 0)}
        s["timer"] = max(0, min(int(raw.get("timer", 45) or 0), 300))   # craft default 45s
        return s

    def _new_game(self, code, host, settings) -> BattleGame:
        return BattleGame(code, host, settings)

    async def _start(self, code, name) -> None:
        game: BattleGame = self._games.get(code or "")
        if game is None or game.phase != "lobby":
            return
        if name != game.host:
            raise ValueError("only the host can start")
        if game.n < 2:
            raise ValueError("need at least 2 players")
        game.round = 0
        game.begin_round()
        await self._begin_craft(code)

    async def _on_action(self, code, name, action, msg) -> None:
        if action == "craft":
            await self._craft(code, name, msg)
        elif action == "next":
            await self._next(code, name)
        else:
            raise ValueError(f"unknown action {action!r}")

    async def _on_leave_playing(self, code, name) -> None:
        game: BattleGame = self._games.get(code)
        if game and game.phase == "craft" and game.subs and \
                len(game.subs) >= len(set(self.conns(code))):
            await self._end_round(code)

    async def _craft(self, code, name, msg) -> None:
        game: BattleGame = self._games.get(code or "")
        if game is None or game.phase != "craft" or name in game.subs:
            return
        text = (msg.get("text") or "").strip()
        if not text:
            raise ValueError("empty prompt")
        spec = PromptSpec(text_a=text, text_b=(msg.get("text_b") or "").strip() or None,
                          blend=float(msg.get("blend", 0.0)), key=game.settings["key"],
                          density=float(msg.get("density", 0.3)), drums=bool(msg.get("drums", False)),
                          chunks=game.settings["chunks"])
        clip = await asyncio.to_thread(self._forge.generate, spec, name)
        self._storage.put_clip(clip)
        game.subs[name] = {"text": text, "clip_id": clip.id, "embedding": clip.embedding}
        await self._broadcast(code, {"type": "progress", "done": len(game.subs),
                                     "total": len(self.conns(code))})
        if len(game.subs) >= len(set(self.conns(code))):
            await self._end_round(code)

    async def _next(self, code, name) -> None:
        game: BattleGame = self._games.get(code or "")
        if game is None or game.phase != "reveal" or name != game.host or game.is_last:
            return
        game.round += 1
        game.begin_round()
        await self._begin_craft(code)

    async def _begin_craft(self, code) -> None:
        self._cancel_timer(code)
        game: BattleGame = self._games[code]
        secs = game.settings.get("timer", 0) or 0
        craft_secs = max(45, secs) if secs else 0
        await self._broadcast(code, {"type": "brief", "brief": game.brief,
                                     "round": game.round + 1, "rounds": game.total_rounds,
                                     "seconds": craft_secs or None})
        if craft_secs:
            self._arm_timer(code, self._craft_timer(code, game.round, craft_secs))

    async def _end_round(self, code) -> None:
        self._cancel_timer(code)
        game: BattleGame = self._games.get(code)
        if game is None or game.phase != "craft":
            return
        game.phase = "reveal"
        await self._broadcast(code, game.score_round(self._judge))

    async def _craft_timer(self, code, rnd, secs) -> None:
        try:
            await asyncio.sleep(secs)
        except asyncio.CancelledError:
            return
        game: BattleGame = self._games.get(code)
        if game is None or game.phase != "craft" or game.round != rnd:
            return
        for name in list(self.conns(code)):
            if game.phase != "craft" or game.round != rnd:
                return
            if name in game.subs:
                continue
            spec = PromptSpec(text_a=random.choice(_DEFAULTS),
                              key=game.settings["key"], chunks=game.settings["chunks"])
            clip = await asyncio.to_thread(self._forge.generate, spec, name)
            self._storage.put_clip(clip)
            game.subs[name] = {"text": spec.text_a, "clip_id": clip.id, "embedding": clip.embedding}
        if game.phase == "craft" and game.round == rnd:
            await self._end_round(code)
