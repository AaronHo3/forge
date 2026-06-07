"""
showdown.py - Showdown (Phase 3): Kahoot-style synchronous guessing.

Each ROUND one player is the Composer: they craft a clip from a hidden prompt while
everyone waits. The clip then plays for all guessers at once and a countdown runs -
faster correct guesses score more. The Composer rotates each round.

  start → [compose → listen+guess → reveal] × rounds → final leaderboard

Scoring: points = judge_accuracy × speed_factor (1.0 instant → 0.5 at the buzzer);
the Composer earns the round's average accuracy. Connection/timer plumbing is in
BaseHub; this file is the Showdown game state + phase flow.
"""

from __future__ import annotations

import asyncio
import random
import time

from .basehub import BaseGame, BaseHub
from .models import PromptSpec

_DEFAULT_PROMPTS = ["warm lo-fi piano", "dreamy synth pad", "mellow jazz guitar",
                    "calm ambient texture", "soft felt keys"]


class ShowdownGame(BaseGame):
    def __init__(self, code, host, settings):
        super().__init__(code, host, settings)
        self.round = 0
        self.scores: dict[str, float] = {host: 0.0}
        self.truth: str | None = None
        self.clip_id: str | None = None
        self.guesses: dict[str, dict] = {}
        self.guess_open = 0.0

    def _on_add(self, name):
        self.scores[name] = 0.0

    def _on_remove(self, name):
        self.scores.pop(name, None)

    @property
    def total_rounds(self) -> int:
        r = int(self.settings.get("rounds") or 0)
        return r if r >= 1 else self.n

    @property
    def composer(self) -> str:
        return self.players[self.round % self.n]

    @property
    def is_last(self) -> bool:
        return self.round + 1 >= self.total_rounds

    def begin_round(self):
        self.phase = "compose"
        self.truth = None
        self.clip_id = None
        self.guesses = {}

    def round_reveal(self) -> dict:
        guesses = [{"player": p, "text": g["text"], "score": round(g["score"])}
                   for p, g in sorted(self.guesses.items(), key=lambda kv: -kv[1]["score"])]
        leaderboard = sorted(({"player": p, "score": round(s)} for p, s in self.scores.items()),
                             key=lambda d: -d["score"])
        return {"type": "reveal", "truth": self.truth, "composer": self.composer,
                "guesses": guesses, "leaderboard": leaderboard,
                "round": self.round + 1, "rounds": self.total_rounds, "is_last": self.is_last}


class ShowdownHub(BaseHub):
    def _parse_settings(self, raw: dict) -> dict:
        return {**self._common_settings(raw), "rounds": int(raw.get("rounds") or 0)}

    def _new_game(self, code, host, settings) -> ShowdownGame:
        return ShowdownGame(code, host, settings)

    async def _start(self, code, name) -> None:
        game: ShowdownGame = self._games.get(code or "")
        if game is None or game.phase != "lobby":
            return
        if name != game.host:
            raise ValueError("only the host can start")
        if game.n < 2:
            raise ValueError("need at least 2 players")
        game.round = 0
        game.begin_round()
        await self._begin_compose(code)

    async def _on_action(self, code, name, action, msg) -> None:
        if action == "compose":
            await self._compose(code, name, msg)
        elif action == "guess":
            await self._guess(code, name, msg)
        elif action == "next":
            await self._next(code, name)
        else:
            raise ValueError(f"unknown action {action!r}")

    async def _on_leave_playing(self, code, name) -> None:
        game: ShowdownGame = self._games.get(code)
        if game and game.phase == "guess" and len(game.guesses) >= self._n_guessers(code, game):
            await self._end_round(code)

    # ── Phases ────────────────────────────────────────────────────────────────
    async def _compose(self, code, name, msg) -> None:
        game: ShowdownGame = self._games.get(code or "")
        if game is None or game.phase != "compose" or name != game.composer:
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
        game.truth, game.clip_id = text, clip.id
        await self._begin_guess(code)

    async def _guess(self, code, name, msg) -> None:
        game: ShowdownGame = self._games.get(code or "")
        if game is None or game.phase != "guess" or name == game.composer or name in game.guesses:
            return
        text = (msg.get("text") or "").strip()
        if not text:
            raise ValueError("empty guess")
        acc = self._judge.score(text, game.truth, game.settings.get("skill", "advanced")).score
        game.guesses[name] = {"text": text, "score": acc * self._speed_factor(game)}
        await self._broadcast(code, {"type": "progress", "done": len(game.guesses),
                                     "total": self._n_guessers(code, game)})
        if len(game.guesses) >= self._n_guessers(code, game):
            await self._end_round(code)

    async def _next(self, code, name) -> None:
        game: ShowdownGame = self._games.get(code or "")
        if game is None or game.phase != "reveal" or name != game.host or game.is_last:
            return
        game.round += 1
        game.begin_round()
        await self._begin_compose(code)

    async def _begin_compose(self, code) -> None:
        self._cancel_timer(code)
        game: ShowdownGame = self._games[code]
        secs = game.settings.get("timer", 0) or 0
        compose_secs = max(60, secs * 2) if secs else 0
        for player, ws in list(self.conns(code).items()):
            await self._send(ws, {"type": "compose", "round": game.round + 1,
                                  "rounds": game.total_rounds, "composer": game.composer,
                                  "you_are_composer": player == game.composer,
                                  "seconds": compose_secs or None})
        if compose_secs:
            self._arm_timer(code, self._compose_timer(code, game.round, compose_secs))

    async def _begin_guess(self, code) -> None:
        self._cancel_timer(code)
        game: ShowdownGame = self._games[code]
        game.phase = "guess"
        game.guess_open = time.monotonic()
        secs = game.settings.get("timer", 0) or 0
        for player, ws in list(self.conns(code).items()):
            await self._send(ws, {"type": "listen", "clip_url": f"/clips/{game.clip_id}.wav",
                                  "composer": game.composer, "you_are_composer": player == game.composer,
                                  "round": game.round + 1, "rounds": game.total_rounds,
                                  "seconds": secs or None})
        if secs:
            self._arm_timer(code, self._guess_timer(code, game.round, secs))

    async def _end_round(self, code) -> None:
        self._cancel_timer(code)
        game: ShowdownGame = self._games.get(code)
        if game is None or game.phase != "guess":
            return
        if game.guesses:
            avg = sum(g["score"] for g in game.guesses.values()) / len(game.guesses)
            game.scores[game.composer] += avg
        for p, g in game.guesses.items():
            game.scores[p] += g["score"]
        game.phase = "reveal"
        await self._broadcast(code, game.round_reveal())

    def _speed_factor(self, game: ShowdownGame) -> float:
        total = game.settings.get("timer", 0) or 0
        if not total:
            return 1.0
        remaining = max(0.0, total - (time.monotonic() - game.guess_open))
        return 0.5 + 0.5 * (remaining / total)

    def _n_guessers(self, code, game: ShowdownGame) -> int:
        return max(0, len(set(self.conns(code)) - {game.composer}))

    async def _compose_timer(self, code, rnd, secs) -> None:
        try:
            await asyncio.sleep(secs)
        except asyncio.CancelledError:
            return
        game: ShowdownGame = self._games.get(code)
        if game is None or game.phase != "compose" or game.round != rnd:
            return
        if game.composer not in self.conns(code) or game.truth is None:
            spec = PromptSpec(text_a=random.choice(_DEFAULT_PROMPTS),
                              key=game.settings["key"], chunks=game.settings["chunks"])
            clip = await asyncio.to_thread(self._forge.generate, spec, game.composer)
            self._storage.put_clip(clip)
            game.truth, game.clip_id = spec.text_a, clip.id
            if game.phase == "compose" and game.round == rnd:
                await self._begin_guess(code)

    async def _guess_timer(self, code, rnd, secs) -> None:
        try:
            await asyncio.sleep(secs)
        except asyncio.CancelledError:
            return
        game: ShowdownGame = self._games.get(code)
        if game is None or game.phase != "guess" or game.round != rnd:
            return
        await self._end_round(code)
