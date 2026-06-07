"""
net.py - networked Broken Record (Phase 2b): real-time rooms over WebSockets.

PARALLEL CHAINS (Gartic-Phone style): with P players there are P chains. Each turn
every player works on a DIFFERENT chain simultaneously, so nobody waits. Assignment:

    holder of chain c at turn t  =  player (c + t) mod P
    so player i at turn t works on chain (i - t) mod P

  turn 0  → player i seeds chain i (their own).
  turn t  → player i hears chain (i-t)'s previous clip and guesses → new clip.

Connection/lobby/timer plumbing lives in BaseHub; this file is just the telephone
game state + its turn lifecycle.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from .basehub import BaseGame, BaseHub
from .models import PromptSpec

_DEFAULT_SEEDS = ["calm ambient pad", "warm felt piano", "mellow lo-fi keys",
                  "soft acoustic guitar", "dreamy synth texture"]


@dataclass
class Hop:
    player: str
    text: str
    clip_id: str
    is_seed: bool
    score: float = 0.0
    tutor: str = ""


class Chain:
    def __init__(self, cid: int, origin: str):
        self.id = cid
        self.origin = origin
        self.entries: list[Hop] = []


class TelephoneRoom(BaseGame):
    def __init__(self, code: str, host: str, settings: dict):
        super().__init__(code, host, settings)
        self.turn = 0
        self.chains: list[Chain] = []
        self.submitted: set[str] = set()

    @property
    def turns(self) -> int:
        return self.settings["turns"]

    def start(self) -> None:
        self.settings["turns"] = max(2, min(int(self.settings.get("turns") or self.n), self.n))
        self.chains = [Chain(i, self.players[i]) for i in range(self.n)]
        self.phase = "playing"
        self.turn = 0
        self.submitted = set()

    def chain_index_for(self, player: str) -> int:
        return (self.players.index(player) - self.turn) % self.n

    def listen_clip_id(self, player: str) -> str | None:
        if self.turn == 0:
            return None
        return self.chains[self.chain_index_for(player)].entries[self.turn - 1].clip_id

    def assignment(self, player: str) -> dict:
        cid = self.listen_clip_id(player)
        return {"type": "turn", "turn": self.turn, "turns": self.turns,
                "is_seed": self.turn == 0,
                "listen": (f"/clips/{cid}.wav" if cid else None)}

    def record(self, player: str, text: str, clip_id: str) -> None:
        self.chains[self.chain_index_for(player)].entries.append(
            Hop(player, text, clip_id, self.turn == 0))
        self.submitted.add(player)

    def turn_done(self, expected: set[str]) -> bool:
        return expected.issubset(self.submitted)

    def advance(self) -> None:
        self.turn += 1
        self.submitted = set()
        if self.turn >= self.turns:
            self.phase = "reveal"

    def reveal(self, judge) -> dict:
        skill = self.settings.get("skill", "advanced")
        scores = {p: 0.0 for p in self.players}
        chains_out = []
        for c in self.chains:
            hops = []
            for k, e in enumerate(c.entries):
                if not e.is_seed and k > 0:
                    v = judge.score(e.text, c.entries[k - 1].text, skill)
                    e.score, e.tutor = v.score, v.tutor_note
                    scores[e.player] += v.score
                hops.append({"player": e.player, "text": e.text,
                             "clip_url": f"/clips/{e.clip_id}.wav",
                             "is_seed": e.is_seed, "score": round(e.score), "tutor": e.tutor})
            fidelity = None
            if len(c.entries) >= 2:
                fidelity = round(judge.score(c.entries[-1].text, c.entries[0].text, skill).score)
            chains_out.append({"id": c.id, "origin": c.origin, "hops": hops, "fidelity": fidelity})
        leaderboard = sorted(({"player": p, "score": round(s)} for p, s in scores.items()),
                             key=lambda d: -d["score"])
        return {"type": "reveal", "chains": chains_out, "leaderboard": leaderboard}


class Hub(BaseHub):
    # ── BaseHub hooks ─────────────────────────────────────────────────────────
    def _parse_settings(self, raw: dict) -> dict:
        return {**self._common_settings(raw), "turns": int(raw.get("turns") or 0)}

    def _new_game(self, code, host, settings) -> TelephoneRoom:
        return TelephoneRoom(code, host, settings)

    async def _start(self, code, name) -> None:
        room = self._games.get(code or "")
        if room is None or room.phase != "lobby":
            return
        if name != room.host:
            raise ValueError("only the host can start")
        if room.n < 2:
            raise ValueError("need at least 2 players")
        room.start()
        await self._begin_turn(code)

    async def _on_action(self, code, name, action, msg) -> None:
        if action == "submit":
            await self._submit(code, name, msg)
        else:
            raise ValueError(f"unknown action {action!r}")

    async def _on_leave_playing(self, code, name) -> None:
        room: TelephoneRoom = self._games.get(code)
        if room is None or room.phase != "playing":
            return
        if name not in room.submitted and name in room.players:
            cid = room.listen_clip_id(name) or (
                room.chains and room.chains[0].entries and room.chains[0].entries[-1].clip_id)
            if cid:
                room.record(name, "(left)", cid)
        if room.turn_done(set(self.conns(code))):
            await self._finish_turn(code)

    # ── Telephone logic ───────────────────────────────────────────────────────
    async def _submit(self, code, name, msg) -> None:
        room: TelephoneRoom = self._games.get(code or "")
        if room is None or room.phase != "playing":
            raise ValueError("not in a live game")
        if msg.get("turn") is not None and msg.get("turn") != room.turn:
            return
        if name in room.submitted:
            return
        text = (msg.get("text") or "").strip()
        from_clip = msg.get("from_clip")
        if room.turn == 0 and from_clip and self._storage.get_clip(from_clip):
            clip = self._storage.get_clip(from_clip)
            room.record(name, clip.spec.text_a, clip.id)
        else:
            if not text:
                raise ValueError("empty prompt")
            spec = self._build_spec(room, msg, text)
            clip = await asyncio.to_thread(self._forge.generate, spec, name)
            self._storage.put_clip(clip)
            room.record(name, text, clip.id)
        await self._broadcast(code, {"type": "progress", "done": len(room.submitted), "total": room.n})
        if room.turn_done(set(self.conns(code))):
            await self._finish_turn(code)

    def _build_spec(self, room: TelephoneRoom, msg: dict, text: str) -> PromptSpec:
        seed = room.turn == 0
        return PromptSpec(
            text_a=text,
            text_b=((msg.get("text_b") or "").strip() or None) if seed else None,
            blend=float(msg.get("blend", 0.0)) if seed else 0.0,
            key=room.settings["key"],
            density=float(msg.get("density", 0.3)) if seed else 0.3,
            drums=bool(msg.get("drums", False)) if seed else False,
            chunks=room.settings["chunks"])

    def _deadline(self, room: TelephoneRoom) -> int:
        t = room.settings.get("timer", 0) or 0
        if not t:
            return 0
        return max(45, t * 2) if room.turn == 0 else t

    async def _begin_turn(self, code) -> None:
        self._cancel_timer(code)
        room: TelephoneRoom = self._games.get(code)
        if room is None:
            return
        secs = self._deadline(room)
        for player, ws in list(self.conns(code).items()):
            a = room.assignment(player)
            a["seconds"] = secs or None
            await self._send(ws, a)
        if secs:
            self._arm_timer(code, self._turn_timer(code, room.turn, secs))

    async def _finish_turn(self, code) -> None:
        room: TelephoneRoom = self._games.get(code)
        if room is None:
            return
        room.advance()
        if room.phase == "reveal":
            self._cancel_timer(code)
            await self._broadcast(code, room.reveal(self._judge))
        else:
            await self._begin_turn(code)

    async def _turn_timer(self, code, turn_idx, secs) -> None:
        try:
            await asyncio.sleep(secs)
        except asyncio.CancelledError:
            return
        room: TelephoneRoom = self._games.get(code)
        if room is None or room.phase != "playing" or room.turn != turn_idx:
            return
        for name in list(self.conns(code)):
            if room.turn != turn_idx or room.phase != "playing":
                return
            if name in room.submitted:
                continue
            if room.turn == 0:
                spec = PromptSpec(text_a=random.choice(_DEFAULT_SEEDS),
                                  key=room.settings["key"], chunks=room.settings["chunks"])
                clip = await asyncio.to_thread(self._forge.generate, spec, name)
                self._storage.put_clip(clip)
                room.record(name, spec.text_a, clip.id)
            else:
                cid = room.listen_clip_id(name)
                if cid:
                    room.record(name, "(no answer)", cid)
        if room.turn == turn_idx and room.phase == "playing" and room.turn_done(set(self.conns(code))):
            await self._finish_turn(code)
