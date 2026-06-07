"""
basehub.py - shared plumbing for the networked party modes.

Broken Record, Showdown, and Forge Battle all need the same connection lifecycle:
create/join rooms by code, track one WebSocket per player, broadcast lobby/state,
clean up on disconnect, and run a single per-room countdown timer. That machinery
lives here once; each mode subclasses BaseHub and implements only its game-specific
hooks (settings, game object, start, actions, leave-during-play).

Concurrency: all room-state mutation happens on the asyncio event loop (never inside
a thread), so it's effectively serialized - no locks. Modes push only the blocking
MRT2 generation off-loop via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import random

from fastapi import WebSocket, WebSocketDisconnect

from .forge_core import ForgeCore
from .judge import Judge
from .storage import Storage


def gen_code() -> str:
    return "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4))


class BaseGame:
    """Common room state: code, host, players, settings, phase."""

    def __init__(self, code: str, host: str, settings: dict):
        self.code = code
        self.host = host
        self.players: list[str] = [host]
        self.settings = settings
        self.phase = "lobby"

    @property
    def n(self) -> int:
        return len(self.players)

    def add_player(self, name: str) -> None:
        if self.phase != "lobby":
            raise ValueError("game already started")
        if not name:
            raise ValueError("need a name")
        if name in self.players:
            raise ValueError("that name is taken")
        if self.n >= 8:
            raise ValueError("room is full")
        self.players.append(name)
        self._on_add(name)

    def remove_player(self, name: str) -> None:
        if name in self.players:
            self.players.remove(name)
            self._on_remove(name)

    def _on_add(self, name: str) -> None:    # hook (e.g. init score)
        pass

    def _on_remove(self, name: str) -> None:
        pass

    def lobby_state(self) -> dict:
        return {"type": "lobby", "players": list(self.players),
                "host": self.host, "settings": self.settings}


class BaseHub:
    """Owns rooms + their WebSocket connections; drives create/join/leave + timers.
    Subclasses implement the game-specific hooks below."""

    def __init__(self, forge: ForgeCore, judge: Judge, storage: Storage):
        self._forge = forge
        self._judge = judge
        self._storage = storage
        self._games: dict[str, BaseGame] = {}
        self._conns: dict[str, dict[str, WebSocket]] = {}
        self._timers: dict[str, asyncio.Task] = {}

    # ── Connection lifecycle ─────────────────────────────────────────────────
    async def handle(self, ws: WebSocket) -> None:
        await ws.accept()
        code: str | None = None
        name: str | None = None
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    action = msg.get("action")
                    if action == "create":
                        code, name = await self._create(ws, msg)
                    elif action == "join":
                        code, name = await self._join(ws, msg)
                    elif action == "start":
                        await self._start(code, name)
                    else:
                        await self._on_action(code, name, action, msg)
                except ValueError as e:
                    # Recoverable user/protocol error (bad code, too few players,
                    # empty prompt, malformed JSON). Report it but KEEP the socket
                    # and the room alive so the player can simply retry - do NOT
                    # _leave (that would destroy a 1-player room on a failed start).
                    await self._err(ws, str(e))
        except WebSocketDisconnect:
            await self._leave(code, name)
        except Exception as e:  # noqa: BLE001 - unexpected: report + tear down
            await self._err(ws, str(e))
            await self._leave(code, name)

    async def _create(self, ws: WebSocket, msg: dict) -> tuple[str, str]:
        name = (msg.get("name") or "").strip() or "Host"
        settings = self._parse_settings(msg.get("settings") or {})
        code = gen_code()
        while code in self._games:
            code = gen_code()
        self._games[code] = self._new_game(code, name, settings)
        self._conns[code] = {name: ws}
        await self._send(ws, {"type": "joined", "code": code, "you": name, "is_host": True})
        await self._lobby(code)
        return code, name

    async def _join(self, ws: WebSocket, msg: dict) -> tuple[str, str]:
        code = (msg.get("code") or "").strip().upper()
        name = (msg.get("name") or "").strip() or "Player"
        game = self._games.get(code)
        if game is None:
            raise ValueError("no room with that code")
        game.add_player(name)
        self._conns[code][name] = ws
        await self._send(ws, {"type": "joined", "code": code, "you": name, "is_host": False})
        await self._lobby(code)
        return code, name

    async def _leave(self, code: str | None, name: str | None) -> None:
        if not code or code not in self._conns:
            return
        self._conns[code].pop(name, None)
        game = self._games.get(code)
        if game is not None:
            if game.phase == "lobby":
                game.remove_player(name)
                await self._lobby(code)
            else:
                await self._on_leave_playing(code, name)
        if not self._conns.get(code):
            self._cancel_timer(code)
            self._games.pop(code, None)
            self._conns.pop(code, None)

    # ── Hooks for subclasses ─────────────────────────────────────────────────
    def _parse_settings(self, raw: dict) -> dict:
        raise NotImplementedError

    def _new_game(self, code: str, host: str, settings: dict) -> BaseGame:
        raise NotImplementedError

    async def _start(self, code: str | None, name: str | None) -> None:
        raise NotImplementedError

    async def _on_action(self, code: str | None, name: str | None,
                         action: str, msg: dict) -> None:
        raise NotImplementedError

    async def _on_leave_playing(self, code: str, name: str) -> None:
        pass

    # ── Shared helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _common_settings(raw: dict) -> dict:
        skill = raw.get("skill")
        return {
            "chunks": max(2, min(int(raw.get("chunks", 4)), 40)),
            "key": (raw.get("key") or None),
            "skill": skill if skill in ("beginner", "advanced") else "advanced",
            "timer": max(0, min(int(raw.get("timer", 30) or 0), 300)),
        }

    def conns(self, code: str | None) -> dict[str, WebSocket]:
        return self._conns.get(code or "", {})

    async def _lobby(self, code: str) -> None:
        game = self._games.get(code)
        if game:
            await self._broadcast(code, game.lobby_state())

    async def _broadcast(self, code: str, msg: dict) -> None:
        for ws in list(self.conns(code).values()):
            await self._send(ws, msg)

    @staticmethod
    async def _send(ws: WebSocket, msg: dict) -> None:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:  # noqa: BLE001 - socket gone; _leave handles cleanup
            pass

    async def _err(self, ws: WebSocket, message: str) -> None:
        await self._send(ws, {"type": "error", "message": message})

    def _cancel_timer(self, code: str) -> None:
        t = self._timers.pop(code, None)
        if t and not t.done():
            t.cancel()

    def _arm_timer(self, code: str, coro) -> None:
        self._cancel_timer(code)
        self._timers[code] = asyncio.create_task(coro)
