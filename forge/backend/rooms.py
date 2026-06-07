"""
rooms.py - per-room game state + the GameMode interface.

A Room is a lobby of Players running one mode. The mode-specific logic
(Telephone, Showdown, Battle) lives in backend/modes/* and implements GameMode.
RoomManager just holds rooms and routes player events to the active mode.

State is immutable (models.py frozen dataclasses): handlers return a NEW Room.

STATUS: scaffold - interface + routing skeleton.
"""

from __future__ import annotations

import random
import string
from typing import Protocol

from .forge_core import ForgeCore
from .judge import Judge
from .models import Player, Room


class GameMode(Protocol):
    """Each game mode implements this. Pure-ish: takes a Room, returns a Room."""

    name: str

    def start_round(self, room: Room) -> Room: ...
    def on_submit(self, room: Room, player_id: str, guess: str) -> Room: ...
    def reveal(self, room: Room) -> Room: ...


class RoomManager:
    def __init__(self, forge: ForgeCore, judge: Judge):
        self._forge = forge
        self._judge = judge
        self._rooms: dict[str, Room] = {}
        self._modes: dict[str, GameMode] = {}   # registered by name

    def register_mode(self, mode: GameMode) -> None:
        self._modes[mode.name] = mode

    def create(self, mode: str, skill: str = "advanced") -> Room:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        room = Room(code=code, mode=mode, skill=skill)  # type: ignore[arg-type]
        self._rooms[code] = room
        return room

    def join(self, code: str, player: Player) -> Room:
        room = self._rooms[code].with_player(player)
        self._rooms[code] = room
        return room

    def get(self, code: str) -> Room | None:
        return self._rooms.get(code)

    # Routing: server.py (WebSocket hub) calls these; they delegate to the mode.
    def start_round(self, code: str) -> Room:
        room = self._rooms[code]
        room = self._modes[room.mode].start_round(room)
        self._rooms[code] = room
        return room

    def submit(self, code: str, player_id: str, guess: str) -> Room:
        room = self._rooms[code]
        room = self._modes[room.mode].on_submit(room, player_id, guess)
        self._rooms[code] = room
        return room

    def reveal(self, code: str) -> Room:
        room = self._rooms[code]
        room = self._modes[room.mode].reveal(room)
        self._rooms[code] = room
        return room
