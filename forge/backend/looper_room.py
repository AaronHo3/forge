"""
looper_room.py - the Multiplayer Looper Room.

Each player joins and claims an INSTRUMENT (a track). MRT2 renders that
instrument as a bar-locked loop (see looper.py). The SERVER holds every track's
loop buffer and runs a lightweight CPU mixer that loops and sums them into one
synced stream, broadcast to the whole room. Everyone hears the same band.

WHY SERVER-SIDE MIXING: one shared mix that is identical and in sync for every
listener. Each browser just plays the received PCM. Mix knobs (volume, pan, mute,
solo) update the mixer instantly; instrument/re-roll changes re-render through the
worker and swap in at the next loop boundary.

TWO CLOCKS:
- the MODEL renders loops occasionally + slowly, through the single worker queue;
- a MIXER THREAD loops the finished buffers in real time and fans out PCM.
They never block each other. Sync is free because every track buffer is an exact
multiple of the same loop length L, so the mixer keeps ONE master position and
samples every track at it.
"""

from __future__ import annotations

import asyncio
import os
import random
import string
import threading
import time
import uuid

import numpy as np

from .looper import INSTRUMENTS, instrument_list, read_buffer
from .mrt_worker import MRTWorker

SR = 48_000
MIX_CHUNK = 4800        # 0.1s mixer blocks
MIX_LEAD = 0.3          # keep the broadcast ~this far ahead of the wall clock
AUDIO_BACKLOG = 8
DEFAULT_KEY = "C major"
DEFAULT_BARS = 2


class Track:
    def __init__(self, owner: str, instrument: str):
        self.owner = owner
        self.instrument = instrument
        self.custom_prompt = ""                    # per-instrument re-prompt (overrides preset)
        self.buf: np.ndarray | None = None       # (windows*lsamp, 2) float32
        self.pending: np.ndarray | None = None    # swapped in at the next loop wrap
        self.windows = 1
        self.win = 0
        self.pending_win: int | None = None
        self.vol = 0.85
        self.pan = 0.0
        self.muted = False
        self.solo = False
        self.ready = False
        self.rendering = False
        self.tempo_detected = 0.0
        self.stretch = 1.0

    def meta(self) -> dict:
        inst = INSTRUMENTS.get(self.instrument, {})
        return {"owner": self.owner, "instrument": self.instrument,
                "label": inst.get("label", self.instrument), "icon": inst.get("icon", "🎵"),
                "prompt": self.custom_prompt or inst.get("prompt", ""),
                "vol": round(self.vol, 3), "pan": round(self.pan, 3),
                "muted": self.muted, "solo": self.solo, "win": self.win,
                "windows": self.windows, "ready": self.ready, "rendering": self.rendering,
                "tempo_detected": self.tempo_detected, "stretch": self.stretch}


class _Client:
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name
        self.q: asyncio.Queue = asyncio.Queue()
        self.pump: asyncio.Task | None = None


class LooperRoom:
    def __init__(self, code: str, host: str):
        self.code = code
        self.host = host
        self.clients: dict[str, _Client] = {}
        self.tracks: dict[str, Track] = {}       # one track per player, keyed by owner
        self.key = DEFAULT_KEY
        self.bars = DEFAULT_BARS
        self.session_bpm: float = 0.0            # set by the first rendered loop
        self.loop_secs: float = 0.0
        self.lsamp: int = 0
        self.engine = "mrt2"                      # model used for renders: mrt2 | sa3
        self.pos = 0                             # master mixer position (samples into L)
        self.alive = True
        self.mixing = False
        self.mixer: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.audiotool_project: str | None = None   # all stems go to ONE project

    def roster(self) -> list[dict]:
        return [{"name": c.name, "host": (c.name == self.host),
                 "has_track": c.name in self.tracks} for c in self.clients.values()]


class LooperHub:
    """Manages multiplayer looper rooms: one server-side mix per room."""

    def __init__(self, engine, audiotool=None, transmuter=None):
        self._engine = engine                    # looper.LooperEngine
        self._audiotool = audiotool              # AudiotoolBridge | None
        self._transmuter = transmuter            # transmute.Transmuter | None (SA3 finisher)
        self._rooms: dict[str, LooperRoom] = {}

    # ── connection lifecycle ─────────────────────────────────────────────────
    async def handle(self, ws) -> None:
        from fastapi import WebSocketDisconnect  # noqa: PLC0415
        await ws.accept()
        room: LooperRoom | None = None
        name: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "create":
                    room, name = self._create(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": True, "instruments": instrument_list()})
                    await self._broadcast_room(room)
                elif action == "join":
                    room, name = self._join(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": (name == room.host), "instruments": instrument_list()})
                    await self._broadcast_room(room)
                elif not room or not name:
                    continue
                elif action == "engine":
                    room.engine = "sa3" if msg.get("engine") == "sa3" else "mrt2"
                    await self._broadcast_room(room)
                elif action == "pick":
                    await self._pick(room, name, msg.get("instrument"))
                elif action == "prompt":
                    await self._reprompt(room, name, msg.get("text"))
                elif action == "reroll":
                    await self._render_track(room, name)
                elif action == "audiotool":
                    await self._export_audiotool(room, name, finish=bool(msg.get("finish")))
                elif action == "newtake":
                    self._newtake(room, name)
                    await self._broadcast_room(room)
                elif action in ("vol", "pan", "mute", "solo"):
                    self._mix_ctl(room, name, action, msg)
                    await self._broadcast_room(room)
                elif action == "remove":
                    self._remove_track(room, name)
                    await self._broadcast_room(room)
                elif action == "stop":
                    break
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            if room and name:
                await self._leave(room, name)

    # ── room ops ──────────────────────────────────────────────────────────────
    def _create(self, ws, msg) -> tuple[LooperRoom, str]:
        name = (msg.get("name") or "Host").strip()[:24] or "Host"
        code = self._gen_code()
        room = LooperRoom(code, host=name)
        room.loop = asyncio.get_running_loop()
        self._rooms[code] = room
        self._add_client(room, ws, name)
        return room, name

    def _join(self, ws, msg) -> tuple[LooperRoom, str]:
        code = (msg.get("code") or "").strip().upper()
        room = self._rooms.get(code)
        if room is None or not room.alive:
            raise ValueError("no room with that code")
        name = (msg.get("name") or "Player").strip()[:24] or "Player"
        base, i = name, 2
        while name in room.clients:
            name = f"{base} {i}"; i += 1
        self._add_client(room, ws, name)
        return room, name

    def _add_client(self, room: LooperRoom, ws, name: str) -> None:
        c = _Client(ws, name)
        room.clients[name] = c
        c.pump = asyncio.create_task(self._pump(c))

    async def _leave(self, room: LooperRoom, name: str) -> None:
        room.tracks.pop(name, None)
        c = room.clients.pop(name, None)
        if c and c.pump:
            c.pump.cancel()
        if not room.clients:
            room.alive = False
            room.mixing = False
            self._rooms.pop(room.code, None)
        else:
            await self._broadcast_room(room)

    # ── instruments + rendering ───────────────────────────────────────────────
    async def _pick(self, room: LooperRoom, name: str, instrument) -> None:
        if instrument not in INSTRUMENTS:
            return
        t = room.tracks.get(name)
        if t is None:
            t = Track(name, instrument)
            room.tracks[name] = t
        else:
            t.instrument = instrument
        await self._render_track(room, name)

    async def _reprompt(self, room: LooperRoom, name: str, text) -> None:
        t = room.tracks.get(name)
        if t is None:
            return
        t.custom_prompt = (text or "").strip()
        await self._render_track(room, name)

    async def _render_track(self, room: LooperRoom, name: str) -> None:
        t = room.tracks.get(name)
        if t is None or t.rendering:
            return
        t.rendering = True
        await self._broadcast_room(room)
        target_bpm = room.session_bpm if room.session_bpm > 0 else 0.0
        try:
            clip, info = await asyncio.to_thread(
                self._engine.render, t.instrument, room.key, target_bpm, room.bars,
                t.custom_prompt or None, room.engine)
            buf = await asyncio.to_thread(read_buffer, clip.wav_path)
        except Exception as e:  # noqa: BLE001
            t.rendering = False
            self._send_to(room, name, {"type": "error", "error": f"render failed: {e}"})
            await self._broadcast_room(room)
            return

        if room.lsamp == 0:                       # first loop sets the session grid
            room.session_bpm = info["bpm"]
            room.loop_secs = info["loop_secs"]
            room.lsamp = int(round(info["loop_secs"] * SR))
        t.windows = info["windows"]
        t.tempo_detected = info.get("tempo_detected", 0.0)
        t.stretch = info.get("stretch", 1.0)
        t.rendering = False
        if t.buf is None:                         # first buffer: play immediately
            t.buf = buf
            t.win = 0
            t.ready = True
        else:                                     # swap in at the next loop boundary
            t.pending = buf
            t.pending_win = 0
        self._ensure_mixer(room)
        await self._broadcast_room(room)

    def _newtake(self, room: LooperRoom, name: str) -> None:
        t = room.tracks.get(name)
        if t and t.ready and t.windows > 1:
            t.pending_win = (t.win + 1) % t.windows

    def _mix_ctl(self, room: LooperRoom, name: str, action, msg) -> None:
        t = room.tracks.get(name)
        if t is None:
            return
        if action == "vol":
            t.vol = max(0.0, min(1.5, float(msg.get("value", t.vol))))
        elif action == "pan":
            t.pan = max(-1.0, min(1.0, float(msg.get("value", t.pan))))
        elif action == "mute":
            t.muted = bool(msg.get("on"))
        elif action == "solo":
            t.solo = bool(msg.get("on"))

    def _remove_track(self, room: LooperRoom, name: str) -> None:
        room.tracks.pop(name, None)

    # ── stems -> Audiotool ──────────────────────────────────────────────────────
    async def _export_audiotool(self, room: LooperRoom, name: str, finish: bool = False) -> None:
        """Send each ready loop to Audiotool as its own track, all in ONE project,
        so a room jam becomes a real multitrack project. Each stem is the exact
        current loop window, tagged with the session tempo so they line up.

        finish=True runs each stem through SA3 first (Transmute, low restyle) so the
        parts arrive at SA3 fidelity while STILL landing as separate tracks. This is
        the 'MRT2 performs, SA3 produces' pipeline, per-stem so multitrack survives.
        """
        if self._audiotool is None or not getattr(self._audiotool, "configured", False):
            self._send_to(room, name, {"type": "audiotool", "ok": False,
                                       "error": "Audiotool is not connected on this server."})
            return
        if finish and self._transmuter is None:
            self._send_to(room, name, {"type": "audiotool", "ok": False,
                                       "error": "SA3 is not available on this server."})
            return
        ready = [t for t in room.tracks.values() if t.ready and t.buf is not None and room.lsamp]
        if not ready:
            self._send_to(room, name, {"type": "audiotool", "ok": False,
                                       "error": "No loops to send yet."})
            return
        verb = "Producing with SA3 + sending" if finish else "Sending"
        await self._broadcast(room, {"type": "audiotool", "ok": True,
                                     "status": f"{verb} {len(ready)} stems to Audiotool..."})
        chunks = max(1, round(room.loop_secs / 0.8))
        url = ""
        for t in ready:
            inst = INSTRUMENTS.get(t.instrument, {})
            label = inst.get("label", t.instrument)
            seg = t.buf[t.win * room.lsamp:(t.win + 1) * room.lsamp]
            path = os.path.join(self._engine._clips_dir, f"stem_{uuid.uuid4().hex[:10]}.wav")
            try:
                await asyncio.to_thread(MRTWorker._write_wav, path, seg.astype(np.float32))
                if finish:                          # SA3 per-stem: restyle low to keep the part
                    await self._broadcast(room, {"type": "audiotool", "ok": True,
                                                 "status": f"Producing {label} with SA3 (slow)..."})
                    prompt = t.custom_prompt or inst.get("prompt", label)
                    clip = await asyncio.to_thread(
                        self._transmuter.transmute, path, prompt,
                        init_noise_level=0.35, chunks=chunks)
                    path = clip.wav_path
                res = await asyncio.to_thread(
                    self._audiotool.send_clip, path, f"{label} - {t.owner}",
                    project=room.audiotool_project, bpm=int(round(room.session_bpm)) or None)
                room.audiotool_project = res.get("project_name") or room.audiotool_project
                url = res.get("project_url", "") or url
                await self._broadcast(room, {"type": "audiotool", "ok": True, "status": f"Sent {label}"})
            except Exception as e:  # noqa: BLE001
                self._send_to(room, name, {"type": "audiotool", "ok": False, "error": f"Audiotool: {e}"})
                return
        await self._broadcast(room, {"type": "audiotool", "ok": True, "done": True,
                                     "url": url, "count": len(ready), "by": name,
                                     "finished": finish})

    # ── the mixer (one thread per room) ────────────────────────────────────────
    def _ensure_mixer(self, room: LooperRoom) -> None:
        if room.mixing or room.lsamp == 0:
            return
        room.mixing = True
        room.mixer = threading.Thread(target=self._mix_run, args=(room,), daemon=True)
        room.mixer.start()

    def _mix_run(self, room: LooperRoom) -> None:
        t0 = time.monotonic()
        produced = 0.0
        while room.alive and room.mixing:
            lsamp = room.lsamp
            if lsamp == 0:
                time.sleep(0.05); continue
            if room.pos == 0:                     # apply boundary swaps on the downbeat
                for t in room.tracks.values():
                    if t.pending is not None:
                        t.buf = t.pending; t.pending = None; t.windows = t.buf.shape[0] // lsamp or 1; t.ready = True
                    if t.pending_win is not None:
                        t.win = min(t.pending_win, max(0, t.windows - 1)); t.pending_win = None

            n = min(MIX_CHUNK, lsamp - room.pos)
            mix = np.zeros((n, 2), dtype=np.float32)
            tracks = list(room.tracks.values())
            solo_any = any(t.solo for t in tracks if t.ready)
            for t in tracks:
                if not t.ready or t.buf is None:
                    continue
                g = 0.0 if (solo_any and not t.solo) else (0.0 if t.muted else t.vol)
                if g <= 0.0:
                    continue
                base = t.win * lsamp + room.pos
                seg = t.buf[base:base + n]
                if seg.shape[0] < n:
                    continue
                ang = (t.pan + 1.0) * 0.25 * np.pi      # equal-power pan
                mix[:, 0] += seg[:, 0] * g * np.cos(ang)
                mix[:, 1] += seg[:, 1] * g * np.sin(ang)

            mix = MRTWorker._soft_limit(mix)
            pcm = (np.clip(mix, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            if room.loop is not None:
                room.loop.call_soon_threadsafe(self._fanout_audio, room, pcm)

            room.pos += n
            if room.pos >= lsamp:
                room.pos = 0
            produced += n / SR
            ahead = (t0 + produced) - time.monotonic()
            if ahead > MIX_LEAD:
                time.sleep(ahead - MIX_LEAD)

    # ── audio fan-out + sending (shared spine) ─────────────────────────────────
    def _fanout_audio(self, room: LooperRoom, pcm: bytes) -> None:
        for c in room.clients.values():
            if c.q.qsize() < AUDIO_BACKLOG:
                c.q.put_nowait(pcm)

    async def _pump(self, c: _Client) -> None:
        try:
            while True:
                item = await c.q.get()
                if isinstance(item, tuple) and item and item[0] == "j":
                    await c.ws.send_json(item[1])
                else:
                    await c.ws.send_bytes(item)
        except Exception:  # noqa: BLE001
            pass

    async def _send(self, ws, obj: dict) -> None:
        try:
            await ws.send_json(obj)
        except Exception:  # noqa: BLE001
            pass

    def _send_to(self, room: LooperRoom, name: str, obj: dict) -> None:
        c = room.clients.get(name)
        if c:
            c.q.put_nowait(("j", obj))

    async def _broadcast(self, room: LooperRoom, obj: dict) -> None:
        for c in list(room.clients.values()):
            c.q.put_nowait(("j", obj))

    async def _broadcast_room(self, room: LooperRoom) -> None:
        await self._broadcast(room, {"type": "room", "code": room.code, "host": room.host,
                                     "players": room.roster(),
                                     "tracks": [t.meta() for t in room.tracks.values()],
                                     "key": room.key, "bars": room.bars, "engine": room.engine,
                                     "session_bpm": round(room.session_bpm, 1),
                                     "loop_secs": round(room.loop_secs, 3)})

    @staticmethod
    def _gen_code() -> str:
        return "".join(random.choices(string.ascii_uppercase, k=4))
