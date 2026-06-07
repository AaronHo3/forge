"""
ensemble.py - the Ensemble Room: a live band with no instruments.

A host opens a room; players join and each claims a SEAT that drives one musical
dimension of ONE shared, continuously-generated MRT2 stream:

  - Harmonist   the chords / progression   (JamState.chord, key)
  - Texturist   the lead sound / genre      (JamState.prompt_a)
  - Atmosphere  a second layer + how much   (JamState.prompt_b, blend)
  - Drive       energy + rhythm             (JamState.density, drums)

Everyone hears the same evolving track in real time. The group performs together;
good takes are captured and sent to AudioTool, where every take from one session
lands in the SAME project so the room becomes a writers' room and AudioTool the
studio.

This is the thing only a continuous, multi-input, steerable model can be: a
real-time musical performance shared by several people. Taste and direction are
the skill, so beginners and advanced musicians use the same surface.

ARCHITECTURE
- One MLX model = one live stream, so a room broadcasts ONE jam to N listeners
  (worker.run_jam reads the room's JamState, emits PCM, we fan it out). Shared
  spine with classroom.py.
- Four players writing four independent fields of one JamState never collide; the
  model fuses their intents. Each control is routed through a per-seat allowlist.
- All sends to a client (audio bytes + JSON) go through ONE per-client queue.
  Audio is dropped under backpressure; control/state JSON is not.
"""

from __future__ import annotations

import asyncio
import random
import string

from .harmony import chord_name
from .jam import JamState

AUDIO_BACKLOG = 8       # per-client audio queue depth before we drop (keeps latency low)

# Seats and the control actions each is allowed to send. Routing through this map
# is what makes a seat mean something instead of a free-for-all.
SEATS = ("harmonist", "texturist", "atmosphere", "drive")
SEAT_ACTIONS = {
    "harmonist": {"chord", "key", "free"},
    "texturist": {"prompt_a"},
    "atmosphere": {"prompt_b", "blend"},
    "drive": {"density", "drums", "chaos", "focus"},
}
SEAT_INFO = {
    "harmonist": {"label": "Harmonist", "drives": "the chords, the key, and 7th color"},
    "texturist": {"label": "Texturist", "drives": "the lead sound and genre"},
    "atmosphere": {"label": "Atmosphere", "drives": "a second layer and how much it blends in"},
    "drive": {"label": "Drive", "drives": "energy, drums, chaos and focus"},
}

# Challenge mode: the room is handed a creative brief and steers the live stream
# toward it together, then the take is captured. Co-op, constraint-driven.
BRIEFS = [
    "a rain-soaked midnight drive", "sunrise over a quiet city",
    "a triumphant final boss battle", "a tense spy infiltration",
    "a warm summer block party", "drifting through deep space",
    "a heartbreak slow dance", "an underground club at 2am",
    "a hopeful new beginning", "a haunted forest at dusk",
    "a victory lap, top down, windows open", "the calm before a storm",
]
CHALLENGE_SECS = 75.0

# Quick texture palette offered to Texturist / Atmosphere (free text also allowed).
TEXTURES = [
    "warm electric piano", "lush analog synth pads", "cinematic strings",
    "gritty electric guitar", "dusty lo-fi keys", "bright marimba and bells",
    "deep dub bass", "brushed jazz drums and upright bass", "ethereal choir",
    "retro funk clavinet", "ambient guitar swells", "pulsing arpeggiated synth",
]

DEFAULT_PROMPT_A = "warm session band, electric piano, upright bass, soft drums"


def _instr(s: str) -> str:
    s = (s or "").strip()
    return s if "instrumental" in s.lower() else (s + ", instrumental")


class _Client:
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name
        self.seat: str | None = None
        self.q: asyncio.Queue = asyncio.Queue()   # bytes (audio) or ("j", dict) (json)
        self.pump: asyncio.Task | None = None


class EnsembleRoom:
    def __init__(self, code: str, host: str):
        self.code = code
        self.host = host
        self.clients: dict[str, _Client] = {}
        self.seats: dict[str, str | None] = {s: None for s in SEATS}   # seat -> player name
        self.phase = "lobby"                         # lobby | playing
        self.state: JamState | None = None
        self.alive = True
        self.audio_started: asyncio.Event = asyncio.Event()
        self.last_clip = None                        # most recent captured Clip
        self.audiotool_project: str | None = None    # all takes go to ONE project
        self.challenge: dict | None = None           # {brief} while a challenge is running
        self.reactions = 0                           # 👍 on the current/last challenge take

    def roster(self) -> list[dict]:
        return [{"name": c.name, "seat": c.seat, "host": (c.name == self.host)}
                for c in self.clients.values()]


class EnsembleHub:
    """Manages Ensemble rooms and the one shared MRT2 stream per active room."""

    def __init__(self, worker_getter, capture_fn, to_json_fn, audiotool=None):
        self._get_worker = worker_getter      # -> shared MRTWorker
        self._capture = capture_fn            # (state, seconds) -> Clip (saves to library)
        self._to_json = to_json_fn            # (Clip) -> dict
        self._audiotool = audiotool           # AudiotoolBridge | None
        self._rooms: dict[str, EnsembleRoom] = {}

    # ── connection lifecycle ─────────────────────────────────────────────────
    async def handle(self, ws) -> None:
        from fastapi import WebSocketDisconnect  # noqa: PLC0415
        await ws.accept()
        room: EnsembleRoom | None = None
        name: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "create":
                    room, name = self._create(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": True, "seatInfo": SEAT_INFO, "textures": TEXTURES})
                    await self._broadcast_room(room)
                elif action == "join":
                    room, name = self._join(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": (name == room.host), "seatInfo": SEAT_INFO,
                                          "textures": TEXTURES})
                    await self._broadcast_room(room)
                    if room.phase == "playing":
                        await self._send(ws, {"type": "started"})
                        await self._send(ws, self._state_msg(room))
                elif not room or not name:
                    continue
                elif action == "seat":
                    self._take_seat(room, name, msg.get("seat"))
                    await self._broadcast_room(room)
                elif action == "unseat":
                    self._leave_seat(room, name)
                    await self._broadcast_room(room)
                elif action == "start" and name == room.host and room.phase == "lobby":
                    await self._start(room)
                elif action == "challenge" and name == room.host:
                    await self._start_challenge(room)
                elif action == "react":
                    self._react(room)
                elif action == "save":
                    await self._capture_take(room, name, msg.get("seconds"))
                elif action == "audiotool":
                    await self._to_audiotool(room, name)
                elif action == "stop":
                    break
                else:
                    self._control(room, name, action, msg)   # seat-gated steering
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            if room and name:
                await self._leave(room, name)

    # ── room ops ──────────────────────────────────────────────────────────────
    def _create(self, ws, msg) -> tuple[EnsembleRoom, str]:
        name = (msg.get("name") or "Host").strip()[:24] or "Host"
        code = self._gen_code()
        room = EnsembleRoom(code, host=name)
        self._rooms[code] = room
        self._add_client(room, ws, name)
        return room, name

    def _join(self, ws, msg) -> tuple[EnsembleRoom, str]:
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

    def _add_client(self, room: EnsembleRoom, ws, name: str) -> None:
        c = _Client(ws, name)
        room.clients[name] = c
        c.pump = asyncio.create_task(self._pump(c))

    def _take_seat(self, room: EnsembleRoom, name: str, seat) -> None:
        if seat not in SEATS or room.seats.get(seat):     # taken or invalid
            return
        self._leave_seat(room, name)                       # one seat per player
        room.seats[seat] = name
        room.clients[name].seat = seat

    def _leave_seat(self, room: EnsembleRoom, name: str) -> None:
        c = room.clients.get(name)
        if c and c.seat:
            room.seats[c.seat] = None
            c.seat = None

    async def _leave(self, room: EnsembleRoom, name: str) -> None:
        self._leave_seat(room, name)
        c = room.clients.pop(name, None)
        if c and c.pump:
            c.pump.cancel()
        if not room.clients:
            room.alive = False
            if room.state:
                room.state.running = False
            self._rooms.pop(room.code, None)
        else:
            await self._broadcast_room(room)

    # ── the shared stream ─────────────────────────────────────────────────────
    async def _start(self, room: EnsembleRoom) -> None:
        room.phase = "playing"
        loop = asyncio.get_running_loop()
        room.state = JamState(prompt_a=_instr(DEFAULT_PROMPT_A), density=0.4, drums=True)

        def emit(pcm: bytes) -> None:
            loop.call_soon_threadsafe(self._fanout_audio, room, pcm)

        self._get_worker().submit_jam(room.state, emit)
        await self._broadcast(room, {"type": "started"})
        await self._broadcast(room, self._state_msg(room))

    def _control(self, room: EnsembleRoom, name: str, action, msg) -> None:
        """Apply a steering message, gated by the player's seat."""
        c = room.clients.get(name)
        st = room.state
        if c is None or st is None or not c.seat:
            return
        if action not in SEAT_ACTIONS.get(c.seat, ()):    # not your instrument
            return
        if action == "chord":
            root = int(msg.get("root", 0)) % 12
            quality = str(msg.get("quality", "maj"))
            st.set_chord((root, quality))
        elif action == "free":
            st.set_chord(None)
        elif action == "key":
            st.set_prompt(key=msg.get("key"))
        elif action == "prompt_a":
            st.set_prompt(prompt_a=_instr(msg.get("text") or DEFAULT_PROMPT_A))
        elif action == "prompt_b":
            txt = (msg.get("text") or "").strip()
            st.set_prompt(prompt_b=(_instr(txt) if txt else ""))
        elif action == "blend":
            st.set_params(blend=msg.get("value"))
        elif action == "density":
            st.set_params(density=msg.get("value"))
        elif action == "drums":
            st.set_params(drums=bool(msg.get("on")))
        elif action == "chaos":
            st.set_params(temperature=msg.get("value"))
        elif action == "focus":
            st.set_params(top_k=msg.get("value"))
        # reflect the change to the whole room so everyone sees the ensemble move
        asyncio.create_task(self._broadcast(room, self._state_msg(room)))

    def _state_msg(self, room: EnsembleRoom) -> dict:
        st = room.state
        chord = None
        if st and st.chord:
            chord = chord_name(st.chord[0], st.chord[1])
        return {"type": "state",
                "chord": chord, "key": (st.key if st else None),
                "prompt_a": (st.prompt_a if st else ""),
                "prompt_b": (st.prompt_b if st else ""),
                "blend": (st.blend if st else 0.0),
                "density": (st.density if st else 0.0),
                "drums": (st.drums if st else False)}

    # ── capture + AudioTool ───────────────────────────────────────────────────
    async def _capture_take(self, room: EnsembleRoom, name: str, seconds) -> None:
        if room.state is None:
            return
        try:
            clip = await asyncio.to_thread(self._capture, room.state, seconds)
            room.last_clip = clip
            await self._broadcast(room, {"type": "saved", "by": name,
                                         "clip": self._to_json(clip)})
        except ValueError as e:
            self._send_to(room, name, {"type": "error", "error": str(e)})

    async def _to_audiotool(self, room: EnsembleRoom, name: str) -> None:
        if self._audiotool is None or not getattr(self._audiotool, "configured", False):
            self._send_to(room, name, {"type": "error",
                                       "error": "Audiotool is not connected on this server."})
            return
        if room.last_clip is None:
            self._send_to(room, name, {"type": "error", "error": "Capture a take first."})
            return
        clip = room.last_clip
        display = clip.name or f"Ensemble {room.code}"
        try:
            res = await asyncio.to_thread(
                self._audiotool.send_clip, clip.wav_path, display,
                project=room.audiotool_project)
            room.audiotool_project = res.get("project_name") or room.audiotool_project
            await self._broadcast(room, {"type": "audiotool", "ok": True, "by": name,
                                         "url": res.get("project_url", "")})
        except Exception as e:  # noqa: BLE001
            self._send_to(room, name, {"type": "error", "error": f"Audiotool: {e}"})

    # ── Challenge mode ────────────────────────────────────────────────────────
    async def _start_challenge(self, room: EnsembleRoom) -> None:
        """Hand the room a creative brief, run a timer, then auto-capture the take."""
        if room.phase != "playing" or room.challenge is not None:
            return
        brief = random.choice(BRIEFS)
        room.challenge = {"brief": brief}
        room.reactions = 0
        await self._broadcast(room, {"type": "challenge", "brief": brief,
                                     "seconds": CHALLENGE_SECS})
        asyncio.create_task(self._challenge_timer(room, brief))

    async def _challenge_timer(self, room: EnsembleRoom, brief: str) -> None:
        await asyncio.sleep(CHALLENGE_SECS)
        if not room.alive or room.challenge is None:
            return
        room.challenge = None
        clip_json = None
        try:
            clip = await asyncio.to_thread(self._capture, room.state, CHALLENGE_SECS)
            room.last_clip = clip
            clip_json = self._to_json(clip)
        except Exception:  # noqa: BLE001 - capture can fail if nothing recorded yet
            pass
        await self._broadcast(room, {"type": "challenge_end", "brief": brief,
                                     "clip": clip_json, "reactions": room.reactions})

    def _react(self, room: EnsembleRoom) -> None:
        room.reactions += 1
        asyncio.create_task(self._broadcast(room, {"type": "react", "count": room.reactions}))

    # ── audio fan-out + sending (shared spine with classroom) ─────────────────
    def _fanout_audio(self, room: EnsembleRoom, pcm: bytes) -> None:
        if not room.audio_started.is_set():
            room.audio_started.set()
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

    def _send_to(self, room: EnsembleRoom, name: str, obj: dict) -> None:
        c = room.clients.get(name)
        if c:
            c.q.put_nowait(("j", obj))

    async def _broadcast(self, room: EnsembleRoom, obj: dict) -> None:
        for c in list(room.clients.values()):
            c.q.put_nowait(("j", obj))

    async def _broadcast_room(self, room: EnsembleRoom) -> None:
        await self._broadcast(room, {"type": "room", "code": room.code, "host": room.host,
                                     "phase": room.phase, "players": room.roster(),
                                     "seats": room.seats})

    @staticmethod
    def _gen_code() -> str:
        return "".join(random.choices(string.ascii_uppercase, k=4))
