"""
classroom.py - the Live Music Classroom (a game show for the ear, on a live MRT2 stream).

A host opens a room; players join (by code / QR). Everyone hears the SAME live
MRT2 stream. The class is a sequence of varied mini-games, each teaching a
different listening + prompting skill:

  - DETECTIVE  "Which prompt is the AI playing?"   -> reading sound as language
  - CHANGED    "What just changed?"                -> hearing musical dimensions
  - GENRE      "Name the genre it morphed into"    -> genre by ear

Each round picks one game, choreographs the shared stream, then the room races to
answer. Faster + on a streak = more points. Live leaderboard.

This is the social/interactive face of the education thesis: people learn to hear
music AND to prompt for it, together, in real time, with no instrument.

ARCHITECTURE
- One MLX model = one live stream, so a room broadcasts ONE jam to N listeners
  (worker.run_jam reads the room's JamState, emits PCM, we fan it out).
- All sends to a client (audio bytes + JSON control) go through ONE per-client
  queue, because a WebSocket can't be written from two tasks at once. Audio is
  dropped under backpressure; control messages are not.
- The round orchestrator is an asyncio task that drives the shared JamState. The
  worker thread bridges to the loop via call_soon_threadsafe.
- Prompt changes crossfade through `blend` (prompt_b ramp, then commit) so the
  live stream morphs cleanly instead of garbling on a hard prompt swap.
"""

from __future__ import annotations

import asyncio
import random
import string
import time

from .jam import JamState

REVEAL_SECS = 6.0       # how long the reveal/leaderboard shows
BASELINE_SECS = 4.0     # hear the "before" / setup state before the question opens
MAX_ROUNDS = 8
AUDIO_BACKLOG = 8       # per-client audio queue depth before we drop (keeps latency low)

XFADE_SECS = 1.6        # prompt crossfade duration
XFADE_STEPS = 8

BASE_POINTS = 500       # for a correct answer
SPEED_POINTS = 500      # max additional, scaled by how fast
STREAK_STEP = 100       # per consecutive correct (capped)
STREAK_CAP = 5


def _instr(s: str) -> str:
    s = (s or "").strip()
    return s if "instrumental" in s.lower() else (s + ", instrumental")


# ── Prompt banks (curated to be audibly distinct, so distractors are fair) ──────
# Full multi-attribute prompts: this is what "prompting" actually looks like.
DETECTIVE_PROMPTS = [
    ("Lo-fi hip hop", "lo-fi hip hop beat, dusty drums, mellow Rhodes"),
    ("Epic orchestra", "epic orchestral strings and brass, cinematic, dramatic"),
    ("Ambient dreamscape", "ambient synth pads, slow, dreamy, ethereal"),
    ("Driving techno", "driving techno, four on the floor, pulsing bass"),
    ("Funk groove", "funky electric bass, wah guitar, tight groove"),
    ("Chiptune", "8-bit chiptune, playful, retro video game"),
    ("Spanish guitar", "flamenco spanish guitar, passionate, rhythmic"),
    ("Smooth jazz", "smooth jazz saxophone, late night, laid back"),
]

# Single-genre prompts: name-that-tune for genres.
GENRE_PROMPTS = [
    ("Reggae", "reggae groove, offbeat guitar skank, warm bass"),
    ("Country", "country acoustic guitar, banjo, twangy"),
    ("Disco", "disco strings, four on the floor, funky bass"),
    ("Metal", "heavy metal, distorted guitars, aggressive"),
    ("Bossa nova", "bossa nova, soft nylon guitar, brushed drums"),
    ("Trap", "trap beat, booming 808 bass, fast hi-hats"),
    ("Classical", "classical solo piano, romantic, expressive"),
    ("Blues", "slow blues, electric guitar, soulful bends"),
]

_WARM = "warm mellow analog keys, soft and dark"
_BRIGHT = "bright crisp shimmering bells, airy and sparkling"


def _shuffle(true_label: str, distractors: list[str]) -> tuple[list[str], int]:
    opts = [true_label] + distractors
    random.shuffle(opts)
    return opts, opts.index(true_label)


# ── Round builders. Each returns a dict the orchestrator + client understand. ──
#   kind / concept     -> shown in the UI ("what skill is this round teaching")
#   question / sub      -> the prompt to the player
#   options / correct   -> the quiz
#   explain / reveal    -> shown on the answer reveal
#   pre / change        -> stream choreography (state to hear, optional change)
#   pre_xfade / change_xfade -> crossfade the prompt change (clean morph)
#   seconds             -> answer window for this round

def _r_detective() -> dict:
    bank = random.sample(DETECTIVE_PROMPTS, 4)
    (true_label, true_text), *rest = bank
    options, correct = _shuffle(true_label, [lbl for lbl, _ in rest])
    return {
        "kind": "detective", "concept": "Prompt to sound",
        "question": "Which prompt is the AI playing?",
        "sub": "match what you hear to the words that made it",
        "options": options, "correct": correct, "seconds": 16.0,
        "explain": f"It was {true_label}.",
        "reveal": f'Prompt: "{true_text}"',
        "pre": {"prompt_a": true_text, "blend": 0.0, "density": 0.55, "drums": True},
        "pre_xfade": True, "change": None,
    }


def _r_genre() -> dict:
    bank = random.sample(GENRE_PROMPTS, 4)
    (true_label, true_text), *rest = bank
    options, correct = _shuffle(true_label, [lbl for lbl, _ in rest])
    return {
        "kind": "genre", "concept": "Genre by ear",
        "question": "Name the genre it morphed into",
        "sub": "the stream is sliding into a new style",
        "options": options, "correct": correct, "seconds": 14.0,
        "explain": f"That was {true_label}.",
        "reveal": f'Prompt: "{true_text}"',
        "pre": {"prompt_a": "warm neutral electric piano, steady groove",
                "blend": 0.0, "density": 0.45, "drums": True},
        "pre_xfade": False,
        "change": {"prompt_a": true_text}, "change_xfade": True,
    }


# what-changed events: (description, before-state, after-state, after crossfades?)
_CHANGED_EVENTS = [
    ("Busier and denser", {"density": 0.18}, {"density": 0.85}, False),
    ("Sparser, more space", {"density": 0.85}, {"density": 0.18}, False),
    ("Drums came in", {"drums": False}, {"drums": True}, False),
    ("Drums dropped out", {"drums": True}, {"drums": False}, False),
    ("Brighter, more highs", {"prompt_a": _WARM}, {"prompt_a": _BRIGHT}, True),
    ("Warmer and darker", {"prompt_a": _BRIGHT}, {"prompt_a": _WARM}, True),
]


def _r_changed() -> dict:
    true_desc, before, after, xfade = random.choice(_CHANGED_EVENTS)
    others = [d for (d, *_rest) in _CHANGED_EVENTS if d != true_desc]
    options, correct = _shuffle(true_desc, random.sample(others, 3))
    base = {"blend": 0.0, "density": 0.45, "drums": True, "prompt_a": _WARM}
    pre = {**base, **before}
    return {
        "kind": "changed", "concept": "Hearing dimensions",
        "question": "What just changed?",
        "sub": "one quality of the music shifted",
        "options": options, "correct": correct, "seconds": 11.0,
        "explain": f"It got {true_desc.lower()}.", "reveal": "",
        "pre": pre, "pre_xfade": False,
        "change": after, "change_xfade": xfade,
    }


_BUILDERS = [_r_detective, _r_genre, _r_changed]


def _make_round() -> dict:
    return random.choice(_BUILDERS)()


def _apply(state: JamState, d: dict | None) -> None:
    """Apply a non-crossfaded state dict to the shared stream."""
    if not d:
        return
    p = {k: d[k] for k in ("prompt_a", "prompt_b", "key") if k in d}
    if p.get("prompt_a"):
        p["prompt_a"] = _instr(p["prompt_a"])
    if p.get("prompt_b"):
        p["prompt_b"] = _instr(p["prompt_b"])
    if p:
        state.set_prompt(**p)
    c = {k: d[k] for k in ("blend", "density", "drums") if k in d}
    if c:
        state.set_params(**c)


class _Client:
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name
        self.q: asyncio.Queue = asyncio.Queue()   # holds bytes (audio) or ("j", dict) (json)
        self.pump: asyncio.Task | None = None


class ClassRoom:
    def __init__(self, code: str, host: str):
        self.code = code
        self.host = host
        self.clients: dict[str, _Client] = {}
        self.scores: dict[str, int] = {}
        self.streak: dict[str, int] = {}
        self.phase = "lobby"                        # lobby | playing | done
        self.state: JamState | None = None          # the shared jam stream
        self.round: dict | None = None              # current round
        self.alive = True
        self.audio_started: asyncio.Event = asyncio.Event()

    def leaderboard(self) -> list[dict]:
        return [{"name": n, "score": s, "streak": self.streak.get(n, 0)}
                for n, s in sorted(self.scores.items(), key=lambda kv: -kv[1])]


class ClassroomHub:
    """Manages classroom rooms and the one shared MRT2 stream per active room."""

    def __init__(self, worker_getter):
        self._get_worker = worker_getter           # callable -> the shared MRTWorker
        self._rooms: dict[str, ClassRoom] = {}

    # ── connection lifecycle ─────────────────────────────────────────────────
    async def handle(self, ws) -> None:
        from fastapi import WebSocketDisconnect  # noqa: PLC0415
        await ws.accept()
        room: ClassRoom | None = None
        name: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "create":
                    room, name = self._create(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "host": True, "name": name})
                    await self._broadcast_lobby(room)
                elif action == "join":
                    room, name = self._join(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "host": (name == room.host), "name": name})
                    await self._broadcast_lobby(room)
                elif action == "start" and room and name == room.host and room.phase == "lobby":
                    asyncio.create_task(self._run(room))
                elif action == "answer" and room and room.round:
                    self._answer(room, name, msg.get("choice"))
                # (rounds auto-advance; no manual next needed)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            if room and name:
                await self._leave(room, name)

    # ── room ops ──────────────────────────────────────────────────────────────
    def _create(self, ws, msg) -> tuple[ClassRoom, str]:
        name = (msg.get("name") or "Host").strip()[:24] or "Host"
        code = self._gen_code()
        room = ClassRoom(code, host=name)
        self._rooms[code] = room
        self._add_client(room, ws, name)
        return room, name

    def _join(self, ws, msg) -> tuple[ClassRoom, str]:
        code = (msg.get("code") or "").strip().upper()
        room = self._rooms.get(code)
        if room is None or not room.alive:
            raise ValueError("no room with that code")
        name = (msg.get("name") or "Player").strip()[:24] or "Player"
        base, i = name, 2
        while name in room.clients:               # de-dupe names
            name = f"{base} {i}"; i += 1
        self._add_client(room, ws, name)
        return room, name

    def _add_client(self, room: ClassRoom, ws, name: str) -> None:
        c = _Client(ws, name)
        room.clients[name] = c
        room.scores.setdefault(name, 0)
        room.streak.setdefault(name, 0)
        c.pump = asyncio.create_task(self._pump(c))

    async def _leave(self, room: ClassRoom, name: str) -> None:
        c = room.clients.pop(name, None)
        if c and c.pump:
            c.pump.cancel()
        if not room.clients:                       # last one out -> tear down
            room.alive = False
            if room.state:
                room.state.running = False
            self._rooms.pop(room.code, None)
        else:
            await self._broadcast_lobby(room)

    # ── round orchestrator ──────────────────────────────────────────────────
    async def _run(self, room: ClassRoom) -> None:
        room.phase = "playing"
        loop = asyncio.get_running_loop()
        room.state = JamState(prompt_a=_instr("warm neutral electric piano, steady groove"),
                              density=0.4)

        def emit(pcm: bytes) -> None:              # worker thread -> loop
            loop.call_soon_threadsafe(self._fanout_audio, room, pcm)

        self._get_worker().submit_jam(room.state, emit)
        await self._broadcast(room, {"type": "started"})
        try:
            await asyncio.wait_for(room.audio_started.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(1.0)

        for rnd in range(1, MAX_ROUNDS + 1):
            if not room.alive:
                break
            ch = _make_round()

            # 1) set up the state players hear during the listen window
            await self._setup(room, ch["pre"], crossfade=ch.get("pre_xfade"))
            await self._broadcast(room, {"type": "listen", "round": rnd, "total": MAX_ROUNDS,
                                         "kind": ch["kind"], "concept": ch["concept"]})
            await asyncio.sleep(BASELINE_SECS)
            if not room.alive:
                break

            # 2) apply the change (for "what changed" / genre morph) right as the question opens
            if ch.get("change"):
                await self._setup(room, ch["change"], crossfade=ch.get("change_xfade"))

            secs = float(ch.get("seconds", 12.0))
            room.round = {"ch": ch, "answers": {}, "secs": secs,
                          "deadline": time.monotonic() + secs}
            await self._broadcast(room, {"type": "round", "round": rnd, "total": MAX_ROUNDS,
                                         "kind": ch["kind"], "concept": ch["concept"],
                                         "question": ch["question"], "sub": ch["sub"],
                                         "options": ch["options"], "seconds": secs})

            # 3) wait, score, reveal
            await self._await_answers(room)
            results = self._score(room)
            got = sum(1 for r in results.values() if r["correct"])
            await self._broadcast(room, {"type": "reveal", "correct": ch["correct"],
                                         "explain": ch["explain"], "reveal": ch.get("reveal", ""),
                                         "got": got, "players": len(room.clients),
                                         "scores": room.leaderboard()})
            for nm, res in results.items():        # personalized result (points + streak pop)
                self._send_to(room, nm, {"type": "result", **res})
            room.round = None
            await asyncio.sleep(REVEAL_SECS)

        room.phase = "done"
        if room.state:
            room.state.running = False
        await self._broadcast(room, {"type": "final", "scores": room.leaderboard()})

    async def _setup(self, room: ClassRoom, d: dict | None, crossfade: bool = False) -> None:
        """Apply a state dict; crossfade the prompt if requested (clean morph)."""
        if not d or room.state is None:
            return
        prompt_a = d.get("prompt_a")
        if prompt_a and crossfade:
            await self._crossfade(room.state, prompt_a)
            d = {k: v for k, v in d.items() if k != "prompt_a"}
        _apply(room.state, d)

    async def _crossfade(self, state: JamState, lead_prompt: str) -> None:
        """Morph to a new lead prompt via blend, then commit it, so the live MLX
        stream slides over instead of garbling on a hard prompt_a swap."""
        lead = _instr(lead_prompt)
        state.set_prompt(prompt_b=lead)
        for i in range(1, XFADE_STEPS + 1):
            if not state.running:
                return
            state.set_params(blend=i / XFADE_STEPS)
            await asyncio.sleep(XFADE_SECS / XFADE_STEPS)
        state.set_prompt(prompt_a=lead)
        state.set_params(blend=0.0)

    async def _await_answers(self, room: ClassRoom) -> None:
        """Wait until everyone has answered or the deadline passes."""
        while room.alive:
            r = room.round
            if r is None:
                return
            if len(r["answers"]) >= len(room.clients):
                return
            if time.monotonic() >= r["deadline"]:
                return
            await asyncio.sleep(0.2)

    def _answer(self, room: ClassRoom, name: str, choice) -> None:
        r = room.round
        if r is None or name in r["answers"] or choice is None:
            return
        r["answers"][name] = (int(choice), time.monotonic())

    def _score(self, room: ClassRoom) -> dict[str, dict]:
        """Update scores + streaks, return per-player result for personalized pops."""
        r = room.round
        results: dict[str, dict] = {}
        if r is None:
            return results
        correct = r["ch"]["correct"]
        secs = r["secs"]
        for name in list(room.clients.keys()):
            ans = r["answers"].get(name)
            ok = ans is not None and ans[0] == correct
            if ok:
                remaining = max(0.0, r["deadline"] - ans[1])
                speed = int(SPEED_POINTS * remaining / secs)
                room.streak[name] = room.streak.get(name, 0) + 1
                streak_bonus = min(room.streak[name] - 1, STREAK_CAP) * STREAK_STEP
                gained = BASE_POINTS + speed + streak_bonus
                room.scores[name] = room.scores.get(name, 0) + gained
                results[name] = {"correct": True, "gained": gained,
                                 "streak": room.streak[name], "streak_bonus": streak_bonus}
            else:
                room.streak[name] = 0
                results[name] = {"correct": False, "gained": 0, "streak": 0, "streak_bonus": 0}
        return results

    # ── audio fan-out + sending ───────────────────────────────────────────────
    def _fanout_audio(self, room: ClassRoom, pcm: bytes) -> None:
        if not room.audio_started.is_set():
            room.audio_started.set()
        for c in room.clients.values():
            if c.q.qsize() < AUDIO_BACKLOG:        # drop audio under backpressure, never JSON
                c.q.put_nowait(pcm)

    async def _pump(self, c: _Client) -> None:
        try:
            while True:
                item = await c.q.get()
                if isinstance(item, tuple) and item and item[0] == "j":
                    await c.ws.send_json(item[1])
                else:
                    await c.ws.send_bytes(item)
        except Exception:  # noqa: BLE001 - client gone; handle() finally cleans up
            pass

    async def _send(self, ws, obj: dict) -> None:
        try:
            await ws.send_json(obj)
        except Exception:  # noqa: BLE001
            pass

    def _send_to(self, room: ClassRoom, name: str, obj: dict) -> None:
        c = room.clients.get(name)
        if c:
            c.q.put_nowait(("j", obj))

    async def _broadcast(self, room: ClassRoom, obj: dict) -> None:
        for c in list(room.clients.values()):
            c.q.put_nowait(("j", obj))

    async def _broadcast_lobby(self, room: ClassRoom) -> None:
        await self._broadcast(room, {"type": "lobby", "code": room.code, "host": room.host,
                                     "players": list(room.clients.keys()), "phase": room.phase})

    @staticmethod
    def _gen_code() -> str:
        return "".join(random.choices(string.ascii_uppercase, k=4))
