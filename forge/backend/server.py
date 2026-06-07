"""
server.py - the web layer (FastAPI + uvicorn).

Phase 1 (DONE): the solo Forge workbench over REST -
  GET  /                       the workbench UI
  POST /api/generate           {spec}            -> {clip}
  POST /api/variations         {spec, n}         -> {clips[]}
  POST /api/keep               {clip_id}         -> {ok, crate_size}
  GET  /api/crate              -> {clips[]}
  GET  /clips/{id}.wav         the rendered audio

Phase 2 (TODO): a WebSocket hub for party modes (rooms, turns, reveal). The hub
serializes all generation through the same single engine (see engines/).

Run:
    cd forge
    uvicorn backend.server:app --port 8000
    # open http://localhost:8000
"""

from __future__ import annotations

import os

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .engines import make_backend
from .forge_core import ForgeCore
from .jam import JamState, capture as jam_capture
from .judge import Judge
from .models import Clip, PromptSpec
from .modes.telephone import TelephoneManager
from .battle import BattleHub
from .net import Hub
from .prompt_hints import HINTS
from .audiotool import AudiotoolBridge
from .classroom import ClassroomHub
from .ensemble import EnsembleHub
from .looper import LooperEngine, instrument_list
from .looper_room import LooperHub
from .prompt_game import PromptGameHub
from .prompt_guess import PromptGuessHub
from .daily import DailyChallenge
from .showdown import ShowdownHub
from .trainer import Trainer
from .transmute import Transmuter
from .rooms import RoomManager
from .storage import Storage


# ── Request models (MUST be module-level so FastAPI can resolve the annotations
#    under `from __future__ import annotations`) ────────────────────────────────
class SpecIn(BaseModel):
    """Validated generation request from the browser (system boundary)."""
    text_a: str = Field(min_length=1)
    text_b: str | None = None
    blend: float = Field(0.0, ge=0.0, le=1.0)
    key: str | None = None
    density: float = Field(0.3, ge=0.0, le=1.0)
    drums: bool = False
    chunks: int = Field(8, ge=1, le=40)        # ~0.8s each → 1..32s
    engine: str | None = None                  # "mrt2" | "sa3"; None = server default

    def to_spec(self) -> PromptSpec:
        tb = self.text_b.strip() if self.text_b and self.text_b.strip() else None
        return PromptSpec(
            text_a=self.text_a.strip(), text_b=tb, blend=self.blend,
            key=(self.key or None), density=self.density,
            drums=self.drums, chunks=self.chunks,
        )


class VariationsIn(SpecIn):
    n: int = Field(4, ge=1, le=8)


class KeepIn(BaseModel):
    clip_id: str


class RenameIn(BaseModel):
    clip_id: str
    name: str = ""


class TransmuteIn(BaseModel):
    """Audio-to-audio restyle of an existing clip (the MRT2→SA3 bridge)."""
    clip_id: str
    prompt: str = Field(min_length=1)
    noise: float = Field(0.8, ge=0.0, le=1.0)   # 0=preserve source, 1=ignore it
    chunks: int = Field(10, ge=1, le=40)


class AudiotoolSendIn(BaseModel):
    """Send a clip into an Audiotool project. `project` (a projects/{uuid} name)
    chains clips into one project; omit it to start a new one. `bpm` tags the
    inserted clip so it locks to the project grid (tempo-matched generation)."""
    clip_id: str
    project: str | None = None
    bpm: int | None = Field(None, ge=20, le=300)


class AudiotoolProjectIn(BaseModel):
    """Connect to an existing Audiotool project to read its tempo/time signature.
    Accepts a studio URL, a projects/{uuid} name, or a bare uuid."""
    project: str = Field(min_length=1)


class NewGameIn(BaseModel):
    players: list[str] = Field(min_length=2, max_length=8)
    chunks: int = Field(8, ge=1, le=40)
    key: str | None = None
    skill: str = "advanced"


class SubmitIn(BaseModel):
    game_id: str
    text: str = Field(min_length=1)


class GameIdIn(BaseModel):
    game_id: str


class TrainAnswerIn(BaseModel):
    token: str
    choice: int


class LooperRenderIn(BaseModel):
    instrument: str
    key: str | None = None
    bpm: float = 0.0          # 0 = AUTO (the first loop sets the session tempo)
    bars: int = 2
    engine: str = "mrt2"


class DailyPlayIn(BaseModel):
    name: str = "anon"
    prompt: str
    engine: str = "mrt2"


class ChordBandIn(BaseModel):
    root: int          # pitch class 0-11
    quality: str       # maj | min | maj7 | 7 | min7 ...
    drums: bool = False


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from forge/.env into the environment (without
    overriding anything already set), so secrets like AUDIOTOOL_PAT and settings
    like FORGE_BACKEND don't have to be typed on every launch. No dependency -
    handles `export KEY=val`, quotes, and # comments."""
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()   # must run before the singletons below read the environment

# ── Singletons (one generation engine for the whole server) ───────────────────
# Engine is selectable via $FORGE_BACKEND (default mrt2); everything downstream
# depends only on the Backend protocol, never on a concrete model.
backend = make_backend()
forge = ForgeCore(backend)
judge = Judge(api_key=os.environ.get("ANTHROPIC_API_KEY"))
storage = Storage()
rooms = RoomManager(forge, judge)
telephone = TelephoneManager(forge, judge, storage)
hub = Hub(forge, judge, storage)
showdown = ShowdownHub(forge, judge, storage)
battle = BattleHub(forge, judge, storage)
trainer = Trainer(forge, storage)
# Transmute uses a dedicated SA3 engine regardless of the main backend (it's an
# SA3-only audio-to-audio op). It shares the one sidecar via the health check, so
# constructing it here is cheap and the sidecar only spawns on first use.
transmuter = Transmuter(make_backend("sa3"))
audiotool = AudiotoolBridge()   # sends clips into Audiotool; needs $AUDIOTOOL_PAT

# Per-engine ForgeCores so the workbench can pick MRT2 or SA3 per generation. Built
# lazily and cached; the default engine reuses the already-started `forge` above.
_DEFAULT_ENGINE = getattr(backend, "name", "mrt2")
_forges: dict[str, ForgeCore] = {_DEFAULT_ENGINE: forge}
_ENGINE_ALIASES = {"mrt": "mrt2", "magenta": "mrt2", "stableaudio": "sa3", "stable-audio": "sa3"}


def _forge_for(name: str | None) -> ForgeCore:
    """Return a ForgeCore bound to the requested engine (mrt2|sa3), building +
    starting it on first use. Unknown/empty names fall back to the default."""
    key = _ENGINE_ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())
    if key not in ("mrt2", "sa3"):
        key = _DEFAULT_ENGINE
    if key not in _forges:
        eng = make_backend(key)
        eng.start()                       # start its worker/sidecar (idempotent)
        _forges[key] = ForgeCore(eng)
    return _forges[key]


def _mrt_worker():
    """The single shared MRT2 worker (one MLX model per process). Reused for both
    clip generation and the live jam, so the model is never loaded twice."""
    return _forge_for("mrt2")._backend


# Live Music Classroom: multiplayer ear-training on one shared MRT2 stream.
classroom_hub = ClassroomHub(_mrt_worker)


def _ensemble_capture(state, seconds):
    """Capture an Ensemble take into the shared Library (same funnel as jam save).
    Late-binds SOLO_USER / _clip_json, which are defined further down the module."""
    clip = jam_capture(state, forge._clips_dir, seconds=seconds)
    storage.put_clip(clip)
    storage.keep(SOLO_USER, clip.id)
    return clip


# Ensemble Room: several players co-steer one shared MRT2 stream; takes -> Audiotool.
ensemble_hub = EnsembleHub(_mrt_worker, _ensemble_capture, lambda c: _clip_json(c), audiotool)

# Per-Instrument Looper: render instrument loops, tempo-align, stack and mix them.
looper_engine = LooperEngine(_forge_for, forge._clips_dir, storage)

# Multiplayer Looper Room: each player owns an instrument; the server mixes the
# loops into one synced stream broadcast to everyone. Stems export to Audiotool,
# optionally finished per-stem through SA3 (MRT2 performs, SA3 produces).
looper_hub = LooperHub(looper_engine, audiotool, transmuter)

# Prompt Party: a social game that teaches AI music prompting (brief -> prompt ->
# MRT2 generates -> AI judge feedback + room vote). The education front door.
prompt_game_hub = PromptGameHub(_forge_for, storage, judge)

# Prompt Detective: reverse game - a DJ composes a secret track, others guess the prompt.
prompt_guess_hub = PromptGuessHub(_forge_for, storage, judge)

# Daily Challenge: one shared brief per day, persistent leaderboard (the habit loop).
daily = DailyChallenge(_forge_for, storage, judge,
                       os.path.join(os.path.dirname(forge._clips_dir), "daily.json"))


async def jam_session(ws: WebSocket) -> None:
    """Live Morph / Jam: stream continuous MRT2 audio to the browser and accept
    live control (blend/density/prompt) + capture. See jam.py / mrt_worker.run_jam."""
    import asyncio  # noqa: PLC0415

    await ws.accept()
    try:
        init = await ws.receive_json()
    except Exception:  # noqa: BLE001
        await ws.close()
        return

    state = JamState(
        prompt_a=(init.get("prompt_a") or "warm evolving ambient pad").strip(),
        prompt_b=(init.get("prompt_b") or None),
        key=(init.get("key") or None),
        blend=float(init.get("blend", 0.0)),
        density=float(init.get("density", 0.3)),
        drums=bool(init.get("drums", False)),
    )

    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=8)   # small: pacing keeps it ~1-2 deep; cap latency

    def _enqueue(pcm: bytes) -> None:
        if audio_q.full():                 # drop oldest to stay near real-time
            try:
                audio_q.get_nowait()
            except Exception:  # noqa: BLE001
                pass
        try:
            audio_q.put_nowait(pcm)
        except Exception:  # noqa: BLE001
            pass

    def emit(pcm: bytes) -> None:          # called from the worker thread
        loop.call_soon_threadsafe(_enqueue, pcm)

    worker = _mrt_worker()
    worker.start()
    worker.submit_jam(state, emit)

    async def pump() -> None:
        try:
            while state.running:
                await ws.send_bytes(await audio_q.get())
        except Exception:  # noqa: BLE001
            state.running = False

    pump_task = loop.create_task(pump())
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "control":
                state.set_params(blend=msg.get("blend"), density=msg.get("density"),
                                 drums=msg.get("drums"))
            elif action == "prompt":
                state.set_prompt(prompt_a=msg.get("prompt_a"), prompt_b=msg.get("prompt_b"),
                                 key=msg.get("key"))
            elif action == "save":
                try:
                    clip = jam_capture(state, forge._clips_dir, seconds=msg.get("seconds"))
                    storage.put_clip(clip)
                    storage.keep(SOLO_USER, clip.id)   # show it in the workbench Library
                    await ws.send_json({"type": "saved", "clip": _clip_json(clip)})
                except ValueError as e:
                    await ws.send_json({"type": "error", "error": str(e)})
            elif action == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        state.running = False             # stops the worker's run_jam loop
        pump_task.cancel()

async def harmony_session(ws: WebSocket) -> None:
    """Harmony Sandbox: a live MRT2 band that arranges itself around the chords the
    learner picks. Same streaming spine as jam_session; steering is a chord (root +
    quality) fed to MRT2's note conditioning instead of sliders. See harmony.py."""
    import asyncio  # noqa: PLC0415

    await ws.accept()
    try:
        init = await ws.receive_json()
    except Exception:  # noqa: BLE001
        await ws.close()
        return

    state = JamState(
        prompt_a=(init.get("prompt_a") or "warm session band, electric piano, upright bass, soft drums").strip(),
        density=float(init.get("density", 0.5)),
        drums=bool(init.get("drums", True)),
    )

    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=8)

    def _enqueue(pcm: bytes) -> None:
        if audio_q.full():
            try:
                audio_q.get_nowait()
            except Exception:  # noqa: BLE001
                pass
        try:
            audio_q.put_nowait(pcm)
        except Exception:  # noqa: BLE001
            pass

    def emit(pcm: bytes) -> None:
        loop.call_soon_threadsafe(_enqueue, pcm)

    worker = _mrt_worker()
    worker.start()
    worker.submit_jam(state, emit)

    async def pump() -> None:
        try:
            while state.running:
                await ws.send_bytes(await audio_q.get())
        except Exception:  # noqa: BLE001
            state.running = False

    pump_task = loop.create_task(pump())
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "chord":
                root = int(msg.get("root", 0)) % 12
                quality = str(msg.get("quality", "maj"))
                state.set_chord((root, quality))
            elif action == "free":
                state.set_chord(None)
            elif action == "control":
                state.set_params(density=msg.get("density"), drums=msg.get("drums"))
            elif action == "prompt":
                state.set_prompt(prompt_a=msg.get("prompt_a"))
            elif action == "save":
                try:
                    clip = jam_capture(state, forge._clips_dir, seconds=msg.get("seconds"))
                    storage.put_clip(clip)
                    storage.keep(SOLO_USER, clip.id)
                    await ws.send_json({"type": "saved", "clip": _clip_json(clip)})
                except ValueError as e:
                    await ws.send_json({"type": "error", "error": str(e)})
            elif action == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        state.running = False
        pump_task.cancel()


# Solo workbench has no accounts yet - everything goes to one crate.
SOLO_USER = "me"

_FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")


def _clip_json(clip: Clip) -> dict:
    """Shape a Clip for the browser (audio URL + the prompt that made it)."""
    s = clip.spec
    return {
        "id": clip.id,
        "url": f"/clips/{clip.id}.wav",
        "name": clip.name,
        "engine": clip.engine,
        "spec": {
            "text_a": s.text_a, "text_b": s.text_b, "blend": s.blend,
            "key": s.key, "density": s.density, "drums": s.drums, "chunks": s.chunks,
        },
    }


def build_app():
    from fastapi import FastAPI, HTTPException  # noqa: PLC0415
    from fastapi.responses import FileResponse  # noqa: PLC0415

    app = FastAPI(title="Forge")

    @app.on_event("startup")
    def _startup() -> None:
        backend.start()   # heavy models load lazily on the first job

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONTEND, "index.html"))

    @app.get("/app.css")
    def app_css():
        return FileResponse(os.path.join(_FRONTEND, "app.css"), media_type="text/css")

    @app.get("/wave.js")
    def wave_js():
        return FileResponse(os.path.join(_FRONTEND, "wave.js"), media_type="application/javascript")

    @app.get("/jam-client.js")
    def jam_client_js():
        return FileResponse(os.path.join(_FRONTEND, "jam-client.js"), media_type="application/javascript")

    @app.get("/api/hints")
    def hints():
        return HINTS

    @app.post("/api/generate")
    def generate(body: SpecIn):
        clip = _forge_for(body.engine).generate(body.to_spec(), created_by=SOLO_USER)
        storage.put_clip(clip)
        return _clip_json(clip)

    @app.post("/api/variations")
    def variations(body: VariationsIn):
        clips = _forge_for(body.engine).variations(body.to_spec(), n=body.n, created_by=SOLO_USER)
        for c in clips:
            storage.put_clip(c)
        return {"clips": [_clip_json(c) for c in clips]}

    @app.post("/api/keep")
    def keep(body: KeepIn):
        if storage.get_clip(body.clip_id) is None:
            raise HTTPException(404, "unknown clip")
        storage.keep(SOLO_USER, body.clip_id)
        return {"ok": True, "crate_size": len(storage.crate(SOLO_USER))}

    @app.post("/api/unkeep")
    def unkeep(body: KeepIn):
        storage.unkeep(SOLO_USER, body.clip_id)
        return {"ok": True, "crate_size": len(storage.crate(SOLO_USER))}

    @app.post("/api/clip/name")
    def clip_rename(body: RenameIn):
        if not storage.rename(body.clip_id, body.name.strip()[:80]):
            raise HTTPException(404, "unknown clip")
        return {"ok": True}

    @app.post("/api/transmute")
    def transmute(body: TransmuteIn):
        src = storage.get_clip(body.clip_id)
        if src is None or not os.path.exists(src.wav_path):
            raise HTTPException(404, "unknown clip")
        try:
            clip = transmuter.transmute(
                src.wav_path, body.prompt, init_noise_level=body.noise,
                chunks=body.chunks, created_by=SOLO_USER,
            )
        except (ValueError, RuntimeError, TimeoutError) as e:
            raise HTTPException(502, f"transmute failed: {e}")
        storage.put_clip(clip)
        return _clip_json(clip)

    @app.get("/api/audiotool/status")
    def audiotool_status():
        """Whether the Audiotool bridge is usable (PAT configured) - lets the UI
        show/hide the Send button without exposing the token."""
        return {"connected": audiotool.configured}

    @app.post("/api/audiotool/send")
    def audiotool_send(body: AudiotoolSendIn):
        clip = storage.get_clip(body.clip_id)
        if clip is None or not os.path.exists(clip.wav_path):
            raise HTTPException(404, "unknown clip")
        name = clip.name or clip.spec.text_a
        try:
            return audiotool.send_clip(clip.wav_path, name, project=body.project, bpm=body.bpm)
        except (ValueError, RuntimeError, TimeoutError) as e:
            raise HTTPException(502, str(e))

    @app.post("/api/audiotool/project")
    def audiotool_project(body: AudiotoolProjectIn):
        """Connect to an Audiotool project and read its musical context (tempo)."""
        try:
            return audiotool.project_info(body.project)
        except (ValueError, RuntimeError, TimeoutError) as e:
            raise HTTPException(502, str(e))

    @app.get("/api/crate")
    def crate():
        return {"clips": [_clip_json(c) for c in storage.crate(SOLO_USER)]}

    @app.get("/clips/{clip_id}.wav")
    def clip_wav(clip_id: str):
        clip = storage.get_clip(clip_id)
        if clip is None or not os.path.exists(clip.wav_path):
            raise HTTPException(404, "clip not found")
        return FileResponse(clip.wav_path, media_type="audio/wav")

    # ── Telephone ("Broken Record") ──────────────────────────────────────────
    @app.get("/telephone")
    def telephone_page():
        return FileResponse(os.path.join(_FRONTEND, "telephone.html"))

    @app.get("/play")
    def play_page():
        return FileResponse(os.path.join(_FRONTEND, "room.html"))

    @app.get("/showdown")
    def showdown_page():
        return FileResponse(os.path.join(_FRONTEND, "showdown.html"))

    @app.get("/battle")
    def battle_page():
        return FileResponse(os.path.join(_FRONTEND, "battle.html"))

    @app.get("/train")
    def train_page():
        return FileResponse(os.path.join(_FRONTEND, "train.html"))

    @app.get("/jam")
    def jam_page():
        return FileResponse(os.path.join(_FRONTEND, "jam.html"))

    @app.get("/journey")
    def journey_page():
        return FileResponse(os.path.join(_FRONTEND, "journey.html"))

    @app.get("/live-trainer")
    def live_trainer_page():
        return FileResponse(os.path.join(_FRONTEND, "live-trainer.html"))

    @app.get("/classroom")
    def classroom_page():
        return FileResponse(os.path.join(_FRONTEND, "classroom.html"))

    @app.get("/harmony")
    def harmony_page():
        return FileResponse(os.path.join(_FRONTEND, "harmony.html"))

    @app.get("/ensemble")
    def ensemble_page():
        return FileResponse(os.path.join(_FRONTEND, "ensemble.html"))

    @app.get("/looper")
    def looper_page():
        return FileResponse(os.path.join(_FRONTEND, "looper.html"))

    @app.get("/looper-room")
    def looper_room_page():
        return FileResponse(os.path.join(_FRONTEND, "looper-room.html"))

    @app.get("/prompt-party")
    def prompt_party_page():
        return FileResponse(os.path.join(_FRONTEND, "prompt-party.html"))

    @app.get("/prompt-detective")
    def prompt_detective_page():
        return FileResponse(os.path.join(_FRONTEND, "prompt-detective.html"))

    @app.get("/daily")
    def daily_page():
        return FileResponse(os.path.join(_FRONTEND, "daily.html"))

    @app.get("/chord-coach")
    def chord_coach_page():
        return FileResponse(os.path.join(_FRONTEND, "chord-coach.html"))

    @app.post("/api/chord-band")
    def chord_band(body: ChordBandIn):
        # A short, GENTLE MRT2 accompaniment of the chord the learner played, so
        # they hear what it sounds like with a band. Slow + soft + sparse so it's
        # easy to follow while practicing (note conditioning = MRT2 only).
        spec = PromptSpec(
            text_a="warm gentle band, soft electric piano and mellow bass, slow, "
                   "sustained, relaxed, instrumental",
            density=0.22, drums=bool(body.drums), chunks=6,
            chord=(int(body.root) % 12, str(body.quality)),
        )
        clip = _forge_for("mrt2").generate(spec, "chord-coach")
        storage.put_clip(clip)
        return {"ok": True, "clip_url": f"/clips/{clip.id}.wav"}

    @app.get("/api/daily")
    def daily_state():
        return daily.state()

    @app.post("/api/daily/play")
    def daily_play(body: DailyPlayIn):
        return daily.play(body.name, body.prompt, body.engine)

    @app.get("/api/looper/instruments")
    def looper_instruments():
        return {"instruments": instrument_list()}

    @app.post("/api/looper/render")
    def looper_render(body: LooperRenderIn):
        try:
            clip, info = looper_engine.render(body.instrument, body.key, body.bpm, body.bars,
                                              engine=body.engine)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "clip": _clip_json(clip), **info}

    @app.get("/api/harmony")
    def harmony_key(root: str = "C", mode: str = "major"):
        from .harmony import key_payload  # noqa: PLC0415
        return key_payload(root, mode)

    @app.get("/api/train/next")
    def train_next(category: str = "mixed"):
        return trainer.question(category=category)

    @app.post("/api/train/answer")
    def train_answer(body: TrainAnswerIn):
        try:
            return trainer.answer(body.token, body.choice)
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.get("/train/audio/{token}.wav")
    def train_audio(token: str):
        path = trainer.audio_path(token)
        if not path or not os.path.exists(path):
            raise HTTPException(404, "audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/telephone/new")
    def telephone_new(body: NewGameIn):
        names = [p.strip() for p in body.players if p.strip()]
        if len(names) < 2:
            raise HTTPException(400, "need at least 2 player names")
        skill = body.skill if body.skill in ("beginner", "advanced") else "advanced"
        g = telephone.new(names, chunks=body.chunks, key=(body.key or None), skill=skill)
        return g.public_state()

    @app.post("/api/telephone/submit")
    def telephone_submit(body: SubmitIn):
        g = telephone.get(body.game_id)
        if g is None:
            raise HTTPException(404, "unknown game")
        try:
            return g.submit(body.text)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/telephone/reveal")
    def telephone_reveal(body: GameIdIn):
        g = telephone.get(body.game_id)
        if g is None:
            raise HTTPException(404, "unknown game")
        if not g.complete:
            raise HTTPException(400, "game not complete yet")
        return g.reveal()

    # ── Networked Broken Record (Phase 2b) ────────────────────────────────────
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await hub.handle(websocket)

    @app.websocket("/ws/showdown")
    async def ws_showdown(websocket: WebSocket):
        await showdown.handle(websocket)

    @app.websocket("/ws/battle")
    async def ws_battle(websocket: WebSocket):
        await battle.handle(websocket)

    @app.websocket("/ws/jam")
    async def ws_jam(websocket: WebSocket):
        await jam_session(websocket)

    @app.websocket("/ws/classroom")
    async def ws_classroom(websocket: WebSocket):
        await classroom_hub.handle(websocket)

    @app.websocket("/ws/harmony")
    async def ws_harmony(websocket: WebSocket):
        await harmony_session(websocket)

    @app.websocket("/ws/looper-room")
    async def ws_looper_room(websocket: WebSocket):
        await looper_hub.handle(websocket)

    @app.websocket("/ws/prompt-party")
    async def ws_prompt_party(websocket: WebSocket):
        await prompt_game_hub.handle(websocket)

    @app.websocket("/ws/prompt-detective")
    async def ws_prompt_detective(websocket: WebSocket):
        await prompt_guess_hub.handle(websocket)

    @app.websocket("/ws/ensemble")
    async def ws_ensemble(websocket: WebSocket):
        await ensemble_hub.handle(websocket)

    return app


app = build_app()


def main() -> None:
    import uvicorn  # noqa: PLC0415
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
