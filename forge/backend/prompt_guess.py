"""
prompt_guess.py - Prompt Detective: the reverse game.

One player (the DJ) writes a SECRET prompt; MRT2 composes it. Everyone else hears
the track and races to GUESS the prompt. Guesses are scored on accuracy (the AI
judge compares each guess to the true prompt) and speed. Rhythm-game flavor on the
client: PERFECT / GREAT / GOOD ratings and combos.

Where Prompt Party teaches prompt -> sound, this teaches sound -> prompt: training
your ear to hear a clip and name the words behind it. The DJ rotates each round, and
the DJ earns points for how guessable (clearly-prompted) their track was, so both
sides are learning to prompt.

Reuses MRT2 (forge) for generation and the original Judge.score(guess, truth).
JSON-only WS; clips played from /clips/<id>.wav.
"""

from __future__ import annotations

import asyncio
import random
import string
import time

from .models import PromptSpec

SECRET_SECS = 55.0
GUESS_SECS = 35.0
REVEAL_SECS = 11.0
CLIP_CHUNKS = 10        # ~8s tracks, long enough to study
SPEED_MAX = 150         # max speed bonus for a fast, accurate guess
COMBO_STEP = 40         # bonus per combo level (consecutive GREAT+)
GREAT = 70              # accuracy needed to keep/extend a combo


def rating_for(score: float) -> str:
    if score >= 90: return "PERFECT"
    if score >= 70: return "GREAT"
    if score >= 50: return "GOOD"
    if score >= 30: return "OK"
    return "MISS"


class _Client:
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name
        self.q: asyncio.Queue = asyncio.Queue()
        self.pump: asyncio.Task | None = None


class GuessRoom:
    def __init__(self, code: str, host: str):
        self.code = code
        self.host = host
        self.clients: dict[str, _Client] = {}
        self.scores: dict[str, int] = {}
        self.combo: dict[str, int] = {}
        self.stats: dict[str, dict] = {}           # session stats for end-of-game badges
        self.engine = "mrt2"                        # host-chosen model: mrt2 | sa3
        self.phase = "lobby"
        self.dj = ""
        self.secret = ""
        self.guesses: dict[str, tuple[str, float]] = {}   # name -> (text, monotonic)
        self.deadline = 0.0
        self.alive = True

    def leaderboard(self) -> list[dict]:
        return [{"name": n, "score": s}
                for n, s in sorted(self.scores.items(), key=lambda kv: -kv[1])]


class PromptGuessHub:
    def __init__(self, forge_getter, storage, judge):
        self._get_forge = forge_getter
        self._storage = storage
        self._judge = judge
        self._rooms: dict[str, GuessRoom] = {}

    async def handle(self, ws) -> None:
        from fastapi import WebSocketDisconnect  # noqa: PLC0415
        await ws.accept()
        room: GuessRoom | None = None
        name: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "create":
                    room, name = self._create(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name, "host": True})
                    await self._broadcast_lobby(room)
                elif action == "join":
                    room, name = self._join(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": (name == room.host)})
                    await self._broadcast_lobby(room)
                elif not room or not name:
                    continue
                elif action == "start" and name == room.host and room.phase == "lobby":
                    if len(room.clients) >= 2:
                        room.engine = "sa3" if msg.get("engine") == "sa3" else "mrt2"
                        asyncio.create_task(self._run(room))
                    else:
                        await self._send(ws, {"type": "error", "error": "Need at least 2 players."})
                elif action == "secret" and room.phase == "secret" and name == room.dj:
                    self._set_secret(room, msg.get("prompt"))
                elif action == "guess" and room.phase == "guessing" and name != room.dj:
                    self._guess(room, name, msg.get("text"))
                elif action == "react":
                    em = str(msg.get("emoji", ""))[:8]
                    if em:
                        await self._broadcast(room, {"type": "react", "emoji": em, "by": name})
                elif action == "stop":
                    break
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            if room and name:
                await self._leave(room, name)

    # ── rooms ───────────────────────────────────────────────────────────────
    def _create(self, ws, msg) -> tuple[GuessRoom, str]:
        name = (msg.get("name") or "Host").strip()[:24] or "Host"
        room = GuessRoom(self._gen_code(), host=name)
        self._rooms[room.code] = room
        self._add_client(room, ws, name)
        return room, name

    def _join(self, ws, msg) -> tuple[GuessRoom, str]:
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

    def _add_client(self, room: GuessRoom, ws, name: str) -> None:
        c = _Client(ws, name)
        room.clients[name] = c
        room.scores.setdefault(name, 0)
        room.combo.setdefault(name, 0)
        room.stats.setdefault(name, {"best_acc": 0, "max_combo": 0, "correct": 0, "dj_best": 0})
        c.pump = asyncio.create_task(self._pump(c))

    async def _leave(self, room: GuessRoom, name: str) -> None:
        c = room.clients.pop(name, None)
        if c and c.pump:
            c.pump.cancel()
        if not room.clients:
            room.alive = False
            self._rooms.pop(room.code, None)
        else:
            await self._broadcast_lobby(room)

    def _set_secret(self, room: GuessRoom, prompt) -> None:
        p = (prompt or "").strip()
        if p:
            room.secret = p[:200]

    def _guess(self, room: GuessRoom, name: str, text) -> None:
        t = (text or "").strip()
        if t and name not in room.guesses:
            room.guesses[name] = (t[:200], time.monotonic())

    # ── round loop ───────────────────────────────────────────────────────────
    async def _run(self, room: GuessRoom) -> None:
        room.phase = "playing"
        order = list(room.clients.keys())
        total = max(2, min(6, len(order)))
        for rnd in range(1, total + 1):
            if not room.alive:
                break
            names = list(room.clients.keys())
            if len(names) < 2:
                break
            dj = names[(rnd - 1) % len(names)]
            room.dj = dj
            room.secret = ""
            room.phase = "secret"
            await self._broadcast(room, {"type": "dj", "round": rnd, "total": total, "dj": dj})
            await self._await(room, lambda: bool(room.secret), SECRET_SECS)
            if not room.alive:
                break
            if not room.secret:
                await self._broadcast(room, {"type": "skip", "dj": dj,
                                             "reason": f"{dj} did not compose in time"})
                await asyncio.sleep(2.0)
                continue

            await self._broadcast(room, {"type": "generating", "dj": dj})
            clip_url = await self._generate(room, room.secret)
            if not clip_url:
                await self._broadcast(room, {"type": "skip", "dj": dj, "reason": "generation failed"})
                await asyncio.sleep(2.0)
                continue

            room.guesses = {}
            room.deadline = time.monotonic() + GUESS_SECS
            room.phase = "guessing"
            await self._broadcast(room, {"type": "listen", "dj": dj, "clip_url": clip_url,
                                         "seconds": GUESS_SECS})
            await self._await(room, lambda: len(room.guesses) >= max(0, len(room.clients) - 1), GUESS_SECS)

            results = self._score(room)
            room.phase = "reveal"
            await self._broadcast(room, {"type": "reveal", "dj": dj, "truth": room.secret,
                                         "clip_url": clip_url, "results": results,
                                         "scores": room.leaderboard()})
            await asyncio.sleep(REVEAL_SECS)

        room.phase = "done"
        await self._broadcast(room, {"type": "final", "scores": room.leaderboard(),
                                     "badges": self._badges(room)})

    def _badges(self, room: GuessRoom) -> dict:
        """End-of-game superlatives from the session stats."""
        badges: dict[str, list[str]] = {}
        lb = room.leaderboard()
        if lb and lb[0]["score"] > 0:
            badges.setdefault(lb[0]["name"], []).append("🏆 Champion")
        st = room.stats
        if st:
            sharp = max(st, key=lambda n: st[n]["best_acc"])
            if st[sharp]["best_acc"] >= 70:
                badges.setdefault(sharp, []).append("🎯 Sharp Ear")
            combo = max(st, key=lambda n: st[n]["max_combo"])
            if st[combo]["max_combo"] >= 3:
                badges.setdefault(combo, []).append("🔥 Combo King")
            maestro = max(st, key=lambda n: st[n]["dj_best"])
            if st[maestro]["dj_best"] >= 70:
                badges.setdefault(maestro, []).append("🎚 Maestro")
        return badges

    async def _generate(self, room: GuessRoom, prompt: str) -> str:
        forge = self._get_forge(room.engine)
        drums = any(w in prompt.lower() for w in ("drum", "beat", "percussion", "808", "groove"))
        spec = PromptSpec(text_a=prompt, density=0.4, drums=drums, chunks=CLIP_CHUNKS)
        try:
            clip = await asyncio.to_thread(forge.generate, spec, room.dj)
            self._storage.put_clip(clip)
            return f"/clips/{clip.id}.wav"
        except Exception as e:  # noqa: BLE001
            print(f"[prompt_guess] generate failed: {e}")
            return ""

    def _score(self, room: GuessRoom) -> list[dict]:
        truth = room.secret
        results: list[dict] = []
        accs: list[float] = []
        for name, (guess, t) in room.guesses.items():
            verdict = self._judge.score(guess, truth)
            acc = float(getattr(verdict, "score", 0.0))
            accs.append(acc)
            remaining = max(0.0, room.deadline - t)
            speed = int(SPEED_MAX * remaining / GUESS_SECS) if acc >= 50 else 0
            if acc >= GREAT:
                room.combo[name] = room.combo.get(name, 0) + 1
            else:
                room.combo[name] = 0
            combo = room.combo[name]
            combo_bonus = min(combo, 5) * COMBO_STEP if combo >= 2 else 0
            pts = int(acc) + speed + combo_bonus
            room.scores[name] = room.scores.get(name, 0) + pts
            s = room.stats.setdefault(name, {"best_acc": 0, "max_combo": 0, "correct": 0, "dj_best": 0})
            s["best_acc"] = max(s["best_acc"], int(acc))
            s["max_combo"] = max(s["max_combo"], combo)
            if acc >= 50:
                s["correct"] += 1
            results.append({"name": name, "guess": guess, "score": int(acc),
                            "rating": rating_for(acc), "note": getattr(verdict, "tutor_note", ""),
                            "points": pts, "combo": combo})
        # DJ earns points for how guessable (clearly prompted) their track was
        dj_pts = int(sum(accs) / len(accs)) if accs else 0
        room.scores[room.dj] = room.scores.get(room.dj, 0) + dj_pts
        ds = room.stats.setdefault(room.dj, {"best_acc": 0, "max_combo": 0, "correct": 0, "dj_best": 0})
        ds["dj_best"] = max(ds["dj_best"], dj_pts)
        results.sort(key=lambda r: -r["points"])
        results.append({"name": room.dj, "guess": "(the DJ)", "score": dj_pts,
                        "rating": "DJ", "note": "Points for a clearly-prompted track.",
                        "points": dj_pts, "combo": 0, "is_dj": True})
        return results

    async def _await(self, room: GuessRoom, cond, secs: float) -> None:
        deadline = time.monotonic() + secs
        while room.alive and not cond() and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

    # ── sending ──────────────────────────────────────────────────────────────
    async def _pump(self, c: _Client) -> None:
        try:
            while True:
                obj = await c.q.get()
                await c.ws.send_json(obj)
        except Exception:  # noqa: BLE001
            pass

    async def _send(self, ws, obj: dict) -> None:
        try:
            await ws.send_json(obj)
        except Exception:  # noqa: BLE001
            pass

    async def _broadcast(self, room: GuessRoom, obj: dict) -> None:
        for c in list(room.clients.values()):
            c.q.put_nowait(obj)

    async def _broadcast_lobby(self, room: GuessRoom) -> None:
        await self._broadcast(room, {"type": "lobby", "code": room.code, "host": room.host,
                                     "players": list(room.clients.keys())})

    @staticmethod
    def _gen_code() -> str:
        return "".join(random.choices(string.ascii_uppercase, k=4))
