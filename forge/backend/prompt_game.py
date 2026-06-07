"""
prompt_game.py - the Prompt Party: a social game that teaches AI music prompting.

The education thesis made into a game. The skill that matters most in AI music is
PROMPTING: turning a feeling into words that produce the sound you imagined. So:

  1. The room gets a BRIEF (a scene or feeling, e.g. "a tense underwater chase").
  2. Everyone writes a prompt to hit it (with beginner chips: mood/genre/instrument/
     tempo, so no one faces a blank box and no theory is needed).
  3. MRT2 generates each player's clip.
  4. An AI judge scores each PROMPT against the brief and explains WHY in plain
     language (the teaching), while the room VOTES on the clips (the fun).
  5. Points = AI score + votes. Leaderboard, rounds, winner.

This is the front door: learn to prompt here, then use that skill in the Looper to
build a real track, then export to a DAW.

Comms are JSON-only over the WS; clips are played by the browser from /clips/<id>.wav.
"""

from __future__ import annotations

import asyncio
import random
import re
import string
import time

from .models import PromptSpec

WRITE_SECS = 50.0
VOTE_SECS = 25.0
REVEAL_SECS = 10.0
MAX_ROUNDS = 3
VOTE_BONUS = 120        # points per vote received
SEC_PER_CHUNK = 0.8
DEFAULT_CLIP_SECS = 8.0  # host can change; the point is the music is long enough to judge

# Words too generic to ban from the brief.
_BRIEF_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "with",
               "into", "through", "over", "under", "by", "for", "you", "your"}


def _brief_banned(brief: str) -> set[str]:
    """The brief's content words. Players may not parrot these; they must translate
    the scene into actual SOUND. (Synonyms are discouraged by the judge.)"""
    return {w for w in re.findall(r"[a-z]+", brief.lower())
            if w not in _BRIEF_STOP and len(w) > 2}


def _banned_hits(banned: set[str], prompt: str) -> list[str]:
    hits = set()
    for t in re.findall(r"[a-z]+", (prompt or "").lower()):
        for b in banned:
            if t == b or (len(b) >= 4 and (b in t or t in b)):
                hits.add(b)
    return sorted(hits)

BRIEFS = [
    "a triumphant sunrise", "a tense underwater chase", "a rainy night drive",
    "a cozy coffee shop morning", "an epic final boss battle", "a heartbroken slow dance",
    "a dreamy float through space", "a chaotic carnival", "a peaceful forest at dawn",
    "a retro arcade victory", "the calm before a storm", "a hopeful new beginning",
]

# Beginner prompt-builder: tap chips instead of facing a blank box. Also teaches the
# vocabulary a good prompt is built from.
CHIPS = {
    "Mood": ["warm", "dark", "tense", "dreamy", "uplifting", "melancholy", "epic", "playful"],
    "Genre": ["lo-fi hip hop", "orchestral", "techno", "jazz", "ambient", "funk", "synthwave", "folk"],
    "Instrument": ["piano", "strings", "synth pad", "electric guitar", "saxophone", "bells", "808 bass"],
    "Tempo": ["slow", "mid-tempo", "fast", "driving"],
}

_DRUM_WORDS = ("drum", "beat", "percussion", "808", "groove")


class _Client:
    def __init__(self, ws, name: str):
        self.ws = ws
        self.name = name
        self.q: asyncio.Queue = asyncio.Queue()
        self.pump: asyncio.Task | None = None


class GameRoom:
    def __init__(self, code: str, host: str):
        self.code = code
        self.host = host
        self.clients: dict[str, _Client] = {}
        self.scores: dict[str, int] = {}
        self.phase = "lobby"
        self.brief = ""
        self.banned: set[str] = set()              # this round's forbidden words
        self.clip_secs = DEFAULT_CLIP_SECS         # host-set clip length
        self.engine = "mrt2"                        # host-chosen model: mrt2 | sa3
        self.stats: dict[str, dict] = {}           # session stats for end-of-game badges
        self.submissions: dict[str, str] = {}      # name -> prompt (this round)
        self.entries: list[dict] = []              # this round's generated entries
        self.votes: dict[str, int] = {}            # voter -> eid
        self.alive = True

    def leaderboard(self) -> list[dict]:
        return [{"name": n, "score": s}
                for n, s in sorted(self.scores.items(), key=lambda kv: -kv[1])]


class PromptGameHub:
    """Manages Prompt Party rooms. Reuses MRT2 (forge) + the Judge."""

    def __init__(self, forge_getter, storage, judge):
        self._get_forge = forge_getter
        self._storage = storage
        self._judge = judge
        self._rooms: dict[str, GameRoom] = {}

    # ── connection lifecycle ─────────────────────────────────────────────────
    async def handle(self, ws) -> None:
        from fastapi import WebSocketDisconnect  # noqa: PLC0415
        await ws.accept()
        room: GameRoom | None = None
        name: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "create":
                    room, name = self._create(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": True, "chips": CHIPS})
                    await self._broadcast_lobby(room)
                elif action == "join":
                    room, name = self._join(ws, msg)
                    await self._send(ws, {"type": "joined", "code": room.code, "name": name,
                                          "host": (name == room.host), "chips": CHIPS})
                    await self._broadcast_lobby(room)
                elif not room or not name:
                    continue
                elif action == "start" and name == room.host and room.phase == "lobby":
                    room.clip_secs = max(8.0, min(20.0, float(msg.get("seconds", DEFAULT_CLIP_SECS))))
                    room.engine = "sa3" if msg.get("engine") == "sa3" else "mrt2"
                    asyncio.create_task(self._run(room))
                elif action == "submit" and room.phase == "writing":
                    hits = _banned_hits(room.banned, msg.get("prompt") or "")
                    if hits:
                        await self._send(ws, {"type": "rejected", "words": hits})
                    else:
                        self._submit(room, name, msg.get("prompt"))
                        await self._send(ws, {"type": "accepted"})
                elif action == "vote" and room.phase == "voting":
                    self._vote(room, name, msg.get("eid"))
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

    # ── room ops ──────────────────────────────────────────────────────────────
    def _create(self, ws, msg) -> tuple[GameRoom, str]:
        name = (msg.get("name") or "Host").strip()[:24] or "Host"
        room = GameRoom(self._gen_code(), host=name)
        self._rooms[room.code] = room
        self._add_client(room, ws, name)
        return room, name

    def _join(self, ws, msg) -> tuple[GameRoom, str]:
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

    def _add_client(self, room: GameRoom, ws, name: str) -> None:
        c = _Client(ws, name)
        room.clients[name] = c
        room.scores.setdefault(name, 0)
        room.stats.setdefault(name, {"votes": 0, "best_match": 0, "round_wins": 0})
        c.pump = asyncio.create_task(self._pump(c))

    async def _leave(self, room: GameRoom, name: str) -> None:
        c = room.clients.pop(name, None)
        if c and c.pump:
            c.pump.cancel()
        if not room.clients:
            room.alive = False
            self._rooms.pop(room.code, None)
        else:
            await self._broadcast_lobby(room)

    def _submit(self, room: GameRoom, name: str, prompt) -> None:
        p = (prompt or "").strip()
        if p:
            room.submissions[name] = p[:200]

    def _vote(self, room: GameRoom, name: str, eid) -> None:
        try:
            eid = int(eid)
        except (TypeError, ValueError):
            return
        if name in room.votes:
            return
        ent = next((e for e in room.entries if e["eid"] == eid), None)
        if ent is None or ent["name"] == name:     # cannot vote your own
            return
        room.votes[name] = eid

    # ── round orchestrator ──────────────────────────────────────────────────
    async def _run(self, room: GameRoom) -> None:
        room.phase = "playing"
        for rnd in range(1, MAX_ROUNDS + 1):
            if not room.alive:
                break
            brief = random.choice(BRIEFS)
            room.brief = brief
            room.banned = _brief_banned(brief)
            room.submissions = {}
            room.phase = "writing"
            await self._broadcast(room, {"type": "brief", "round": rnd, "total": MAX_ROUNDS,
                                         "brief": brief, "seconds": WRITE_SECS,
                                         "banned": sorted(room.banned)})
            await self._await_until(room, lambda: len(room.submissions) >= len(room.clients), WRITE_SECS)
            if not room.alive:
                break

            room.phase = "generating"
            await self._broadcast(room, {"type": "generating", "count": len(room.submissions)})
            entries = await self._generate_entries(room, brief)
            if not entries:
                await self._broadcast(room, {"type": "reveal", "brief": brief, "results": [],
                                             "scores": room.leaderboard()})
                await asyncio.sleep(REVEAL_SECS)
                continue

            random.shuffle(entries)
            for i, e in enumerate(entries):
                e["eid"] = i
            room.entries = entries
            room.votes = {}
            room.phase = "voting"
            await self._broadcast(room, {"type": "vote", "brief": brief, "seconds": VOTE_SECS,
                                         "entries": [{"eid": e["eid"], "clip_url": e["clip_url"]}
                                                     for e in entries]})
            # advance as soon as everyone who CAN vote has voted (you can't vote
            # your own entry, so eligible = clients with someone else's entry to pick)
            await self._await_until(room, lambda: len(room.votes) >= self._eligible_voters(room), VOTE_SECS)

            self._score_round(room)
            results = sorted(
                [{"name": e["name"], "prompt": e["prompt"], "clip_url": e["clip_url"],
                  "score": e["score"], "note": e["note"], "votes": e["votes"],
                  "points": e["points"]} for e in room.entries],
                key=lambda r: -r["points"])
            room.phase = "reveal"
            await self._broadcast(room, {"type": "reveal", "brief": brief, "results": results,
                                         "scores": room.leaderboard()})
            await asyncio.sleep(REVEAL_SECS)

        room.phase = "done"
        await self._broadcast(room, {"type": "final", "scores": room.leaderboard(),
                                     "badges": self._badges(room)})

    async def _generate_entries(self, room: GameRoom, brief: str) -> list[dict]:
        forge = self._get_forge(room.engine)
        judge = self._judge
        chunks = max(2, round(room.clip_secs / SEC_PER_CHUNK))
        entries: list[dict] = []
        done = 0
        for name, prompt in list(room.submissions.items()):
            if not room.alive:
                break
            drums = any(w in prompt.lower() for w in _DRUM_WORDS)
            spec = PromptSpec(text_a=prompt, density=0.4, drums=drums, chunks=chunks)
            try:
                clip = await asyncio.to_thread(forge.generate, spec, name)
                self._storage.put_clip(clip)
                verdict = await asyncio.to_thread(judge.score_brief, prompt, brief)
                entries.append({"name": name, "prompt": prompt,
                                "clip_url": f"/clips/{clip.id}.wav",
                                "score": int(round(getattr(verdict, "score", 0.0))),
                                "note": getattr(verdict, "note", ""), "votes": 0, "points": 0})
            except Exception as e:  # noqa: BLE001 - one bad gen must not kill the round
                entries.append({"name": name, "prompt": prompt, "clip_url": "",
                                "score": 0, "note": f"(could not generate: {e})",
                                "votes": 0, "points": 0})
            done += 1
            await self._broadcast(room, {"type": "generating", "count": len(room.submissions),
                                         "done": done})
        return entries

    def _score_round(self, room: GameRoom) -> None:
        for eid in room.votes.values():
            ent = next((e for e in room.entries if e["eid"] == eid), None)
            if ent:
                ent["votes"] += 1
        for e in room.entries:
            e["points"] = e["score"] + e["votes"] * VOTE_BONUS
            room.scores[e["name"]] = room.scores.get(e["name"], 0) + e["points"]
            s = room.stats.setdefault(e["name"], {"votes": 0, "best_match": 0, "round_wins": 0})
            s["votes"] += e["votes"]
            s["best_match"] = max(s["best_match"], e["score"])
        if room.entries:
            winner = max(room.entries, key=lambda e: e["points"])
            room.stats.setdefault(winner["name"],
                                  {"votes": 0, "best_match": 0, "round_wins": 0})["round_wins"] += 1

    def _badges(self, room: GameRoom) -> dict:
        """Fun end-of-game superlatives from the session stats."""
        badges: dict[str, list[str]] = {}
        lb = room.leaderboard()
        if lb and lb[0]["score"] > 0:
            badges.setdefault(lb[0]["name"], []).append("🏆 Champion")
        st = room.stats
        if st:
            sharp = max(st, key=lambda n: st[n]["best_match"])
            if st[sharp]["best_match"] >= 70:
                badges.setdefault(sharp, []).append("🎯 Sharpshooter")
            fav = max(st, key=lambda n: st[n]["votes"])
            if st[fav]["votes"] > 0:
                badges.setdefault(fav, []).append("🎤 Crowd Favorite")
            wins = max(st, key=lambda n: st[n]["round_wins"])
            if st[wins]["round_wins"] >= 2:
                badges.setdefault(wins, []).append("🔥 On a Roll")
        return badges

    def _eligible_voters(self, room: GameRoom) -> int:
        """How many players actually have something to vote for (an entry that
        isn't their own). Voting ends once they've all voted."""
        owners = {e["name"] for e in room.entries}
        return sum(1 for c in room.clients if owners - {c})

    async def _await_until(self, room: GameRoom, cond, secs: float) -> None:
        deadline = time.monotonic() + secs
        while room.alive and not cond() and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

    # ── sending ────────────────────────────────────────────────────────────────
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

    async def _broadcast(self, room: GameRoom, obj: dict) -> None:
        for c in list(room.clients.values()):
            c.q.put_nowait(obj)

    async def _broadcast_lobby(self, room: GameRoom) -> None:
        await self._broadcast(room, {"type": "lobby", "code": room.code, "host": room.host,
                                     "players": list(room.clients.keys())})

    @staticmethod
    def _gen_code() -> str:
        return "".join(random.choices(string.ascii_uppercase, k=4))
