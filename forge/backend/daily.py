"""
daily.py - the Daily Challenge (the Wordle-for-music-prompting habit loop).

One brief per day, the SAME for everyone (deterministic from the date). You write a
prompt to nail it, MRT2 (or SA3) composes it, the AI judge scores how well your
prompt evokes the scene, and your best score lands on a persistent daily
leaderboard. Solo + async + shareable = the retention engine.

Persistence is a single JSON file (simplest thing that survives restarts):
  { "2026-06-07": { "ana": {"score": 82, "prompt": "...", "clip_id": "..."}, ... } }
We keep each player's BEST score for the day.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os

from .models import PromptSpec
from .prompt_game import BRIEFS, CHIPS, _banned_hits, _brief_banned

_DRUM_WORDS = ("drum", "beat", "percussion", "808", "groove")


class DailyChallenge:
    def __init__(self, forge_getter, storage, judge, path: str):
        self._get_forge = forge_getter
        self._storage = storage
        self._judge = judge
        self._path = path

    # ── brief + persistence ──────────────────────────────────────────────────
    @staticmethod
    def today() -> str:
        return datetime.date.today().isoformat()

    def brief(self, date: str | None = None) -> str:
        d = date or self.today()
        idx = int(hashlib.sha1(d.encode()).hexdigest(), 16) % len(BRIEFS)
        return BRIEFS[idx]

    def _load(self) -> dict:
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._path)

    def leaderboard(self, date: str | None = None) -> list[dict]:
        day = self._load().get(date or self.today(), {})
        rows = [{"name": n, "score": v.get("score", 0)} for n, v in day.items()]
        rows.sort(key=lambda r: -r["score"])
        return rows[:20]

    def state(self) -> dict:
        brief = self.brief()
        return {"date": self.today(), "brief": brief,
                "banned": sorted(_brief_banned(brief)),
                "chips": CHIPS, "leaderboard": self.leaderboard()}

    # ── play ──────────────────────────────────────────────────────────────────
    def play(self, name: str, prompt: str, engine: str = "mrt2") -> dict:
        name = (name or "anon").strip()[:24] or "anon"
        prompt = (prompt or "").strip()[:200]
        if not prompt:
            return {"ok": False, "error": "Write a prompt first."}
        brief = self.brief()
        hits = _banned_hits(_brief_banned(brief), prompt)
        if hits:
            return {"ok": False, "error": "Remove the brief's words: " + ", ".join(hits)}

        forge = self._get_forge(engine if engine in ("mrt2", "sa3") else "mrt2")
        drums = any(w in prompt.lower() for w in _DRUM_WORDS)
        spec = PromptSpec(text_a=prompt, density=0.4, drums=drums, chunks=10)  # ~8s
        clip = forge.generate(spec, name)
        self._storage.put_clip(clip)

        verdict = self._judge.score_brief(prompt, brief)
        score = int(getattr(verdict, "score", 0))

        data = self._load()
        day = data.setdefault(self.today(), {})
        prev = day.get(name, {}).get("score", -1)
        improved = score > prev
        if improved:
            day[name] = {"score": score, "prompt": prompt, "clip_id": clip.id}
            self._save(data)
        return {"ok": True, "score": score, "note": getattr(verdict, "tutor_note", ""),
                "clip_url": f"/clips/{clip.id}.wav", "improved": improved,
                "best": max(score, prev), "leaderboard": self.leaderboard()}
