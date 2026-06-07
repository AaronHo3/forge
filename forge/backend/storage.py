"""
storage.py - dumb, hackathon-simple persistence.

Clips live as WAV-on-disk (written by the worker) + a metadata record here.
Crates (a user's kept sounds) and the leaderboard persist to a single JSON file
(outputs/forge_state.json) so they survive a restart. Swap to SQLite if it grows.

STATUS: Phase 1 - JSON-file backed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, replace

from .models import Clip, PromptSpec


class Storage:
    def __init__(self, root: str = "outputs"):
        self._root = root
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)
        self._state_path = os.path.join(root, "forge_state.json")
        self._clips: dict[str, Clip] = {}
        self._crates: dict[str, list[str]] = {}     # user_id -> [clip_id]
        self._leaderboard: dict[str, float] = {}    # name -> best score
        self._load()

    # ── Clips ────────────────────────────────────────────────────────────────
    def put_clip(self, clip: Clip) -> None:
        with self._lock:
            self._clips[clip.id] = clip
            self._persist()

    def get_clip(self, clip_id: str) -> Clip | None:
        return self._clips.get(clip_id)

    def rename(self, clip_id: str, name: str) -> bool:
        with self._lock:
            c = self._clips.get(clip_id)
            if c is None:
                return False
            self._clips[clip_id] = replace(c, name=name)
            self._persist()
            return True

    # ── Crates (the "keep" button) ─────────────────────────────────────────────
    def keep(self, user_id: str, clip_id: str) -> None:
        with self._lock:
            self._crates.setdefault(user_id, [])
            if clip_id not in self._crates[user_id]:
                self._crates[user_id].append(clip_id)
            self._persist()

    def unkeep(self, user_id: str, clip_id: str) -> None:
        with self._lock:
            ids = self._crates.get(user_id)
            if ids and clip_id in ids:
                ids.remove(clip_id)
                self._persist()

    def crate(self, user_id: str) -> list[Clip]:
        ids = self._crates.get(user_id, [])
        return [self._clips[i] for i in ids if i in self._clips]

    # ── Leaderboard ────────────────────────────────────────────────────────────
    def record_score(self, name: str, score: float) -> None:
        with self._lock:
            self._leaderboard[name] = max(self._leaderboard.get(name, 0.0), score)
            self._persist()

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self._leaderboard.items(), key=lambda kv: -kv[1])[:n]

    # ── Persistence (caller already holds the lock) ─────────────────────────────
    def _persist(self) -> None:
        data = {
            "clips": {
                cid: {"id": c.id, "wav_path": c.wav_path, "spec": asdict(c.spec),
                      "created_by": c.created_by, "name": c.name, "engine": c.engine}
                for cid, c in self._clips.items()
            },
            "crates": self._crates,
            "leaderboard": self._leaderboard,
        }
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._state_path)   # atomic write

    def _load(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return   # corrupt/empty state - start fresh rather than crash
        for cid, c in data.get("clips", {}).items():
            # Drop clips whose audio file no longer exists on disk.
            if not os.path.exists(c["wav_path"]):
                continue
            self._clips[cid] = Clip(
                id=c["id"], wav_path=c["wav_path"],
                spec=PromptSpec(**c["spec"]), created_by=c.get("created_by", "system"),
                name=c.get("name", ""), engine=c.get("engine", ""),
            )
        self._crates = {u: [i for i in ids if i in self._clips]
                        for u, ids in data.get("crates", {}).items()}
        self._leaderboard = data.get("leaderboard", {})
