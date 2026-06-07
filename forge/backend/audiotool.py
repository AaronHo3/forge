"""
audiotool.py - bridge from Forge (Python) to the Audiotool Nexus SDK (Node).

"Send to Audiotool" turns a crafted Forge clip into a real building block: it
uploads the clip as a sample and drops it onto a project's timeline, then hands
back the studio URL. Editing/arranging (and multiplayer collaboration) happens in
Audiotool itself - we just get the clips in.

The Nexus SDK is JavaScript-only, so this talks over localhost HTTP to the Node
sidecar (audiotool_sidecar/server.mjs), auto-spawning it like the SA3 engine.
Imports nothing beyond the stdlib.

Config (env):
    FORGE_AUDIOTOOL_URL    sidecar base URL    (default http://127.0.0.1:8091)
    FORGE_AUDIOTOOL_DIR    sidecar directory   (default forge/audiotool_sidecar)
    FORGE_AUDIOTOOL_NODE   node binary         (default "node")
    AUDIOTOOL_PAT          Personal Access Token (REQUIRED to actually send)
    FORGE_AUDIOTOOL_AUTOSPAWN  auto-start sidecar? (default 1)
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(__file__)
_DEFAULT_DIR = os.path.abspath(os.path.join(_HERE, "..", "audiotool_sidecar"))


class AudiotoolBridge:
    """Sends Forge clips into Audiotool via the Node Nexus sidecar."""

    def __init__(self) -> None:
        self._url = (os.environ.get("FORGE_AUDIOTOOL_URL") or "http://127.0.0.1:8091").rstrip("/")
        self._dir = os.environ.get("FORGE_AUDIOTOOL_DIR") or _DEFAULT_DIR
        self._node = os.environ.get("FORGE_AUDIOTOOL_NODE") or "node"
        self._pat = os.environ.get("AUDIOTOOL_PAT") or ""
        self._autospawn = (os.environ.get("FORGE_AUDIOTOOL_AUTOSPAWN") or "1") != "0"
        self._proc: subprocess.Popen | None = None

    @property
    def configured(self) -> bool:
        """True once a PAT is available (the only hard requirement to send)."""
        return bool(self._pat)

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Ensure the sidecar is up (idempotent). No-op if no PAT is configured."""
        if not self._pat:
            return
        if self._healthy():
            return
        if self._autospawn:
            self._spawn()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._url}/health", timeout=2) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return False
        if d.get("ok") is not True:
            return False
        # Never reuse an UNauthenticated sidecar when we hold a PAT - that's the
        # stale-leftover trap. Force a fresh, authed spawn instead.
        if self._pat and not d.get("authed"):
            return False
        return True

    def _spawn(self) -> None:
        server = os.path.join(self._dir, "server.mjs")
        if not os.path.exists(server):
            raise RuntimeError(f"[audiotool] sidecar not found at {server}")
        if not os.path.isdir(os.path.join(self._dir, "node_modules")):
            raise RuntimeError(f"[audiotool] run `npm install` in {self._dir} first")
        port = self._url.rsplit(":", 1)[-1]
        self._free_stale_port(port)   # evict any unauthed squatter so we can bind
        env = {**os.environ, "PORT": port, "AUDIOTOOL_PAT": self._pat}
        print(f"[audiotool] spawning sidecar: {self._node} {server} (port {port})")
        self._proc = subprocess.Popen([self._node, server], cwd=self._dir, env=env)
        atexit.register(self._terminate)
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._healthy():
                print("[audiotool] sidecar ready.")
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"[audiotool] sidecar exited early (code {self._proc.returncode})")
            time.sleep(0.5)
        raise TimeoutError("[audiotool] sidecar did not become healthy within 30s")

    @staticmethod
    def _free_stale_port(port: str) -> None:
        """Kill any process squatting on our port (e.g. a stale unauthed sidecar
        from a previous run) so a fresh authed one can bind. Best-effort; macOS/Linux."""
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                                 capture_output=True, text=True, timeout=5).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return
        for pid in out:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        if out:
            time.sleep(1.0)

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ── Send ────────────────────────────────────────────────────────────────────
    def send_clip(
        self, wav_path: str, display_name: str = "", *,
        project: str | None = None, bpm: int | None = None, timeout: float = 120.0,
    ) -> dict:
        """Upload `wav_path` as a sample and insert it into an Audiotool project.
        Returns {project_url, project_name, sample_name}. Pass the returned
        `project_name` back as `project` to keep adding clips to one project."""
        if not self._pat:
            raise RuntimeError(
                "Audiotool is not connected - set AUDIOTOOL_PAT (get a Personal "
                "Access Token at https://rpc.audiotool.com/dev) and restart Forge."
            )
        if not wav_path or not os.path.exists(wav_path):
            raise ValueError("clip audio not found")
        payload = {"wav_path": os.path.abspath(wav_path), "display_name": display_name}
        if project:
            payload["project"] = project
        if bpm:
            payload["bpm"] = bpm
        return self._post("/send", payload, timeout)

    def project_info(self, project: str, *, timeout: float = 30.0) -> dict:
        """Read a project's musical context (tempo + time signature) via Nexus.
        Accepts a project URL, `projects/{uuid}` name, or bare uuid. No track
        required - tempo lives on the project-level config entity. Returns
        {project_name, project_url, tempo_bpm, signature_num, signature_den}."""
        if not project or not project.strip():
            raise ValueError("project link is required")
        return self._post("/project-info", {"project": project.strip()}, timeout)

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        """POST JSON to the sidecar (auto-starting it) and return its parsed body."""
        if not self._pat:
            raise RuntimeError(
                "Audiotool is not connected - set AUDIOTOOL_PAT (get a Personal "
                "Access Token at https://rpc.audiotool.com/dev) and restart Forge."
            )
        self.start()
        req = urllib.request.Request(
            f"{self._url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"[audiotool] {path} failed ({e.code}): {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"[audiotool] sidecar unreachable at {self._url}: {e}") from e
        if not body.get("ok"):
            raise RuntimeError(f"[audiotool] {body.get('error')}")
        return body
