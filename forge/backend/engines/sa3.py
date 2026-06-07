"""
sa3.py - Stable Audio 3 engine (Forge side).

A thin client conforming to engines.base.Backend. It talks to the SA3 sidecar
(sa3_service.py) over localhost HTTP - SA3's torch stack can't share Forge's
interpreter (see sa3_service.py for the why). This module imports NO torch; only
the stdlib. It optionally auto-spawns the sidecar using the SA3 venv's Python.

Config (all via env, with spike-friendly defaults):
    FORGE_SA3_URL        sidecar base URL          (default http://127.0.0.1:8090)
    FORGE_SA3_DIR        SA3 repo checkout          (default the spike clone)
    FORGE_SA3_PYTHON     python that has SA3        (default <DIR>/.venv/bin/python)
    FORGE_SA3_MODEL      model id for render        (default small-music)
    FORGE_SA3_DEVICE     torch device for sidecar   (default cpu)
    FORGE_SA3_AUTOSPAWN  auto-start sidecar?        (default 1)

Run Forge on SA3:  FORGE_BACKEND=sa3 uvicorn backend.server:app --port 8000
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from ..models import PromptSpec

_DEFAULT_DIR = "/Users/aaronho/Desktop/MusicHack/sa3-spike/stable-audio-3"
_SECONDS_PER_CHUNK = 0.8   # match MRT2's CHUNK_FRAMES≈0.8s so the UI slider means the same thing


class SA3Backend:
    """Stable Audio 3 via a warm localhost sidecar. Engine name: 'sa3'."""

    name = "sa3"

    def __init__(self) -> None:
        self._url = (os.environ.get("FORGE_SA3_URL") or "http://127.0.0.1:8090").rstrip("/")
        self._dir = os.environ.get("FORGE_SA3_DIR") or _DEFAULT_DIR
        self._python = os.environ.get("FORGE_SA3_PYTHON") or os.path.join(self._dir, ".venv", "bin", "python")
        self._model = os.environ.get("FORGE_SA3_MODEL") or "small-music"
        self._device = os.environ.get("FORGE_SA3_DEVICE") or "cpu"
        self._autospawn = (os.environ.get("FORGE_SA3_AUTOSPAWN") or "1") != "0"
        self._proc: subprocess.Popen | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Ensure the sidecar is up (idempotent). Auto-spawn + wait if configured."""
        if self._healthy():
            return
        if not self._autospawn:
            print(f"[sa3] sidecar not running at {self._url} and autospawn off - "
                  f"start it manually (see engines/sa3_service.py).")
            return
        self._spawn()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._url}/health", timeout=2) as r:
                return json.loads(r.read()).get("ok") is True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _spawn(self) -> None:
        service = os.path.join(os.path.dirname(__file__), "sa3_service.py")
        if not os.path.exists(self._python):
            raise RuntimeError(
                f"[sa3] SA3 venv python not found at {self._python}. Set FORGE_SA3_DIR "
                f"/ FORGE_SA3_PYTHON to your Stable Audio 3 checkout."
            )
        port = self._url.rsplit(":", 1)[-1]
        env = {**os.environ, "PYTHONPATH": self._dir}   # so the sidecar can import stable_audio_3
        print(f"[sa3] spawning sidecar: {self._python} {service} --port {port}")
        self._proc = subprocess.Popen(
            [self._python, service, "--port", port, "--device", self._device,
             "--preload", self._model],
            cwd=self._dir, env=env,
        )
        atexit.register(self._terminate)   # we spawned it → we clean it up
        # Wait for the model to load (first time can include a cached-weight load).
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._healthy():
                print("[sa3] sidecar ready.")
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"[sa3] sidecar exited early (code {self._proc.returncode})")
            time.sleep(1.0)
        raise TimeoutError("[sa3] sidecar did not become healthy within 120s")

    def _terminate(self) -> None:
        """Stop the sidecar IF we spawned it (leave a pre-existing one running)."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ── Render ──────────────────────────────────────────────────────────────────
    def render_blocking(
        self, spec: PromptSpec, out_path: str, timeout: float = 300.0,
    ) -> tuple[str, None]:
        """Text-to-audio. Render `spec` to a WAV at `out_path`. Returns
        (out_path, None) - SA3 has no MusicCoCa-style embedding for novelty."""
        self._post_generate({
            "prompt": self._prompt(spec),
            "out_path": os.path.abspath(out_path),   # sidecar cwd differs - must be absolute
            "model": self._model,
            "duration": max(1.0, spec.chunks * _SECONDS_PER_CHUNK),
            "seed": -1,                              # random per call → natural variety
        }, timeout)
        return out_path, None

    def render_audio_to_audio(
        self, src_wav_path: str, prompt: str, out_path: str, *,
        init_noise_level: float = 0.8, duration: float | None = None,
        timeout: float = 300.0,
    ) -> tuple[str, None]:
        """Audio-to-audio ("Transmute"): reinterpret an existing WAV toward
        `prompt`. init_noise_level 0→preserve source, 1→ignore it. Source may be
        ANY WAV - MRT2 or SA3 - so this is the MRT2→SA3 bridge."""
        self._post_generate({
            "prompt": prompt,
            "out_path": os.path.abspath(out_path),
            "model": self._model,
            "init_audio_path": os.path.abspath(src_wav_path),
            "init_noise_level": max(0.0, min(1.0, init_noise_level)),
            "duration": duration if duration else 10.0,
            "seed": -1,
        }, timeout)
        return out_path, None

    def _post_generate(self, payload: dict, timeout: float) -> dict:
        """POST a job to the sidecar /generate and return its JSON (or raise)."""
        self.start()  # idempotent safety net if the startup hook didn't run
        req = urllib.request.Request(
            f"{self._url}/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"[sa3] generate failed ({e.code}): {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"[sa3] sidecar unreachable at {self._url}: {e}") from e
        if not body.get("ok"):
            raise RuntimeError(f"[sa3] generate error: {body.get('error')}")
        return body

    @staticmethod
    def _prompt(spec: PromptSpec) -> str:
        """Flatten a PromptSpec into one SA3 text prompt. SA3 has no native A↔B
        style-vector blend or key conditioning, so text_b and key fold into the
        prompt text (a pragmatic mapping, not MRT2's interpolation)."""
        parts = [spec.text_a]
        if spec.text_b:
            parts.append(spec.text_b)
        prompt = ", ".join(parts)
        if spec.key:
            prompt += f", in {spec.key}"
        return prompt
