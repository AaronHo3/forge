"""
sa3_service.py - Stable Audio 3 sidecar (runs in the SA3 venv, NOT Forge's env).

WHY THIS IS A SEPARATE PROCESS
SA3 needs torch 2.7.1 + Python 3.10; Forge runs MLX/magenta-rt on a different
Python. The two dependency stacks cannot coexist in one interpreter. So SA3 lives
here as a tiny, dependency-light HTTP service that loads the model ONCE and stays
warm (load ~5.5s, then ~1.3s per 30s clip). Forge's SA3Backend (engines/sa3.py)
talks to it over localhost HTTP.

This module imports `stable_audio_3` and therefore can ONLY be run by the SA3
venv's Python, with the SA3 repo on PYTHONPATH. Forge never imports it - it spawns
it as a subprocess. Uses the stdlib http.server on purpose: the SA3 venv has no
FastAPI, and we keep it that way.

Run (normally auto-spawned by SA3Backend, but standalone too):
    cd <SA3_REPO> && PYTHONPATH=<SA3_REPO> .venv/bin/python \
        /path/to/forge/backend/engines/sa3_service.py --port 8090

Protocol (localhost JSON):
    GET  /health   -> {"ok": true, "loaded": ["small-music", ...]}
    POST /generate {prompt, out_path, model?, duration?, steps?, cfg_scale?,
                    seed?, negative_prompt?}
                   -> {"ok": true, "duration": <sec>, "sample_rate": <hz>}
                   (writes the WAV directly to out_path - same filesystem)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torchaudio  # from the SA3 venv
from stable_audio_3 import StableAudioModel

# Models are loaded lazily and cached. One generation at a time (the model is not
# assumed thread-safe) - mirrors Forge's serial-worker discipline.
_MODELS: dict[str, StableAudioModel] = {}
_LOCK = threading.Lock()
_DEVICE = "cpu"


def _get_model(model_id: str) -> StableAudioModel:
    """Lazy-load + cache a model by id (small-music | small-sfx | ...)."""
    m = _MODELS.get(model_id)
    if m is None:
        t = time.time()
        print(f"[sa3] loading '{model_id}' (one-time)…", flush=True)
        m = StableAudioModel.from_pretrained(model_id, device=_DEVICE)
        _MODELS[model_id] = m
        print(f"[sa3] '{model_id}' ready in {time.time() - t:.1f}s", flush=True)
    return m


def _generate(req: dict) -> dict:
    """Render one request to a WAV at req['out_path']. Caller holds no lock."""
    prompt = req.get("prompt")
    out_path = req.get("out_path")
    if not prompt or not out_path:
        raise ValueError("prompt and out_path are required")

    model_id = req.get("model") or "small-music"
    duration = float(req.get("duration") or 10.0)
    steps = int(req.get("steps") or 8)
    cfg_scale = float(req.get("cfg_scale") or 1.0)
    seed = int(req.get("seed", -1))
    negative_prompt = req.get("negative_prompt") or None
    # Audio-to-audio ("Transmute"): start from an existing WAV instead of pure
    # noise. init_noise_level 0→keep original, 1→ignore it (full re-gen).
    init_audio_path = req.get("init_audio_path")
    init_noise_level = float(req.get("init_noise_level") or 1.0)

    with _LOCK:  # serialize generation across concurrent HTTP threads
        model = _get_model(model_id)
        t = time.time()
        gen_kwargs = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            duration=duration, steps=steps, cfg_scale=cfg_scale, seed=seed,
        )
        if init_audio_path:
            init_wav, in_sr = torchaudio.load(init_audio_path)   # [ch, time]
            gen_kwargs["init_audio"] = (in_sr, init_wav)          # generate() wants (sr, waveform)
            gen_kwargs["init_noise_level"] = init_noise_level
        audio = model.generate(**gen_kwargs)
        sr = int(model.model.sample_rate)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # audio: [batch, channels, samples]; batch_size defaults to 1.
        # Write 16-bit PCM (not torchaudio's default float32) to match the MRT2
        # engine's format → one uniform WAV format across all Forge engines.
        wav = audio[0].cpu().clamp(-1.0, 1.0)
        torchaudio.save(out_path, wav, sr, encoding="PCM_S", bits_per_sample=16)
        took = time.time() - t

    print(f"[sa3] {model_id} {duration:.1f}s in {took:.2f}s -> {out_path}", flush=True)
    return {"ok": True, "duration": duration, "sample_rate": sr, "took": round(took, 3)}


class _Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "loaded": list(_MODELS)})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, _generate(req))
        except ValueError as e:
            self._json(400, {"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001 - keep the service alive
            print(f"[sa3] generate FAILED: {e}", flush=True)
            self._json(500, {"ok": False, "error": str(e)})

    def log_message(self, *_args) -> None:  # silence default per-request logging
        pass


def main() -> None:
    global _DEVICE
    ap = argparse.ArgumentParser(description="Stable Audio 3 sidecar service")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preload", default="", help="comma-separated model ids to load at startup")
    args = ap.parse_args()
    _DEVICE = args.device

    for mid in (m.strip() for m in args.preload.split(",") if m.strip()):
        _get_model(mid)

    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[sa3] serving on http://{args.host}:{args.port} (device={_DEVICE})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
