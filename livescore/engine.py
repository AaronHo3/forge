"""
engine.py — ScoreEngine
-----------------------
The live Score-the-Story pipeline, wrapped as a controllable object so it can be
driven by a UI (app.py) instead of only the terminal (main.py).

It owns the same components main.py wires together — VoiceAnalyzer → FeatureMapper
→ MRTController, with the LLM director and keepsake capture — and exposes simple
start()/stop()/inject()/state() methods plus the keepsake path on shutdown.

This is the seam that lets a non-technical user click buttons: the web control
panel calls these methods; nothing about the audio pipeline changes.
"""

from __future__ import annotations

import os
import threading
import time

import paths
from voice_analyzer import VoiceAnalyzer
from feature_mapper import FeatureMapper
from mrt_controller import MRTController
from llm_style_director import LLMStyleDirector
from telemetry import Telemetry
from presets import get as get_preset
from speaker_signature import SpeakerSignature
from keepsake import SessionLog
from session_summary import build_summary, write_summary

UPDATE_HZ = 20   # parameter pushes per second to MRT2 (matches main.py)


class ScoreEngine:
    """A start/stoppable instance of the full live scoring pipeline."""

    def __init__(self, mode: str = "python", preset_name: str = "storytelling",
                 telemetry: Telemetry | None = None, telemetry_port: int = 8765):
        self.mode = mode
        self.preset = get_preset(preset_name)
        self.telemetry_port = telemetry_port

        # Telemetry is a PERSISTENT service owned by the caller (app.py): it is
        # injected here and the engine never starts or stops it. That keeps the
        # live dashboard (and the UI's iframe) alive across start/stop cycles.
        self._telemetry: Telemetry | None = telemetry

        self._running = False
        self.phase = "idle"          # idle | starting | running | stopping | error
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._t0 = 0.0

        # Per-session components — created fresh on each start() so a stop/start
        # is clean. Telemetry is intentionally NOT in this list.
        self._analyzer: VoiceAnalyzer | None = None
        self._mapper: FeatureMapper | None = None
        self._controller: MRTController | None = None
        self._detector: LLMStyleDirector | None = None
        self._session_log: SessionLog | None = None
        self._signature: SpeakerSignature | None = None

        self.last_keepsake: str | None = None   # session JSON (data log)
        self.last_song: str | None = None        # rendered offline .wav
        self.last_summary: str | None = None      # structured session report

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    # Heavy work (mic open, MRT2 model load) runs OUTSIDE the lock so /state stays
    # responsive while `phase` reports progress. Callers may run these on a thread.
    def start(self) -> None:
        with self._lock:
            if self._running or self.phase == "starting":
                return
            self.phase = "starting"
            self._stop_event = threading.Event()
        try:
            preset = self.preset
            analyzer = VoiceAnalyzer()
            mapper = FeatureMapper(smoothing=preset.smoothing,
                                   drums_threshold=preset.drums_threshold)
            controller = MRTController(
                mode=self.mode, morph_step=preset.morph_step,
                default_a=preset.default_a, default_b=preset.default_b,
                default_key=preset.default_key, telemetry=self._telemetry,
                enable_drums=preset.enable_drums)
            signature = SpeakerSignature()
            session_log = SessionLog()
            detector = (LLMStyleDirector(
                analyzer, controller,
                transcribe_interval=preset.transcribe_interval,
                audio_window=preset.audio_window, cooldown=preset.cooldown,
                style_hint=preset.style_hint, telemetry=self._telemetry,
                signature=signature, session_log=session_log)
                if self.mode == "python" else None)

            analyzer.start()
            controller.start()
            if detector:
                detector.start()

            with self._lock:
                self._analyzer, self._mapper, self._controller = analyzer, mapper, controller
                self._detector, self._session_log = detector, session_log
                self._signature = signature
                self._t0 = time.monotonic()
                self._running = True
                self.phase = "running"
            threading.Thread(target=self._control_loop, daemon=True).start()
        except Exception as e:  # noqa: BLE001 — keep the server alive, report it
            print(f"[engine] start failed: {e}")
            with self._lock:
                self.phase = "error"
                self._running = False

    def stop(self) -> str | None:
        """Stop the pipeline and save the keepsake. Returns the keepsake path.
        Leaves the (shared) telemetry server running."""
        with self._lock:
            if not self._running:
                return self.last_keepsake
            self.phase = "stopping"
            self._running = False
            self._stop_event.set()
            analyzer, controller = self._analyzer, self._controller
            detector, session_log = self._detector, self._session_log
            signature = self._signature
        # Component shutdowns are non-blocking (event flags / stream close), but
        # do them outside the lock so /state never blocks on them.
        for comp in (analyzer, controller, detector):
            try:
                if comp:
                    comp.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[engine] stop warning: {e}")
        # Stamp the keepsake with the speaker signature (the BLEND of every voice
        # that spoke) BEFORE saving, so the offline render is personalised — the
        # same step main.py does. Without it the render falls back to a default.
        if session_log is not None and signature is not None:
            try:
                session_log.set_signature(signature.signature())
            except Exception as e:  # noqa: BLE001
                print(f"[engine] signature warning: {e}")
        path = session_log.save() if session_log else None
        if path:
            self.last_keepsake = path
            try:
                summary = build_summary(
                    duration_s=session_log.elapsed(),
                    scenes=session_log.scenes(),
                    perf=controller.perf_stats() if controller else None,
                    health=controller.health() if controller else None,
                )
                self.last_summary = write_summary(summary, paths.summary_for(path))
            except Exception as e:  # noqa: BLE001 — summary is best-effort
                print(f"[engine] summary warning: {e}")
        with self._lock:
            self.phase = "idle"
        return self.last_keepsake

    def render_keepsake(self, session_path: str | None = None,
                        dry_run: bool = False) -> str | None:
        """Render a saved session JSON into the offline keepsake song (.wav).

        Heavy: loads its OWN MRT2 and generates layered per-scene stems (minutes).
        Defaults to the most recent telling. Returns the .wav path, or None if
        there's nothing to render.

        NOTE: only call this from a process that has NOT already initialised MLX
        for the live engine. After a live session in the same process, MLX's
        per-thread GPU stream state makes a second in-process model load fail
        ("no Stream(gpu, 1) in current thread") — render in a subprocess instead
        (`python keepsake.py render <json>`), as app.py does on quit."""
        src = session_path or self.last_keepsake
        if not src or not os.path.exists(src):
            return None
        from keepsake import render_keepsake as _render   # lazy: heavy import
        self.last_song = _render(src, dry_run=dry_run)
        return self.last_song

    # ── Live control ──────────────────────────────────────────────────────────
    def inject(self, text: str) -> bool:
        if self._detector and text.strip():
            self._detector.inject(text.strip())
            return True
        return False

    def toggle_mute(self) -> bool:
        return self._detector.toggle_mute() if self._detector else False

    # ── State for the UI ────────────────────────────────────────────────────────
    def state(self) -> dict:
        running = self._running
        scenes = self._session_log.scenes() if self._session_log else []
        det = self._detector
        return {
            "running": running,
            "phase": self.phase,
            "mode": self.mode,
            "preset": self.preset.name,
            "elapsed": round(time.monotonic() - self._t0, 1) if running else 0.0,
            "status": (det.last_status if det else "—"),
            "muted": (det.muted if det else False),
            "current_a": (det.current_a if det else "—"),
            "current_b": (det.current_b if det else "—"),
            "scenes": scenes,
            "scene_count": len(scenes),
            "telemetry_port": self.telemetry_port,
            "last_keepsake": self.last_keepsake,
        }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _control_loop(self) -> None:
        interval = 1.0 / UPDATE_HZ
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            features = self._analyzer.get_features()
            params = self._mapper.update(features)
            self._controller.update(params)
            if self._telemetry:
                out = self._controller.output_stats() or {}
                self._telemetry.record(features, params,
                                       out.get("level", 0.0), out.get("bright", 0.0),
                                       out.get("spectrum"))
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))
