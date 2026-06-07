"""
mrt_worker.py - the single owner of the MRT2 model.

WHY THIS EXISTS (see GAME_PLAN.md §7.2):
MRT2 (MagentaRT2SystemMlxfn) is a heavy MLX model, and MLX's per-thread GPU
stream state means it cannot be safely loaded twice in one process. So exactly
ONE worker owns ONE model, and every generation request - solo Forge, every
telephone hop, every battle submission - is funneled through a single job queue
and processed SERIALLY on the worker thread. This is the constraint that makes
pre-generation and turn-based pacing the right design rather than a workaround.

CRITICAL: all MLX work (model load, embed_style, generate) happens on the worker
thread and nowhere else, mirroring the fallback's mrt_controller.py.

The generate loop is ported from mrt_controller.py:_gen_chunk - embed_style →
generate(... state=state) in a loop, threading `state` chunk-to-chunk,
concatenating to one stereo WAV.

STATUS: Phase 0 - implemented. `python -m backend.mrt_worker --selftest`.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .harmony import chord_notes
from .models import PromptSpec

# Marker that the live notes vector came from an explicit chord (Harmony Sandbox),
# so clearing the chord falls back to the key-derived constraint.
_NOTES_OVERRIDE = object()

SAMPLE_RATE = 48_000
CHUNK_FRAMES = 20      # ~0.8s per generate() call (matches mrt_controller.py)

# Guidance (classifier-free) band - density maps into [CFG_BASE, CFG_BASE+CFG_SPAN].
CFG_BASE = 2.0
CFG_SPAN = 1.0
TEMPERATURE = 1.0
TOP_K = 32

# Output conditioning DSP (ported from the fallback's mrt_controller.py). WITHOUT
# this, raw MRT2 chunks come out at wildly inconsistent levels and the
# autoregressive stream drifts toward silence - which is exactly why ungated MRT2
# output sounds weak/noisy. This chain is what makes MRT2 actually sound good.
TARGET_RMS = 0.14       # loudness-normalize every chunk to the same perceived level
LIMIT_THRESH = 0.80     # soft-knee limiter: tanh-compress transients above this
AGC_MIN_RMS = 0.02      # below this a chunk is a transient, not music - hold the gain
AGC_GAIN_MIN = 0.30     # clamp auto-gain to a sane band so it can never explode
AGC_GAIN_MAX = 3.00
DECAY_RMS = 0.015       # raw chunk RMS below this = "dying" (silence is a stable attractor)
DECAY_CHUNKS = 5        # consecutive near-silent chunks before we re-seed the state
JAM_LEAD = 0.4          # jam: stay only this many seconds ahead of real-time (low steering latency)
JAM_FRAMES = 10         # jam: smaller chunks (~0.4s) so steering lands sooner. Clip gen still uses CHUNK_FRAMES.

# NOTE: this MLX build of MRT2 generates DETERMINISTICALLY. The compiled graph
# ignores mx.random.seed AND mx.random.key (verified: seed 1 vs 999 → byte-identical
# audio), so identical inputs always yield identical output. The only way to get a
# different result is to change the INPUT (prompt / blend / density / key). That's
# fine - games use distinct prompts per round, and "variations" sweep an input
# (see forge_core.variations).

# Music-theory constants for the optional key anchor (ported from mrt_controller).
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
NOTE_LO, NOTE_HI = 36, 84


def _build_notes(key: str | None, num_notes: int) -> list[int] | None:
    """Soft key constraint: mask in-key pitches (-1, played freely) and turn OFF
    (0) out-of-key pitches. Keeps the music in key but free to evolve. Returns
    None on any problem (free harmony)."""
    if not key:
        return None
    try:
        k = key.strip()
        pc = _PC[k[0].upper()]
        if len(k) > 1 and k[1] in "#b":
            pc += 1 if k[1] == "#" else -1
        pc %= 12
        mode = _MAJOR if "maj" in k.lower() else _MINOR
        scale = {(pc + iv) % 12 for iv in mode}
        notes = [-1] * num_notes
        for midi in range(min(128, num_notes)):
            if NOTE_LO <= midi <= NOTE_HI and (midi % 12) not in scale:
                notes[midi] = 0
        return notes
    except Exception:
        return None


@dataclass
class GenJob:
    """One unit of work for the worker: a spec to render into a WAV file."""
    spec: PromptSpec
    out_path: str
    job_id: str
    done: Callable[[str, tuple], None] | None = None   # called with (wav_path, embedding)
    error: Callable[[Exception], None] | None = None


@dataclass
class JamJob:
    """A long-running real-time jam: occupies the worker thread until state.running
    goes False, reading live params from `state` and emitting audio via `emit`."""
    state: object                                      # jam.JamState (duck-typed to avoid a cycle)
    emit: Callable[[bytes], None]                      # called with int16 interleaved-stereo PCM


class MRTWorker:
    """Loads MRT2 once; serves GenJobs from a queue on a single thread.

    Conforms structurally to engines.base.Backend (start + render_blocking + name),
    so it is a drop-in Forge engine without inheriting anything.
    """

    name = "mrt2"

    def __init__(self, model_size: str = "mrt2_base"):
        self._model_size = model_size
        self._mrt = None                          # the loaded MagentaRT2System
        self._jobs: queue.Queue[GenJob] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the worker thread. The model loads lazily on the first job."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────
    def submit(self, spec: PromptSpec, out_path: str,
               done: Callable[[str], None] | None = None,
               error: Callable[[Exception], None] | None = None) -> str:
        """Enqueue a generation. Returns a job id; `done(wav_path)` fires later
        on the worker thread (or `error(exc)` on failure)."""
        job_id = uuid.uuid4().hex[:12]
        self._jobs.put(GenJob(spec, out_path, job_id, done, error))
        return job_id

    def submit_jam(self, state: object, emit: Callable[[bytes], None]) -> None:
        """Start a real-time jam on the worker thread (occupies it until stopped)."""
        self._jobs.put(JamJob(state, emit))

    def render_blocking(self, spec: PromptSpec, out_path: str,
                        timeout: float = 300.0) -> tuple[str, tuple]:
        """Convenience: submit and wait. Returns (wav_path, style_embedding).
        The actual render still runs on the single worker thread."""
        result: dict[str, object] = {}
        evt = threading.Event()
        self.submit(
            spec, out_path,
            done=lambda p, e: (result.update(path=p, emb=e), evt.set()),
            error=lambda e: (result.__setitem__("err", e), evt.set()),
        )
        if not evt.wait(timeout):
            raise TimeoutError(f"generation timed out after {timeout}s")
        if "err" in result:
            raise result["err"]  # type: ignore[misc]
        return result["path"], result.get("emb")  # type: ignore[return-value]

    # ── Internal (worker thread only) ────────────────────────────────────────
    def _ensure_model(self):
        """Load MRT2 exactly once, on the worker thread."""
        if self._mrt is not None:
            return self._mrt
        from magenta_rt.mlx.system import MagentaRT2SystemMlxfn  # noqa: PLC0415
        print(f"[mrt_worker] loading model '{self._model_size}' (one-time, heavy)...")
        self._mrt = MagentaRT2SystemMlxfn(size=self._model_size)
        print("[mrt_worker] model ready.")
        return self._mrt

    def _render(self, spec: PromptSpec) -> tuple[np.ndarray, tuple]:
        """Render a PromptSpec to (stereo float32 audio, style embedding).

        Ported from mrt_controller.py: embed the A pole (and B if given),
        linearly interpolate by `blend`, optionally constrain to a key, then loop
        generate() threading `state` across chunks so the music evolves coherently.
        The blended style vector is returned too - it powers novelty scoring.
        """
        mrt = self._mrt
        style_a = mrt.embed_style(spec.text_a)
        if spec.text_b:
            style_b = mrt.embed_style(spec.text_b)
            style = (1.0 - spec.blend) * style_a + spec.blend * style_b
        else:
            style = style_a
        embedding = tuple(float(x) for x in np.asarray(style).ravel())

        notes = (chord_notes(spec.chord[0], spec.chord[1], mrt._num_notes) if spec.chord
                 else _build_notes(spec.key, mrt._num_notes))
        drums = [1] if spec.drums else [0]        # 1 = on, 0 = off (per generate() doc)
        cfg = CFG_BASE + max(0.0, min(1.0, spec.density)) * CFG_SPAN

        # Stream chunks through the fallback's quality chain: per-chunk loudness
        # AGC (every chunk to TARGET_RMS, clamped so silent chunks can't explode),
        # a soft limiter, and a decay watchdog that re-seeds the state when the
        # stream drifts toward silence. This is the difference between MRT2 sounding
        # good vs. weak/noisy.
        out: list[np.ndarray] = []
        state = None
        cur_g = 1.0          # smoothed auto-gain
        low_streak = 0       # consecutive near-silent chunks (decay watchdog)
        for _ in range(max(1, spec.chunks)):
            reseed = low_streak >= DECAY_CHUNKS
            wav, state = mrt.generate(
                style=style, notes=notes, drums=drums,
                cfg_musiccoca=cfg, frames=CHUNK_FRAMES,
                state=(None if reseed else state),
                temperature=TEMPERATURE, top_k=TOP_K,
            )
            ci = np.ascontiguousarray(wav.samples).astype(np.float32)
            raw_rms = float(np.sqrt(np.mean(ci ** 2)))
            low_streak = 0 if (reseed or raw_rms >= DECAY_RMS) else low_streak + 1
            if reseed:
                cur_g = 1.0                          # restart the AGC cleanly
            if raw_rms > AGC_MIN_RMS:                # only adapt the gain on real music
                cur_g = 0.85 * cur_g + 0.15 * (TARGET_RMS / raw_rms)
                cur_g = min(AGC_GAIN_MAX, max(AGC_GAIN_MIN, cur_g))
            out.append(self._soft_limit(ci * cur_g))

        audio = np.concatenate(out, axis=0).astype(np.float32)
        if audio.ndim == 1:                       # mono → stereo
            audio = np.stack([audio, audio], axis=1)
        return audio, embedding

    def run_jam(self, state, emit) -> None:
        """Continuous real-time generation - the MRT2 jam/morph stream. Runs on the
        worker thread until `state.running` is False. Each ~0.8s chunk: read the live
        params (re-embedding style on a prompt change, on THIS thread as MLX requires),
        blend A↔B, apply the same loudness/limiter/decay DSP as _render, append to the
        session record, and emit int16 interleaved-stereo PCM for browser playback."""
        self._ensure_model()
        mrt = self._mrt

        cur_a = cur_b = cur_key = object()   # sentinels → force first embed
        cur_chord = object()                 # sentinel → force first chord build
        style_a = style_b = None
        notes = None
        chunk_state = None
        cur_g = 1.0
        low_streak = 0

        # Real-time pacing: MRT2 generates faster than real-time, so without this
        # the audio buffer grows and steering changes are heard SECONDS late. Keep
        # generation only ~JAM_LEAD seconds ahead of the wall clock so the live
        # response stays tight (low latency) while still leaving an underrun cushion.
        t0 = time.monotonic()
        generated = 0.0

        while getattr(state, "running", False):
            # live prompt/key changes → re-embed on the worker thread
            if state.prompt_a != cur_a:
                cur_a = state.prompt_a
                style_a = mrt.embed_style(cur_a) if cur_a else None
            if state.prompt_b != cur_b:
                cur_b = state.prompt_b
                style_b = mrt.embed_style(cur_b) if cur_b else None
            # Harmony Sandbox: an explicit chord overrides the soft key constraint.
            # Build the chord's note vector here on the worker thread (we know
            # mrt._num_notes), rebuilding only when the chord actually changes.
            chord = getattr(state, "chord", None)
            if chord is not None:
                if chord != cur_chord:
                    cur_chord = chord
                    notes = chord_notes(chord[0], chord[1], mrt._num_notes)
                    cur_key = _NOTES_OVERRIDE      # so clearing the chord re-derives from key
            elif cur_key is _NOTES_OVERRIDE or state.key != cur_key:
                cur_key = state.key
                cur_chord = object()
                notes = _build_notes(cur_key, mrt._num_notes)

            blend = max(0.0, min(1.0, state.blend))
            if style_a is not None and style_b is not None:
                style = (1.0 - blend) * style_a + blend * style_b
            else:
                style = style_a if style_a is not None else style_b

            cfg = CFG_BASE + max(0.0, min(1.0, state.density)) * CFG_SPAN
            drums = [1] if state.drums else [0]
            reseed = low_streak >= DECAY_CHUNKS
            wav, chunk_state = mrt.generate(
                style=style, notes=notes, drums=drums,
                cfg_musiccoca=cfg, frames=JAM_FRAMES,        # smaller jam chunk → lower steering latency
                state=(None if reseed else chunk_state),
                temperature=(getattr(state, "temperature", None) or TEMPERATURE),  # live "chaos" knob
                top_k=(getattr(state, "top_k", None) or TOP_K),                     # live "focus" knob
            )
            ci = np.ascontiguousarray(wav.samples).astype(np.float32)
            raw_rms = float(np.sqrt(np.mean(ci ** 2)))
            low_streak = 0 if (reseed or raw_rms >= DECAY_RMS) else low_streak + 1
            if reseed:
                cur_g = 1.0
            if raw_rms > AGC_MIN_RMS:
                cur_g = 0.85 * cur_g + 0.15 * (TARGET_RMS / raw_rms)
                cur_g = min(AGC_GAIN_MAX, max(AGC_GAIN_MIN, cur_g))
            ci = self._soft_limit(ci * cur_g)
            if ci.ndim == 1:
                ci = np.stack([ci, ci], axis=1)

            state.append_record(ci)
            pcm = (np.clip(ci, -1.0, 1.0) * 32767.0).astype("<i2")  # int16 LE, interleaved
            try:
                emit(pcm.tobytes())
            except Exception:  # noqa: BLE001 - a slow/closed consumer must not kill the loop
                pass

            # pace to ~real-time so steering stays low-latency (see JAM_LEAD note above)
            generated += ci.shape[0] / SAMPLE_RATE
            ahead = (t0 + generated) - time.monotonic()
            if ahead > JAM_LEAD:
                time.sleep(ahead - JAM_LEAD)

    @staticmethod
    def _soft_limit(x: np.ndarray) -> np.ndarray:
        """Soft-knee limiter: linear below LIMIT_THRESH, tanh-compressed above, so
        loud transients are tamed smoothly instead of clipping into harsh static."""
        th = LIMIT_THRESH
        a = np.abs(x)
        over = a > th
        if over.any():
            x = x.copy()
            x[over] = np.sign(x[over]) * (th + (1.0 - th) * np.tanh((a[over] - th) / (1.0 - th)))
        return x

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray) -> None:
        """Write a stereo float32 array to a 16-bit PCM WAV (broad compatibility)."""
        from scipy.io import wavfile  # noqa: PLC0415
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        wavfile.write(path, SAMPLE_RATE, pcm)

    def _run(self) -> None:
        while self._running:
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(job, JamJob):
                try:
                    self.run_jam(job.state, job.emit)
                except Exception as e:  # noqa: BLE001 - keep the worker alive
                    print(f"[mrt_worker] jam FAILED: {e}")
                continue
            try:
                self._ensure_model()
                audio, embedding = self._render(job.spec)
                self._write_wav(job.out_path, audio)
                dur = audio.shape[0] / SAMPLE_RATE
                print(f"[mrt_worker] job {job.job_id} → {job.out_path} ({dur:.1f}s)")
                if job.done:
                    job.done(job.out_path, embedding)
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                print(f"[mrt_worker] job {job.job_id} FAILED: {e}")
                if job.error:
                    job.error(e)


def _selftest() -> None:
    """prompt → wav, no web layer. Proves the whole Phase-0 path end-to-end."""
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "selftest.wav")
    out = os.path.abspath(out)
    spec = PromptSpec(
        text_a="warm mellow lo-fi rhodes keys",
        text_b="bright fingerpicked acoustic guitar",
        blend=0.4, key="A minor", density=0.3, drums=False, chunks=4,
    )
    worker = MRTWorker()
    worker.start()
    print(f"[selftest] rendering: {spec.text_a!r} ↔ {spec.text_b!r} @ blend {spec.blend}")
    path, emb = worker.render_blocking(spec, out)
    print(f"[selftest] embedding dims: {len(emb) if emb else 0}")
    size = os.path.getsize(path)
    print(f"[selftest] OK → {path} ({size} bytes)")
    worker.stop()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("MRTWorker. Run with --selftest to render a prompt → wav.")
