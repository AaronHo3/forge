"""
looper.py - the Per-Instrument Looper engine.

Each instrument is a short loop MRT2 renders on demand. Loops play stacked and in
sync; you mix them like a band. This does NOT use the live stream (run_jam); it
rides the normal render-job queue, so the single model serves quick loop renders
serially and a browser (later a server) mixer loops and sums the layers.

THE HARD PART: MRT2 has no tempo input, so two independent renders land at
different tempos and will not stack. So every loop is tempo-aligned: beat-track it,
time-stretch to the session BPM, and crop to an exact bar grid. Every layer comes
out the same length and tempo and loops seamlessly.

A render returns a buffer of several bar-aligned WINDOWS (not one loop), so the
browser can re-window for a fresh "take" with no extra model call. The MLX build is
deterministic, so re-rolling the model would give the identical loop; sliding a
window over a longer render is how we get variation for free.
"""

from __future__ import annotations

import os
import uuid

import numpy as np

from .models import Clip, PromptSpec

SR = 48_000
BEATS_PER_BAR = 4
MAX_WINDOWS = 4          # how many bar-aligned loop windows one render yields

# The instrument bank. Prompts push MRT2 to stay on one instrument (it renders a
# full mix, so isolation is approximate). The drummer uses the drums flag.
INSTRUMENTS: dict[str, dict] = {
    "drums":   {"label": "Drums",      "icon": "🥁", "drums": True,  "density": 0.55,
                "prompt": "tight punchy acoustic drum kit groove"},
    "bass":    {"label": "Bass",       "icon": "🎸", "drums": False, "density": 0.45,
                "prompt": "deep electric bass line, groovy, no drums, no other instruments"},
    "keys":    {"label": "Keys",       "icon": "🎹", "drums": False, "density": 0.45,
                "prompt": "warm electric piano chords, rhythmic, no drums"},
    "piano":   {"label": "Piano",      "icon": "🎼", "drums": False, "density": 0.40,
                "prompt": "expressive grand piano, melodic, no drums"},
    "guitar":  {"label": "Guitar",     "icon": "🎸", "drums": False, "density": 0.45,
                "prompt": "plucky clean electric guitar, staccato rhythmic riff, picked strings, "
                          "no drums, no violin, no bowed strings, no synth"},
    "lead":    {"label": "Synth lead", "icon": "🎛", "drums": False, "density": 0.50,
                "prompt": "bright catchy synth lead melody, no drums"},
    "pad":     {"label": "Pad",        "icon": "🌫", "drums": False, "density": 0.25,
                "prompt": "lush warm synth pad, slow and sustained, no drums"},
    "strings": {"label": "Strings",    "icon": "🎻", "drums": False, "density": 0.30,
                "prompt": "cinematic string ensemble, sustained, no drums"},
}


def instrument_list() -> list[dict]:
    return [{"id": k, "label": v["label"], "icon": v["icon"]} for k, v in INSTRUMENTS.items()]


def _read_wav(path: str) -> np.ndarray:
    """Load a 16-bit PCM wav as float32 stereo (n, 2)."""
    from scipy.io import wavfile  # noqa: PLC0415
    _, data = wavfile.read(path)
    y = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        y /= 32768.0
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    return y


def read_buffer(path: str) -> np.ndarray:
    """Public: load an aligned loop WAV into a float32 stereo buffer (n, 2) for
    the server-side mixer in the multiplayer Looper Room."""
    return _read_wav(path)


def _detect_tempo(mono: np.ndarray) -> float:
    import librosa  # noqa: PLC0415
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=SR)
        tempo = float(np.atleast_1d(tempo)[0])
    except Exception:  # noqa: BLE001
        tempo = 0.0
    return tempo


def _octave_fold(tempo: float, target: float) -> float:
    """Fold `tempo` by powers of two to the value closest (in log space) to
    `target`. This corrects the common half/double beat-tracking error so we
    stretch by a small ratio instead of 2x."""
    if tempo <= 0 or target <= 0:
        return tempo or target
    best = tempo
    for k in (-2, -1, 0, 1, 2):
        cand = tempo * (2.0 ** k)
        if abs(np.log2(cand / target)) < abs(np.log2(best / target)):
            best = cand
    return best


def _fold_into_range(tempo: float, lo: float, hi: float) -> float:
    """Fold `tempo` by octaves into [lo, hi] (most music sits in ~[80,160])."""
    if tempo <= 0:
        return (lo + hi) / 2
    while tempo < lo:
        tempo *= 2
    while tempo > hi:
        tempo /= 2
    return tempo


def _time_stretch(channel: np.ndarray, rate: float) -> np.ndarray:
    import librosa  # noqa: PLC0415
    ch = np.ascontiguousarray(channel)
    try:
        return librosa.effects.time_stretch(y=ch, rate=rate)   # newer librosa (keyword)
    except TypeError:
        return librosa.effects.time_stretch(ch, rate)          # older librosa (positional)


def _seamless_window(buf: np.ndarray, start: int, lsamp: int, fade: int) -> np.ndarray:
    """Cut one L-length loop window that wraps without a click.

    We take the window plus `fade` samples of lookahead, then crossfade the
    window's head with that lookahead. Because out[0] equals the lookahead (what
    naturally follows the window), the wrap end->start is continuous.
    """
    seg = buf[start: start + lsamp + fade]
    if seg.shape[0] < lsamp + fade:                 # no lookahead: wrap to own head
        base = buf[start: start + lsamp]
        if base.shape[0] < lsamp:
            reps = int(np.ceil(lsamp / max(1, base.shape[0])))
            base = np.tile(base, (reps, 1))[:lsamp]
        seg = np.concatenate([base, base[:fade]], axis=0)
    out = seg[:lsamp].copy()
    if fade > 0:
        j = np.arange(fade)
        win_in = np.sin(0.5 * np.pi * j / fade)[:, None]    # equal-power crossfade
        win_out = np.cos(0.5 * np.pi * j / fade)[:, None]
        out[:fade] = seg[:fade] * win_in + seg[lsamp:lsamp + fade] * win_out
    return out


def prepare_loop(raw_path: str, out_path: str, target_bpm: float, bars: int) -> dict:
    """Tempo-align a raw render into a bar-locked, seamless, multi-window loop buffer.

    Decides the session tempo (octave-folding to keep the stretch SMALL, since big
    stretches sound bad and MRT2 has no tempo input), time-stretches, then cuts
    several bar-aligned windows spread across the material (so New take contrasts),
    each seam-crossfaded. Returns {tempo_detected, bpm, loop_secs, windows, stretch}.

    target_bpm <= 0 means AUTO: the loop's own (octave-folded) tempo becomes the
    session tempo, so the first track is essentially un-stretched.
    """
    from .mrt_worker import MRTWorker  # noqa: PLC0415 - reuse the 16-bit writer

    y = _read_wav(raw_path)
    mono = y.mean(axis=1)
    detected = _detect_tempo(mono)
    det = detected if (detected and 40 <= detected <= 240) else 0.0

    if target_bpm and target_bpm > 0:
        session = float(max(50.0, min(180.0, target_bpm)))
        folded = _octave_fold(det, session) if det else session
        rate = session / folded if folded > 0 else 1.0
    else:                                     # AUTO: trust the loop's folded tempo
        session = round(_fold_into_range(det if det else 110.0, 80.0, 160.0))
        rate = 1.0
    rate = max(0.5, min(2.0, rate))           # safety net against destructive stretch
    loop_secs = bars * BEATS_PER_BAR * 60.0 / session

    if abs(rate - 1.0) > 0.02:
        left = _time_stretch(y[:, 0], rate)
        right = _time_stretch(y[:, 1], rate)
        n = min(len(left), len(right))
        ys = np.stack([left[:n], right[:n]], axis=1)
    else:
        ys = y

    lsamp = int(round(loop_secs * SR))
    fade = min(int(0.030 * SR), lsamp // 8)         # ~30ms seam crossfade
    avail = ys.shape[0] // lsamp
    if avail < 1:                                    # too short: tile up to one window
        reps = int(np.ceil(lsamp / max(1, ys.shape[0])))
        ys = np.tile(ys, (reps, 1))[:lsamp]
        avail = 1
    windows = max(1, min(MAX_WINDOWS, avail))
    # spread the window start points across the material so takes actually differ
    if avail > windows and windows > 1:
        starts = [int(round(i * (avail - 1) / (windows - 1))) * lsamp for i in range(windows)]
    else:
        starts = [i * lsamp for i in range(windows)]

    buf = np.concatenate([_seamless_window(ys, s, lsamp, fade) for s in starts], axis=0)
    MRTWorker._write_wav(out_path, buf.astype(np.float32))
    return {"tempo_detected": round(detected, 1) if detected else 0.0,
            "bpm": round(session, 1), "loop_secs": round(loop_secs, 4),
            "windows": windows, "stretch": round(rate, 3)}


class LooperEngine:
    """Renders instrument loops through the chosen engine (MRT2 or SA3), then aligns
    them. Generation is clip-based, so either model works; the tempo-align + mix layer
    is engine-agnostic."""

    def __init__(self, forge_getter, clips_dir: str, storage):
        self._get_forge = forge_getter          # _forge_for(name) -> ForgeCore
        self._clips_dir = clips_dir
        self._storage = storage
        self._raw_dir = os.path.join(clips_dir, "_looper_raw")
        os.makedirs(self._raw_dir, exist_ok=True)

    def render(self, instrument: str, key: str | None, bpm: float, bars: int,
               prompt_override: str | None = None, engine: str = "mrt2") -> tuple[Clip, dict]:
        inst = INSTRUMENTS.get(instrument)
        if inst is None:
            raise ValueError(f"unknown instrument: {instrument}")
        base_prompt = (prompt_override.strip() if prompt_override and prompt_override.strip()
                       else inst["prompt"])
        bpm = float(bpm)                       # <= 0 means AUTO (prepare_loop decides)
        bars = max(1, min(8, int(bars)))
        # Render well beyond the windows we need so the takes land on DIFFERENT
        # musical material (MRT2 evolves over a long autoregressive render, and it
        # generates faster than real-time, so this stays cheap). prepare_loop then
        # spreads the window start points across all of it.
        nominal_bpm = bpm if bpm > 0 else 110.0
        nominal_loop = bars * BEATS_PER_BAR * 60.0 / nominal_bpm
        raw_secs = nominal_loop * (MAX_WINDOWS * 2 + 1) + 2.0
        chunks = max(2, round(raw_secs / 0.8))

        spec = PromptSpec(
            text_a=base_prompt + ", instrumental, steady tempo",
            key=key or None, density=inst["density"], drums=inst["drums"], chunks=chunks,
        )
        eng = engine if engine in ("mrt2", "sa3") else "mrt2"
        forge = self._get_forge(eng)
        raw_clip = forge.generate(spec, "looper")          # writes the raw render WAV
        raw_path = raw_clip.wav_path

        clip_id = uuid.uuid4().hex[:12]
        out_path = os.path.join(self._clips_dir, f"{clip_id}.wav")
        info = prepare_loop(raw_path, out_path, bpm, bars)
        try:
            if raw_path != out_path:
                os.remove(raw_path)
        except OSError:
            pass

        clip = Clip(id=clip_id, wav_path=out_path, spec=spec, engine=eng,
                    name=f"{inst['label']} loop")
        self._storage.put_clip(clip)
        return clip, info
