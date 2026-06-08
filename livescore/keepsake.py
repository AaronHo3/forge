"""
keepsake.py
-----------
Turns a told story into a unique, crafted SONG — rendered OFFLINE by MRT2 as ONE
continuous, coherent stream (the clean "studio take" of the live performance).

Two halves:
  SessionLog          — during the live telling, captures the scene/key timeline
                        and the speaker's signature → keepsake-<stamp>.json.
  render_keepsake(...) — offline: replays the captured scenes through a SINGLE
                        state-threaded MRT2 stream — one pulse, one key, the
                        scene styles morphing smoothly from one to the next.

Why one stream (not layered stems): independently-generated stems share only a
key — never a downbeat, tempo phase, or groove — so summing them is incoherent
mush. Threading MRT2's state through the whole song is exactly what makes the
LIVE engine sound good; the keepsake now does the same, offline and cleaner.

Usage:
  # (the live app writes the json automatically on quit)
  python keepsake.py render keepsake-20260606-0142.json
  python keepsake.py render keepsake-...json --dry-run   # test mix w/o MRT2
"""

import json
import sys
import threading
import time
from datetime import datetime

import numpy as np

import config
import dsp
import harmony
import paths

SAMPLE_RATE = config.SAMPLE_RATE
FRAMES_PER_SEC = config.FRAMES_PER_SEC   # 25 MRT2 frames = 1.0s of audio

# Harmony note-mask logic lives in harmony.py, shared with the live controller.


# ══════════════════════════════════════════════════════════════════════════════
#  SessionLog — capture the telling so it can be re-rendered into a keepsake
# ══════════════════════════════════════════════════════════════════════════════

class SessionLog:
    """Records the scene timeline + speaker signature during a live telling."""

    def __init__(self):
        self._t0 = time.monotonic()
        self._scenes = []          # [{t, a, b, key}]
        self._transcript = []      # the story said, for the memory artifact
        self._signature = None
        self._lock = threading.Lock()

    def set_signature(self, sig: dict):
        with self._lock:
            self._signature = sig

    def add_scene(self, a: str, b: str, key: str):
        with self._lock:
            self._scenes.append({
                "t": round(time.monotonic() - self._t0, 2),
                "a": a, "b": b, "key": key,
            })

    def add_transcript(self, text: str):
        with self._lock:
            self._transcript.append(text)

    def scenes(self) -> list[dict]:
        """Thread-safe copy of the captured scene timeline (for live UIs)."""
        with self._lock:
            return [dict(s) for s in self._scenes]

    def elapsed(self) -> float:
        """Seconds since this session started (for the end-of-session summary)."""
        return time.monotonic() - self._t0

    def save(self) -> str | None:
        with self._lock:
            if not self._scenes:
                return None
            path = paths.keepsake_path(
                f"keepsake-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
            data = {
                "created": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.monotonic() - self._t0, 1),
                "signature": self._signature,
                "scenes": self._scenes,
                "transcript": self._transcript,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


# ══════════════════════════════════════════════════════════════════════════════
#  Offline renderer — ONE continuous, state-threaded MRT2 stream
# ══════════════════════════════════════════════════════════════════════════════

SCENE_MIN_SEC = 6.0       # clamp captured scene durations into a musical range
SCENE_MAX_SEC = 16.0

CHUNK_FRAMES   = 25       # 1.0s chunks — offline, favour coherent settling
STYLE_MORPH    = 0.18     # per-chunk glide toward the current scene's style
PALETTE_ANCHOR = 0.20     # how much the speaker's palette tints the WHOLE song
CFG            = config.STYLE_CFG    # style guidance (shared with the live engine)
TEMPERATURE    = config.TEMPERATURE
TOP_K          = 40
DECAY_RMS      = config.DECAY_RMS     # re-seed if the stream fades below this…
DECAY_CHUNKS   = config.DECAY_CHUNKS  # …for this many chunks (stops fade-to-silence)


def _tag(s: str) -> str:
    """A scene pole tag, with a gentle warm fallback if it's missing."""
    return (s or "").strip() or "warm gentle piano"


def _scene_schedule(scenes: list[dict]) -> tuple[list[int], list[float]]:
    """One scene index per 1-second chunk, from the captured timeline (clamped).
    The gap to the next scene's timestamp = that scene's on-screen duration."""
    durations = []
    for i, sc in enumerate(scenes):
        nxt = scenes[i + 1]["t"] if i + 1 < len(scenes) else sc["t"] + SCENE_MAX_SEC
        durations.append(float(np.clip(nxt - sc["t"], SCENE_MIN_SEC, SCENE_MAX_SEC)))
    schedule: list[int] = []
    for si, d in enumerate(durations):
        schedule += [si] * max(1, int(round(d)))
    return schedule, durations


def render_keepsake(session_path: str, out_path: str | None = None,
                    dry_run: bool = False) -> str:
    """Render a session JSON into the keepsake song (.wav) as ONE coherent,
    continuous MRT2 stream — the clean studio take of the live performance.

    The scene timeline drives a single state-threaded generation: the style
    embedding morphs smoothly from scene to scene, the key is locked for the
    whole song, drums hold a steady pulse, and the speaker's palette tints
    everything so the piece has one consistent identity."""
    with open(session_path) as f:
        session = json.load(f)
    scenes = session.get("scenes", [])
    if not scenes:
        raise SystemExit("No scenes in session.")
    sig = session.get("signature") or {}
    palette = sig.get("palette", "warm woody tones: soft piano, acoustic guitar")
    # The home key anchors harmony for the WHOLE song (the speaker's key if we
    # have it, else the first scene's) — so the harmony never yanks mid-piece.
    home_key = sig.get("key", "") or scenes[0].get("key", "")
    out_path = out_path or paths.song_for(session_path)

    schedule, _durations = _scene_schedule(scenes)
    total_chunks = len(schedule)

    print(f"Rendering keepsake from {len(scenes)} scenes → ONE continuous stream")
    print(f"  signature: {sig.get('key','?')} · {palette[:50]}")
    print(f"  length: ~{total_chunks}s · locked key: {home_key or '(free)'}")

    if dry_run:
        song = _dry_stream(total_chunks)
        _write_wav(out_path, song)
        print(f"\n✓ (dry-run) keepsake written: {out_path}  "
              f"({song.shape[0]/SAMPLE_RATE:.0f}s)")
        return out_path

    from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
    print("Loading mrt2_base (offline, high quality)...")
    mrt = MagentaRT2SystemMlxfn(size='mrt2_base')
    num_notes = mrt._num_notes

    notes = harmony.build_notes(home_key, num_notes)

    # The speaker's palette → one consistent timbral identity across the song.
    palette_emb = mrt.embed_style(f"{palette}, warm, gentle, instrumental")

    # Per-scene TARGET styles: blend the scene's two poles, then tint with the
    # palette. scene_style chases these targets a little each chunk (no cuts).
    targets = []
    for sc in scenes:
        a = mrt.embed_style(_tag(sc.get("a", "")))
        b = mrt.embed_style(_tag(sc.get("b", "")))
        scene_emb = 0.5 * a + 0.5 * b
        targets.append((1.0 - PALETTE_ANCHOR) * scene_emb + PALETTE_ANCHOR * palette_emb)

    scene_style = targets[schedule[0]]
    state, low_streak, cur_scene = None, 0, -1
    chunks = []
    for ci in range(total_chunks):
        si = schedule[ci]
        if si != cur_scene:
            cur_scene = si
            sc = scenes[si]
            print(f"\nScene {si+1}/{len(scenes)}  (~{schedule.count(si)}s)  "
                  f"{_tag(sc.get('a',''))} ↔ {_tag(sc.get('b',''))}")

        # Glide toward this scene's style (smooth morph, never a cut).
        scene_style = scene_style + STYLE_MORPH * (targets[si] - scene_style)

        # Decay watchdog: if the stream has faded, generate fresh (state=None) so
        # it re-energises instead of dying out over a long song.
        reseed = low_streak >= DECAY_CHUNKS
        wav, state = mrt.generate(style=scene_style, notes=notes, drums=[1],
                                  cfg_musiccoca=CFG, frames=CHUNK_FRAMES,
                                  state=None if reseed else state,
                                  temperature=TEMPERATURE, top_k=TOP_K)
        samp = np.ascontiguousarray(wav.samples)
        raw = float(np.sqrt(np.mean(samp ** 2)))
        low_streak = 0 if (reseed or raw >= DECAY_RMS) else low_streak + 1
        if reseed:
            print("   …re-seeded (stream had faded)")
        chunks.append(samp)

    song = dsp.soft_limit(dsp.normalize(np.concatenate(chunks), target_rms=0.16))
    _write_wav(out_path, song)
    print(f"\n✓ Keepsake song written: {out_path}  ({song.shape[0]/SAMPLE_RATE:.0f}s)")
    return out_path


def _dry_stream(secs: int) -> np.ndarray:
    """A simple synthetic continuous stream so the mix/wav path is testable
    without loading MRT2 (used by --dry-run)."""
    t = np.arange(max(1, secs) * SAMPLE_RATE) / SAMPLE_RATE
    tone = 0.2 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    pulse = (np.mod(t, 0.5) < 0.04).astype(np.float32) * 0.2
    s = (tone + pulse).astype(np.float32)
    return dsp.normalize(np.stack([s, s], axis=1), target_rms=0.16)


def _write_wav(path, audio):
    from scipy.io import wavfile
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, SAMPLE_RATE, (clipped * 32767).astype(np.int16))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "render":
        dry = "--dry-run" in sys.argv
        render_keepsake(sys.argv[2], dry_run=dry)
    else:
        print("Usage: python keepsake.py render <session.json> [--dry-run]")
