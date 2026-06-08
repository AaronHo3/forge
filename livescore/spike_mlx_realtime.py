"""
spike_mlx_realtime.py
---------------------
A standalone proof that Magenta RealTime 2 generates music in real time,
locally, on this Mac via the MLX backend. Nothing else in livescore/ depends
on this file, and it depends on nothing in livescore/ either.

What it proves and measures:
  1. MRT2 loads and streams continuous audio to the speakers.
  2. The TRUE generation time per chunk on THIS hardware (not the docs).
  3. The real-time factor (gen_time / audio_seconds). Below 1.0 means the
     model produces sound faster than it plays, i.e. real-time capable.
  4. Playback underruns (audible stutters) when generation falls behind.
  5. The structural latency floor: pass --prompt2 to switch style mid-stream
     and hear how long MRT2 takes to actually turn (it conditions on a
     rolling context, so the change lands a couple of seconds later).

All timing data is written to a CSV and, if matplotlib is present, a PNG
chart, so you can analyze how the model behaves rather than guess.

Usage:
    python3 spike_mlx_realtime.py
    python3 spike_mlx_realtime.py --size mrt2_base --seconds 30
    python3 spike_mlx_realtime.py --prompt "warm lo-fi piano" \
        --prompt2 "epic cinematic strings" --switch-at 8

First run downloads the model weights from Hugging Face (small = a few
hundred MB, base = a few GB), so it will pause once before the first sound.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import sounddevice as sd


# ── Where telemetry lands ─────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "outputs" / "spikes"


# ── One row of per-chunk timing data ──────────────────────────────────────────
@dataclass(frozen=True)
class ChunkStat:
    index: int
    prompt: str
    gen_ms: float          # wall time spent generating this chunk
    audio_s: float         # seconds of audio the chunk contains
    rtf: float             # gen_time / audio_seconds  (< 1.0 == real-time)
    buffered_s: float      # seconds of audio queued ahead of the speaker
    underruns: int         # cumulative playback starves so far


# ── Gapless playback: a tiny ring buffer drained by the audio callback ────────
class StreamPlayer:
    """Holds generated samples and feeds them to the output device.

    The producer (generation loop) calls push(); the audio driver calls
    _callback() on its own thread to pull. If the buffer ever runs dry mid
    stream, that is an underrun (an audible gap) and we count it.
    """

    def __init__(self, sample_rate: int, channels: int):
        self._sample_rate = sample_rate
        self._channels = channels
        self._pending = np.zeros((0, channels), dtype=np.float32)
        self._lock = threading.Lock()
        self._started = False        # don't count underruns before first push
        self.underruns = 0
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._pending = np.concatenate([self._pending, samples], axis=0)
            self._started = True

    def buffered_seconds(self) -> float:
        with self._lock:
            return len(self._pending) / self._sample_rate

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            n = min(len(self._pending), frames)
            outdata[:n] = self._pending[:n]
            self._pending = self._pending[n:]
            started = self._started
        if n < frames:
            outdata[n:] = 0.0
            if started:               # ran dry after real audio began = stutter
                self.underruns += 1


# ── Audio shape helper ────────────────────────────────────────────────────────
def to_2d_float32(samples: np.ndarray) -> np.ndarray:
    """Coerce MRT2 output into (frames, channels) float32 for sounddevice."""
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    # If it came channels-first like (2, N), flip to (N, 2).
    if a.ndim == 2 and a.shape[0] <= 2 and a.shape[0] < a.shape[1]:
        a = a.T
    return np.ascontiguousarray(a)


# ── Visualization + persistence ───────────────────────────────────────────────
def sparkline(values: list[float]) -> str:
    """A one-line ASCII chart so you see the shape without leaving the terminal."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / span * 7))] for v in values)


def write_csv(stats: list[ChunkStat], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()
        for s in stats:
            writer.writerow(asdict(s))


def write_chart(stats: list[ChunkStat], path: Path) -> bool:
    """Save a PNG of gen-time and real-time factor per chunk. Best effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    idx = [s.index for s in stats]
    gen = [s.gen_ms for s in stats]
    rtf = [s.rtf for s in stats]
    audio_ms = [s.audio_s * 1000.0 for s in stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(idx, gen, marker="o", label="generation time (ms)")
    ax1.plot(idx, audio_ms, linestyle="--", label="real-time budget (ms)")
    ax1.set_ylabel("milliseconds")
    ax1.set_title("MRT2 per-chunk generation time vs real-time budget")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.axhline(1.0, color="red", linestyle="--", label="real-time threshold")
    ax2.plot(idx, rtf, marker="o", color="green", label="real-time factor")
    ax2.set_xlabel("chunk index"); ax2.set_ylabel("RTF (gen / audio)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def print_summary(stats: list[ChunkStat], load_s: float, player: StreamPlayer) -> None:
    gen = [s.gen_ms for s in stats]
    rtf = [s.rtf for s in stats]
    audio_total = sum(s.audio_s for s in stats)
    realtime_ok = statistics.median(rtf) < 1.0

    print("\n" + "═" * 64)
    print("  MRT2 REAL-TIME SPIKE — RESULTS")
    print("═" * 64)
    print(f"  model load time      {load_s:6.1f} s")
    print(f"  chunks generated     {len(stats)}")
    print(f"  audio produced       {audio_total:6.1f} s")
    print(f"  gen time  median     {statistics.median(gen):6.0f} ms")
    print(f"  gen time  p95        {sorted(gen)[int(len(gen) * 0.95) - 1]:6.0f} ms")
    print(f"  gen time  max        {max(gen):6.0f} ms  (first chunk = warmup)")
    print(f"  real-time factor     {statistics.median(rtf):6.2f}  (median)")
    print(f"  playback underruns   {player.underruns}")
    print(f"  verdict              "
          + ("REAL-TIME CAPABLE ✓" if realtime_ok else "TOO SLOW FOR REAL-TIME ✗"))
    print("\n  gen time per chunk   " + sparkline(gen))
    print("  real-time factor     " + sparkline(rtf))
    print("═" * 64)


# ── The spike itself ──────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    from magenta_rt import MagentaRT2Mlxfn

    print(f"[spike] loading {args.size} via MLX…")
    t0 = time.monotonic()
    system = MagentaRT2Mlxfn(size=args.size, temperature=args.temperature)
    load_s = time.monotonic() - t0
    print(f"[spike] loaded in {load_s:.1f}s")

    # use_mapper=True matches the model's reference invocation for text prompts.
    style = system.embed_style(args.prompt, use_mapper=True)
    style2 = (system.embed_style(args.prompt2, use_mapper=True)
              if args.prompt2 else None)
    frames = args.chunk_frames

    # Generate one chunk up front to learn the real sample rate / channel count
    # and to absorb the first-call warmup before the speaker stream opens.
    print("[spike] warming up (first chunk compiles the graph)…")
    warm_t = time.monotonic()
    wav, state = system.generate(style=style, frames=frames)
    warm_ms = (time.monotonic() - warm_t) * 1000.0
    first = to_2d_float32(wav.samples)
    sample_rate, channels = wav.sample_rate, first.shape[1]
    chunk_s = wav.seconds
    print(f"[spike] chunk = {chunk_s:.2f}s audio @ {sample_rate}Hz, "
          f"{channels}ch · warmup {warm_ms:.0f}ms")

    player = StreamPlayer(sample_rate, channels)
    player.push(first)
    player.start()

    stats: list[ChunkStat] = []
    active_prompt = args.prompt
    target_chunks = max(1, round(args.seconds / chunk_s))
    print(f"[spike] streaming ~{args.seconds}s "
          f"({target_chunks} chunks). Ctrl+C to stop early.\n")

    try:
        for i in range(target_chunks):
            # Mid-stream style switch: demonstrates MRT2's latency floor live.
            if style2 is not None and i == args.switch_at:
                style = style2
                active_prompt = args.prompt2
                print(f"  ── prompt → \"{args.prompt2}\" "
                      f"(listen for how long it takes to turn) ──")

            t = time.monotonic()
            wav, state = system.generate(style=style, frames=frames, state=state)
            gen_ms = (time.monotonic() - t) * 1000.0

            player.push(to_2d_float32(wav.samples))
            buffered = player.buffered_seconds()
            stat = ChunkStat(
                index=i, prompt=active_prompt, gen_ms=gen_ms,
                audio_s=wav.seconds, rtf=gen_ms / 1000.0 / wav.seconds,
                buffered_s=buffered, underruns=player.underruns,
            )
            stats.append(stat)
            flag = "ok " if stat.rtf < 1.0 else "SLOW"
            print(f"  chunk {i:02d}  gen {gen_ms:5.0f}ms  "
                  f"rtf {stat.rtf:4.2f} {flag}  buffered {buffered:4.1f}s  "
                  f"underruns {player.underruns}")

            # Stay roughly one chunk ahead: don't race far past playback, so the
            # measured latency reflects a real streaming scenario (and memory
            # stays bounded). Only sleeps when generation is faster than audio.
            while player.buffered_seconds() > chunk_s * 1.5:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[spike] stopped early.")

    # Let the buffered tail finish playing before tearing down the stream.
    while player.buffered_seconds() > 0.05:
        time.sleep(0.05)
    player.stop()

    if not stats:
        print("[spike] no chunks generated.")
        return

    print_summary(stats, load_s, player)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = OUT_DIR / f"spike-{stamp}.csv"
    png_path = OUT_DIR / f"spike-{stamp}.png"
    write_csv(stats, csv_path)
    print(f"\n[spike] telemetry → {csv_path}")
    if write_chart(stats, png_path):
        print(f"[spike] chart     → {png_path}")
    else:
        print("[spike] (install matplotlib for the PNG chart)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRT2 real-time generation spike")
    p.add_argument("--size", choices=["mrt2_small", "mrt2_base"],
                   default="mrt2_base",
                   help="base = bigger/better, real-time on a Pro/Max chip "
                        "(weights already present); small needs downloading")
    p.add_argument("--chunk-frames", type=int, default=50,
                   help="frames per generated chunk (25 fps; 50 = 2s, "
                        "MRT2's native chunk size)")
    p.add_argument("--prompt", default="warm lo-fi hip hop, mellow rhodes piano",
                   help="style text prompt to start with")
    p.add_argument("--prompt2", default=None,
                   help="optional second prompt to switch to mid-stream")
    p.add_argument("--switch-at", type=int, default=6,
                   help="chunk index at which to switch to --prompt2")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="how many seconds of audio to stream")
    p.add_argument("--temperature", type=float, default=1.3,
                   help="sampling temperature (MRT2 default 1.3)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
