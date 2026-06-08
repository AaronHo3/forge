# Score the Story — Project Brief

## Repository layout (two projects)
This repo now holds two separate projects:
- **`fallback/`** — the original Score-the-Story speech→music system this brief
  describes. All its code + outputs live here. Run its commands from inside
  `fallback/` (e.g. `cd fallback && python main.py`). Generated artifacts are
  sorted under `fallback/outputs/{songs,keepsakes,arrangements,telemetry}/`
  (routed via `fallback/paths.py`).
- **`forge/`** — the NEW project: a generative-music game/playground for
  musicians built on MRT2. See [`GAME_PLAN.md`](GAME_PLAN.md) and
  [`forge/README.md`](forge/README.md). Outputs live in `forge/outputs/`.

The rest of this brief describes the **`fallback/`** project.

## What this is
A live AI music scoring system for the Berklee MusicHackathon (Google DeepMind Challenge).
A narrator speaks. Magenta RealTime 2 (MRT2) composes and plays a musical score in real time,
responding to the emotional qualities of their voice — no pre-composed music, no manual controls.

## The pipeline
```
Microphone → VoiceAnalyzer → FeatureMapper → MRTController → MRT2 → Speakers
```

1. **VoiceAnalyzer** (`voice_analyzer.py`) — captures mic at 48kHz, extracts per-block features:
   energy (RMS), pitch (Hz), speech_rate (onset strength), brightness (spectral centroid), is_silent
2. **FeatureMapper** (`feature_mapper.py`) — maps those features to MRT2 params with exponential smoothing:
   prompt_blend (0=dark/tense, 1=warm/bright), chaos (0=sparse, 1=dense), drums_on (bool)
3. **MRTController** (`mrt_controller.py`) — two modes:
   - **Native / Python mode (PRIMARY, the working real-time path)**: runs MRT2 in-process
     via its MLX backend (`MagentaRT2SystemMlxfn`, `mrt2_base`). The voice blend interpolates
     between two style embeddings each ~0.8s chunk; audio streams straight to the speakers.
     Confirmed real-time on Apple Silicon (RTF ~0.76, 0 underruns).
   - MIDI mode (LEGACY): sends CC1/CC11/CC64 to the MRT2 AU plugin in a DAW via a virtual
     MIDI port. Kept as a fallback; not the path the project actually uses now.
4. **main.py** — orchestrates all three with threading, terminal display, keyboard chord input.
   `--mode python` selects the native path above; plain `main.py` still defaults to legacy MIDI.

## Current status
The native MLX pipeline runs end-to-end in real time. A pytest suite now covers the
deterministic core (`cd fallback && python3 -m pytest`).
- `voice_analyzer.py` ✅ mic capture + feature extraction (unit-tested on synthetic buffers)
- `feature_mapper.py` ✅ voice→param mapping (unit-tested)
- `mrt_controller.py` ✅ native MLX path confirmed real-time; MIDI path is legacy
- `llm_style_director.py` ✅ optional semantic rail (Whisper→Claude); decoupled, degrades gracefully
- `speaker_signature.py`, `keepsake.py`, `paths.py` ✅ unit-tested
- Robustness initiative in progress — see the `fallback-robustness` memory for the full roadmap.

Known issue: drums are currently forced OFF in the live engine (`mrt_controller.py` sends
`drums=[0]`), so preset `drums_threshold` values have no live effect yet.

## Next steps (robustness)
1. Phase 2 resilience: surface engine health — the generation thread can currently die
   silently and still look "running"; also wrap `OutputStream.start` in try/except.
2. Phase 3 seams: inject an `MRTBackend` protocol + a fake backend so the audio loop is
   testable; extract `harmony.py`/`dsp.py`/`config.py` to kill live↔keepsake tunable drift.
3. Decide drums: restore `params.drums_on` or remove the dead `drums_threshold` config.
4. Phase 4 observability: logging instead of print; engine health on the telemetry dashboard.

## MRT2 specifics
- **Native (primary): the `magenta-rt` MLX backend.** Weights live under
  `~/Documents/Magenta/magenta-rt-v2/models/mrt2_base/` (the `.mlxfn` export). The engine is
  `magenta_rt.mlx.system.MagentaRT2SystemMlxfn(size='mrt2_base')`; key methods are
  `embed_style(text) -> ndarray` and `generate(style=..., notes=..., frames=..., state=...)`.
  Runs at 25 frames/sec (50 frames = 2s). Real-time on the M4 Pro.
- Sample rate is 48,000 Hz throughout (matches MRT2's requirement).
- **MIDI / AU (legacy):** MRT2 AU plugin at `~/Desktop/MusicHack/MRT2 Bundle/AudioUnit/`.
  MIDI CC mapping: CC1=blend, CC11=chaos, CC64=drums. Only used by `--mode midi`.

## Key design decisions
- 48kHz sample rate throughout (matches MRT2 requirement)
- Smoothing alpha=0.70 in FeatureMapper — change this to make it more/less reactive
- Silence detection threshold=0.005 RMS — adjust if too sensitive in a noisy room
- Prompt A = "dark minor strings, tense, cinematic" / Prompt B = "warm bright piano jazz, uplifting"
  → These are the two musical poles the voice navigates between. Change them freely.
- UPDATE_HZ=20 in main.py — 20 parameter updates per second to MRT2

## Dependencies
```
sounddevice, librosa, numpy, scipy    # core audio pipeline
magenta-rt, mlx                       # native real-time generation (primary)
openai-whisper, anthropic             # optional semantic steering (Whisper→Claude)
python-rtmidi                         # legacy MIDI mode only
```
Install everything with `cd fallback && pip install -r requirements.txt`
(dev/test extras: `pip install -r requirements-dev.txt`).

## Running individual components
```bash
python3 main.py --mode python   # PRIMARY: native real-time pipeline — start talking
python3 spike_mlx_realtime.py   # standalone proof MRT2 streams in real time + timing chart
python3 -m pytest               # the unit-test suite (no hardware needed)
python3 voice_analyzer.py       # live mic feature test
python3 main.py                 # legacy MIDI mode (needs a DAW + IAC bus)
```
