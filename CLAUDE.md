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
   - MIDI mode (default): sends CC1/CC11/CC64 to MRT2 AU via virtual MIDI port (IAC Driver)
   - Python mode: uses `magenta-rt` library directly (needs `pip install magenta-rt`)
4. **main.py** — orchestrates all three with threading, terminal display, keyboard chord input

## Current status
- `voice_analyzer.py` ✅ working — mic capture and feature extraction confirmed
- `feature_mapper.py` ✅ written — not yet tested end-to-end
- `mrt_controller.py` ✅ written — MIDI and Python library modes
- `main.py` ✅ written — needs MIDI setup or Python library to run fully

## Next steps
1. Test feature mapper: `python feature_mapper.py` (no MRT2 needed — uses simulated input)
2. Set up MIDI bridge: Audio MIDI Setup → MIDI Studio → create IAC Driver bus named "MRT2 Control"
3. Load MRT2 AU in a DAW (GarageBand/Logic/Ableton), set MIDI input to "MRT2 Control", sample rate 48kHz
4. Run: `python main.py` and start talking

## MRT2 specifics
- MRT2 AU plugin is at: `~/Desktop/MusicHack/MRT2 Bundle/AudioUnit/MRT2 (AU).app`
- Must install to /Applications before first use (see INSTALL.md in that folder)
- Sample rate MUST be 48,000 Hz — other rates cause pitch distortion
- AU exposes: `prompts`, `promptSurfaceState`, `_midiNotes`, `parameterTree`
- MIDI CC mapping in use: CC1=blend, CC11=chaos, CC64=drums
- Python library: `pip install magenta-rt` — class may be `system.MagentaRT` or `system.MagentaRT2`
  → verify with Magenta team at hackathon

## Key design decisions
- 48kHz sample rate throughout (matches MRT2 requirement)
- Smoothing alpha=0.70 in FeatureMapper — change this to make it more/less reactive
- Silence detection threshold=0.005 RMS — adjust if too sensitive in a noisy room
- Prompt A = "dark minor strings, tense, cinematic" / Prompt B = "warm bright piano jazz, uplifting"
  → These are the two musical poles the voice navigates between. Change them freely.
- UPDATE_HZ=20 in main.py — 20 parameter updates per second to MRT2

## Dependencies
```
sounddevice, librosa, numpy, scipy   # audio pipeline (all working)
python-rtmidi                         # MIDI mode
magenta-rt                            # Python library mode
```

## Running individual components
```bash
python voice_analyzer.py   # live mic test — confirmed working
python feature_mapper.py   # simulated arc test — no hardware needed
python main.py             # full pipeline (needs MIDI or magenta-rt)
python main.py --mode python  # Python library mode
```
