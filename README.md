# Forge

**Play AI music like an instrument, learn it like a game, finish it like a producer.**

Forge is an AI music platform built on Google DeepMind's Magenta RealTime and
Stability's Stable Audio 3. Anyone, with or without musical training, can create
starter clips alone or with friends: play a chord on your keyboard and a live AI
band answers it, race to write prompts in party games that coach you on how to
describe sound, or stack instrument loops into a track together. Everything you
make exports straight to Audiotool as separate tracks, so a quick idea becomes a
real multitrack project you finish in a DAW.

It runs locally on a Mac, uses each model where it is strongest (Magenta RealTime
for live, interactive, note-aware playing; Stable Audio 3 for high-fidelity clips),
and is built for two audiences with one surface: beginners learning how music and
AI models work, and experienced musicians sparking ideas and collaborating.

---

## What you can do

### Play
- **Chord Coach** — your computer keyboard becomes a piano. Build the chord on
  screen, hear it, get graded with plain-language feedback, then have Magenta
  RealTime play your chord back with a gentle band. No musical background needed.
- **Looper** — pick instruments, the AI renders each as a tempo-aligned loop, and
  they stack into a track you mix in real time. Solo or with friends in a room.

### Learn
- **Prompt Party** — everyone gets the same creative brief, writes a prompt to hit
  it, the AI composes each one, then the room votes and an AI coach explains how to
  prompt better. Teaches the core skill of the AI-music era: describing sound.
- **Prompt Detective** — the reverse game. One player composes a secret track, the
  rest guess the prompt, scored on accuracy and speed with a live audio visualizer
  and rhythm-game ratings.
- **Daily Challenge** — one shared brief per day, scored, with a leaderboard. The
  habit loop.

### Make and finish
- **Workbench** — drive the models with prompts, audition takes, restyle any clip
  (audio-to-audio via Stable Audio 3), keep what you love.
- **Audiotool export** — send clips and Looper stems to Audiotool, each on its own
  track, then open the project in a real DAW to keep building.

---

## Run it locally

Requirements: a modern Mac, Python 3.10+, and the Magenta RealTime library. Audio
deps (numpy, scipy, librosa), the web server (fastapi, uvicorn), and the optional
AI judge (anthropic) are listed in `forge/requirements.txt`.

```bash
cd forge
python3 -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000**. For the multiplayer games and rooms, open the
same address on a second device or tab on the same WiFi.

Optional setup (copy `forge/.env.example` to `forge/.env`):
- `AUDIOTOOL_PAT` — a Personal Access Token from https://rpc.audiotool.com/dev,
  needed for the Audiotool export button.
- `ANTHROPIC_API_KEY` — enables the AI coach in the games (falls back to a
  word-overlap heuristic if unset).
- `FORGE_SA3_DIR` — path to a Stable Audio 3 checkout, only if you want the SA3
  engine.

The first generation loads the model once and is slow; every clip after is quick.

---

## How it works

- **Pluggable engines.** Generation flows through one engine-agnostic core, so any
  mode can use Magenta RealTime or Stable Audio 3. A host picks the model where it
  makes sense; the real-time, note-conditioned modes (Chord Coach, Harmony, Journey)
  use Magenta RealTime because only it can stream and follow notes live.
- **One model, served safely.** The MLX model is loaded once and every request is
  funneled through a single worker, so it is never double-loaded. Live rooms read a
  shared state the worker streams from.
- **Real-time rooms.** Multiplayer modes broadcast one shared audio stream to every
  listener over WebSockets, with per-client backpressure so latency stays low.
- **Tempo-aligned loops.** The Looper renders each instrument, beat-tracks it,
  time-stretches it to the session tempo, and crops it to a bar grid so independent
  renders lock together.
- **On-device.** It runs locally on a Mac with no internet required for generation.

---

## Repo layout

- `forge/` — the platform described above (`backend/` FastAPI + engines,
  `frontend/` the pages, `audiotool_sidecar/` a Node bridge to Audiotool).

> The Score-the-Story speech-to-music system this project grew out of now lives
> in its own repo: [score-the-story](https://github.com/AaronHo3/score-the-story).

---

Built in 36 hours for the Berklee MusicHackathon (Google DeepMind challenge), on
Magenta RealTime 2, Stable Audio 3, and Audiotool.
