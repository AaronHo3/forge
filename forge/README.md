# Forge

A generative-music playground for musicians, powered by MRT2.
See [`../GAME_PLAN.md`](../GAME_PLAN.md) for the full vision + architecture.

> This is a **new, self-contained project**. The Score-the-Story files in the parent
> directory are an untouched fallback.

## The one primitive

```
prompt → MRT2 → audition → keep/export        ("the Forge")
```

Every game mode (Telephone / Showdown / Battle) is a wrapper on this.

## Layout

```
forge/
├── backend/
│   ├── mrt_worker.py     # owns the single MRT2 model; serial job queue
│   ├── forge_core.py     # prompt→clip primitive (+ variations, deck pre-gen)
│   ├── judge.py          # Claude rubric judge + tutor notes (+ novelty)
│   ├── storage.py        # clips, crates, rooms, leaderboard
│   ├── rooms.py          # per-room game state machine
│   ├── server.py         # FastAPI: REST + WebSocket hub + static frontend
│   └── modes/
│       └── telephone.py  # "Broken Record" round logic (first game mode)
├── frontend/             # HTML + JS + WebAudio (added in Phase 1)
└── requirements.txt
```

## Status

- ✅ **Phase 0 — MRT2 worker:** `prompt → wav` (real `MagentaRT2SystemMlxfn`,
  A↔B blend, key anchor, density→cfg). 48 kHz stereo, verified.
- ✅ **Phase 1 — Forge workbench (web, solo):** browser UI → prompt controls,
  A↔B morph, density sweep, audition, ⭐keep, crate (persists across restart),
  ⬇WAV download. Verified end-to-end over HTTP.
  - Note: this MLX build of MRT2 is **deterministic** (same input → same audio);
    you explore by changing inputs, and "density sweep" renders sparse→dense takes.
- ✅ **Phase 2a — Broken Record (Telephone), hotseat:** pass-the-device game at
  `/telephone`. Compose → hear → guess → drift reveal with per-guess scoring,
  tutor notes, leaderboard, and chain-fidelity. AI judge uses Claude when
  `ANTHROPIC_API_KEY` is set, else a word-overlap heuristic.
- ✅ **Phase 2b — networked rooms (`/play`):** everyone on their own phone, playing
  simultaneously (parallel chains). `net.py` engine + `room.html` UI:
  create/join by code → lobby → seed (full workbench controls / **seed from a crate
  clip** / prompt-hint chips) → guess → live waiting → multi-chain drift reveal +
  leaderboard. Host sets clip length, key, **rounds**, and **answer time** (per-turn
  countdown; server auto-fills idle/disconnected players so a turn never hangs).
  Verified end-to-end (3-client sim; full-controls seed, from_clip, timeout auto-fill).
- ✅ Crate remove + 💡 prompt-hints (`/api/hints`) shared with games.
- ✅ **Phase 3 — party modes:**
  - ✅ **Showdown (`/showdown`)** — Kahoot-style: one composer per round, everyone
    races to guess, **faster = more points**, composer earns avg accuracy, rotates,
    live leaderboard. `showdown.py` + `showdown.html`.
  - ✅ **Forge Battle (`/battle`)** — everyone crafts to the same brief; scored on
    AI **match** + a **novelty bonus** (1−cosine of MusicCoCa style embeddings);
    🚀 Most Original award; **keep any sound** into your crate. `battle.py` +
    `battle.html`. Verified (3-player novelty differentiation; keep-from-reveal).
  - Clips now carry their **style embedding** (`mrt_worker`→`forge_core`), powering
    novelty; `Judge.novelty` implemented.
- ✅ **Phase 4 — beginner tier:**
  - **Difficulty selector** (beginner/advanced) in every game lobby → drives the
    judge rubric (genre/mood/tempo vs. subgenre/production/meter).
  - **Ear Trainer (`/train`)** — solo, two kinds of question, each with the right
    tool: THEORY (intervals / chords / scales) via precise **synth tones**
    (`synth.py`) — random root each time, instant, never repeats, with song
    mnemonics; VIBE (mood / tempo / genre…) via **MRT2** (~6s clips). Mode picker +
    difficulty, score/streak, replay. Synth audio served at `/train/audio/<token>.wav`.

### Shared base (no duplicated hub code)
`basehub.py` holds `BaseGame` + `BaseHub` (create/join/lobby/leave, broadcast,
per-room timers). `net.py`, `showdown.py`, `battle.py` each subclass it and add
only their game logic. Verified: all three modes pass after the refactor.

`rooms.py` is an earlier scaffold; the live networked logic lives in `net.py`.

## Play online with friends

```bash
cd forge && uvicorn backend.server:app --port 8000 --host 0.0.0.0
# everyone on the same Wi-Fi opens  http://<your-LAN-ip>:8000/play
```

## Run the workbench (testable now)

```bash
cd forge
pip install -r requirements.txt          # first time only
uvicorn backend.server:app --port 8000
# open http://localhost:8000  → type a prompt, hit Generate, listen
```

First generation loads the model (one-time, ~slower); each clip then takes a few
seconds. Kept clips persist in memory for the session; WAVs are in
`forge/outputs/clips/`.

### Headless self-test (no browser)

```bash
python3 -m backend.mrt_worker --selftest   # prompt → forge/outputs/selftest.wav
```
