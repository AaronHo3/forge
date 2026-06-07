# FORGE — a generative-music playground for musicians

> **Working title.** The platform is "Forge"; the Telephone game mode is "Broken Record."
> Rename freely.

> **This is a brand-new project.** The existing Score-the-Story files (`main.py`, `engine.py`,
> `voice_analyzer.py`, etc.) stay in place untouched as a fallback. The new game lives entirely
> under [`forge/`](forge/).

---

## 1. The one-line vision

**A web playground where musicians drive MRT2 to discover and harvest original sounds —
then take that material into their own productions. Game modes (Telephone, Battles) are
social wrappers that make exploring it fun and competitive.**

## 2. Why a musician would use this (the value prop)

> **Suno is a song *vending machine*** — it hands you a finished track to consume.
> **Forge is a *sound source*** — it hands a pro raw, steerable material to build *with*.

MRT2 is the wrong tool for "make me a finished song" and the right tool for "give me a unique
8-bar bed / texture / motif I can chop into my track." That distinction — stem-level, steerable,
continuous output you *drive and harvest* — is the entire pitch to seasoned musicians, and it's
where MRT2 beats the big consumer models.

## 3. The core idea: one primitive, many wrappers

The whole product is built on a single primitive:

```
        THE FORGE
   prompt → MRT2 → audition → keep/export
```

**Every mode is a wrapper on the Forge:**

| Mode | = Forge + … |
|------|-------------|
| Solo workbench | (just the Forge) |
| Broken Record (Telephone) | + pass-the-clip + guess-the-prompt + drift reveal |
| Showdown | + timer + everyone races to guess one composer's clip |
| Forge Battle | + a constraint brief + AI/peer judging + novelty award |
| Beginner campaign | + training rails (multiple choice, hints, vocab palette) |

Build the Forge **once**, and every game mode becomes *round logic on top of it* — not new
infrastructure. This is why the Forge is built first even though Telephone is the first *game*.

## 4. Two audiences, one difficulty knob

Declaring "I'm advanced" vs "I'm a beginner" routes you to a lobby and flips **one parameter**
threaded through the shared systems — it is *not* two separate games.

| System | Beginner | Advanced (built first) |
|--------|----------|------------------------|
| Answer input | multiple-choice + vocab palette | raw free-text, no safety net |
| AI-judge rubric | genre / mood / tempo | subgenre, era, articulation, *production*, meter, mode |
| Prompt richness (the truth) | "happy piano" | "lo-fi dorian Rhodes, tape wow, swung 5/4" |
| MRT2 controls exposed | blend + tempo | + key/chord, density/cfg, drums, A↔B morph |
| Novelty pressure | none | "most original" bonus, tighter briefs, time pressure |

**Advanced is built first** (every hackathon attendee is an advanced musician), and the hard
version *contains* the easy one — beginner mode is later just "turn the knob down + add rails."

## 5. What advanced musicians actually want (and how we deliver it)

Pros don't want to *learn* music. They want: **a challenge, the thrill of a sound they'd never
have made, and to keep that sound.** So the advanced tier is a *gamified sound-discovery engine* —
competition is the mechanism that pushes them to explore prompt space they'd never wander into
alone, and **capture** is the feature that means nobody leaves empty-handed.

- **⭐ Keep button on every clip** — win or lose. Builds your session "crate."
- **Variations** — "generate 4 takes of this" for divergent exploration.
- **🏆 Trailblazer award** — novelty scored as embedding-distance from clichés / other players.
- **Export** — WAV stems (drag into any DAW) + Audiotool project (reuses `audiotool_arranger.py`).

### The harvest → finish pipeline (the "aid their own music" goal)

```
Explore (prompt/morph MRT2) → Harvest (keep loops/stems)
   → Arrange (string sections, reuse audiotool_arranger) → Export (WAV / Audiotool)
      → Finish in their own DAW
```

Forge is the **idea generator and sound source** that feeds a real workflow — not where the
final song is made.

## 6. The learning spine (beginner tier, secondary)

For non-musicians, the principle is **"name what you already feel."** A beginner already hears
happy/sad and fast/slow — we give names to felt things, attaching vocabulary to a sound they just
heard. Every beginner round drills one rung:

| # | They hear | They learn | Concept |
|---|-----------|-----------|---------|
| 1 | soft/loud | dynamics | expression |
| 2 | fast/slow | tempo / BPM | rhythm |
| 3 | happy/sad | **major/minor** | key/mode ⭐ |
| 4 | empty/busy | texture/density | arrangement |
| 5 | "what's that?" | timbre | instruments |
| 6 | smooth/bouncy | groove | feel |
| 7 | "it's techno!" | genre = recipe of 1–6 | style |

**The AI judge is a tutor, not a scorekeeper** — a wrong guess returns an *explanation*
("you said happy, but this was a minor key — that's the tension you heard"). The honest boundary:
MRT2 teaches the *perceptual* layer (mood, tempo, timbre, genre); note/chord theory is what the
**Audiotool export** is for — that's where a learner graduates to symbolic editing.

---

## 7. Rough technical architecture

### 7.1 High-level picture

```
  ┌─────────────────────────────────────────────────────────────┐
  │  BROWSER (host screen + phone controllers)                   │
  │   Forge UI · game UIs · <audio> playback · WebAudio          │
  └───────────────┬─────────────────────────┬───────────────────┘
        REST (solo actions)        WebSocket (party realtime)
                  │                          │
  ┌───────────────▼──────────────────────────▼──────────────────┐
  │  WEB SERVER  (FastAPI + uvicorn)                             │
  │   ├── REST routes: generate, keep, export, crate, leaderboard│
  │   └── WS hub: rooms, turns, reveal sync                      │
  ├──────────────────────────────────────────────────────────────┤
  │  ROOM MANAGER         FORGE CORE          AI JUDGE           │
  │  per-room state    prompt→clip job     Claude rubric+tutor   │
  │  + mode logic      spec builder        (+ embedding sim)     │
  ├──────────────────────────────────────────────────────────────┤
  │  MRT2 WORKER  (single thread/process owns the model)        │
  │   job queue → embed_style() → generate() loop → WAV          │
  ├──────────────────────────────────────────────────────────────┤
  │  STORAGE   clips (WAV+meta) · crates · rooms · leaderboard   │
  │            (SQLite or JSON files — hackathon-simple)         │
  └──────────────────────────────────────────────────────────────┘
```

### 7.2 The one constraint that shapes everything: MRT2 is a single, serial resource

MRT2 (`MagentaRT2SystemMlxfn`) is a heavy MLX model. Your own `engine.py` documents that MLX's
per-thread GPU stream state means **you cannot safely load it twice in one process.** Therefore:

- **One MRT2 worker owns the one model**, fed by a **job queue**. Every generation — solo Forge,
  every telephone hop, every battle submission — funnels through it and is processed **serially**.
- This is *fine* and even clarifying, because it forces the right design:
  - **Pre-generate** beginner decks offline so solo play has **zero latency**.
  - **Turn-based pacing** hides gen time behind human thinking time (people type slower than MRT2
    generates).
  - **Parallel telephone chains** parallelize the *humans* (everyone's always busy), while the
    *generations* still queue — fine for hackathon room sizes.
- **Scaling later** = run N worker processes (more GPUs/machines) behind the same queue. Not MVP.

> Web mode uses MRT2's **Python library mode only** (`magenta_rt.mlx.system`). MIDI mode needs a
> local DAW and can't be served to remote players, so it's not used here.

### 7.3 Components

- **MRT2 Worker** (`forge/backend/mrt_worker.py`)
  Loads the model once. Consumes `GenJob`s from a queue. A job = a `PromptSpec`
  (text A, optional text B + blend, key, density/cfg, drums, length-in-chunks). Produces a WAV
  file + metadata by looping `generate()` and threading `state` chunk-to-chunk
  (the proven pattern from `mrt_controller.py:_gen_chunk`).

- **Forge Core** (`forge/backend/forge_core.py`)
  The prompt→clip primitive. Builds a `PromptSpec`, submits a `GenJob`, returns a `Clip`
  (id, wav path, prompt, params, novelty embedding). Handles "variations" (N jobs from one spec)
  and deck pre-generation.

- **AI Judge** (`forge/backend/judge.py`)
  Wraps Claude (reuses the Haiku + cached-system-prompt + JSON-extraction pattern from
  `llm_style_director.py`). `score(guess, truth, difficulty)` → `{score, breakdown, tutor_note}`.
  Optionally blends in embedding cosine similarity for a stable backbone + novelty scoring.

- **Room Manager** (`forge/backend/rooms.py`) + **mode logic** (`forge/backend/modes/*.py`)
  One state machine per room: players, mode, round/turn order, scores, leaderboard, phase
  (lobby → round → reveal). Telephone/Showdown/Battle are separate small modules implementing
  a common `GameMode` interface.

- **Web Server** (`forge/backend/server.py`)
  FastAPI: REST for solo Forge actions + static frontend; a WebSocket hub for party realtime
  (join, submit, reveal broadcasts).

- **Storage** (`forge/backend/storage.py`)
  Clips as WAV-on-disk + a metadata record; crates per user; rooms; leaderboard. SQLite or plain
  JSON — keep it dumb for the hackathon.

### 7.4 Data model (sketch)

```
PromptSpec   { text_a, text_b?, blend, key?, density, drums, chunks }
Clip         { id, wav_path, spec, created_by, embedding, novelty_score? }
Crate        { user_id, [clip_id...] }
Player       { id, name, score, skill: beginner|advanced }
Round        { mode, truth(PromptSpec), submissions[{player, guess, clip_id, score}], phase }
Room         { code, mode, players[], rounds[], leaderboard }
```

### 7.5 Reuse map — you're ~40% built already

| Need | Reuse from fallback |
|------|---------------------|
| MRT2 generate loop / state threading | `mrt_controller.py` (`_gen_chunk`, embed/generate) |
| Claude integration (judge) | `llm_style_director.py` (Haiku, prompt caching, `_extract_json`) |
| WAV render / session capture | `keepsake.py` |
| Audiotool export | `audiotool_arranger.py` |
| Web-serving + threaded server pattern | `app.py`, `telemetry.py` |
| Config bundles pattern | `presets.py` |

### 7.6 Tech stack (recommended)

- **Backend:** Python + **FastAPI + uvicorn** — the one justified new dependency, because party
  modes need WebSockets (stdlib `http.server` can't do them cleanly). The **Forge MVP is REST-only**,
  so WS work is deferred to the Telephone phase.
- **Frontend:** plain **HTML + JS + WebAudio** for the MVP (fast, no build step); upgrade to a
  framework only if the UI grows.
- **Audio to browser:** server renders fixed-length WAV clips, serves a URL, browser plays via
  `<audio>`. *(True live-streaming morph audition is a stretch goal — MVP renders fixed clips,
  which fits the turn-based game anyway.)*

### 7.7 How latency is handled (the recurring worry, solved)

- **Solo / beginner:** pre-generated decks → **0 ms** at play time.
- **Showdown:** one generation per round, hidden inside the composer's turn → feels intentional.
- **Telephone:** one gen per hop, hidden by **parallel chains** (everyone busy) + a "the message
  travels…" animation.
- **Forge workbench:** render-on-demand (~seconds) with a clear progress state; "variations" queue
  in the background.

The <200ms MRT2 number is internal step latency on a TPU — irrelevant here, because turn-based
play means generation never sits on the critical path of a player waiting.

---

## 8. Build phases

> Ordering honors: **advanced first**, **Telephone first among games**, **build them all**,
> beginner tier secondary.

- **Phase 0 — MRT2 Worker + Forge Core.** Model loads once; `PromptSpec → WAV` works end-to-end
  from a script. *Foundation for literally everything.*
- **Phase 1 — Forge workbench (web, solo, advanced).** Prompt controls + A↔B morph + variations +
  keep + crate + WAV/Audiotool export. **Immediately useful to a pro, demoable with zero game logic.**
- **Phase 2a — Broken Record (Telephone), hotseat. ✅ DONE.** One device passed around;
  compose → hear → guess → drift reveal with AI-judge scoring, tutor notes, leaderboard,
  chain fidelity.
- **Phase 2b — Broken Record, networked (the real version).** Everyone on their own phone,
  playing **simultaneously** (parallel chains, Gartic-Phone style — no waiting). Requirements
  gathered from playtesting the hotseat:
  - **Rooms + WebSockets:** join by code; live state sync; wait-for-all-then-advance.
  - **Multiple rounds / longer games** — host sets length; one pass is too short.
  - **Full workbench controls in-game** — the seed step uses the SAME controls as the
    workbench (A↔B blend, density, drums, key, length), not just a text box.
  - **Workbench → game bridge:** seed your chain by composing fresh OR by picking a saved
    clip **from your crate**. (Lab → arena.)
  - **Prompt-style hints** (tempo / instrument / mood / genre / production) as clickable chips
    so players new to MRT2 have a starting point — shared `prompt_hints.py` + `/api/hints`,
    reused in the workbench too.
  - Every hop is a **keepable stem** (keep/remove already wired in the workbench crate).
- **Phase 3 — Forge Battle + Showdown.** Constraint briefs, AI/peer judging, novelty award,
  leaderboard.
- **Phase 4 — Beginner tier.** Difficulty knob down + assist rails + the learning spine + tutor judge.

Each phase leaves something demoable, so a stalled later phase never sinks the demo.

## 9. Open questions (deferred, not blocking)

- Export for MVP: WAV stems only to start, Audiotool project next? (MIDI transcription = stretch.)
- Advanced judging: AI judge only at first, or peer blind voting from day one?
- Frontend: stay vanilla JS, or adopt a framework once party UIs land?
