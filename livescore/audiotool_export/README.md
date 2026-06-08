# Score the Story → Audiotool

Turn a told story into an **editable, shareable, multiplayer Audiotool project**.

Score the Story already captures every telling as a `keepsake-*.json` (the scene
timeline + the speaker's musical signature). This tool takes the next step:

```
keepsake-*.json ──(Python)──► arrangement-*.json ──(this, Node)──► Audiotool project URL
   the telling      audiotool_arranger.py   symbolic score      @audiotool/nexus
```

The result is a real Audiotool session — three instrument tracks (bass, harmony,
sparkle) laid out scene-by-scene on a timeline, in each scene's key. Anyone with
the link can open it in the browser, edit the notes, swap instruments, add drums,
and collaborate live.

> **What this is NOT:** it does not upload the MRT2 audio. Audiotool is a
> *symbolic* DAW (MIDI + its own synths), so we hand it an editable **score**
> derived from the story, played by Audiotool's native instruments. That's the
> whole point — it's meant to be remixed, not frozen.

## Setup

```bash
# 1. From the project root, arrange a keepsake into a score:
python3 audiotool_arranger.py keepsake-20260606-080622.json
#    → writes arrangement-20260606-080622.json

# 2. Install this tool's deps:
cd audiotool_export
npm install

# 3. Get a Personal Access Token: https://developer.audiotool.com/personal-access-tokens

# 4. Set it up ONCE in a .env file (auto-loaded; never committed):
cp .env.example .env
#    then edit .env and paste your AT_PAT + the ARRANGEMENT path.

# 5. Push to Audiotool (creates a new project and prints its URL):
npm start
```

Because `.env` is auto-loaded, you only set the token once. You can still
override per-run inline (inline wins):

```bash
AT_PAT=at_pat_xxx ARRANGEMENT=../arrangement-20260606-080622.json npm start
```

To write into an **existing** project instead of creating one, set
`AT_PROJECT="https://beta.audiotool.com/studio?project=…"` in `.env` too.

## How it maps

| Arrangement (Python)        | Audiotool (Nexus entity)                          |
|-----------------------------|---------------------------------------------------|
| track (`foundation`)        | `gakki` soundfont (query `bass`) → `mixerChannel` |
| track (`harmony`)           | `gakki` soundfont (story-derived, e.g. `cello`)   |
| track (`texture`)           | `gakki` soundfont (query `bells`)                 |
| track (`melody`)*           | `gakki` soundfont → `mixerChannel`                |
| scene                       | one `noteRegion` per track on the timeline        |
| note (beats)                | `note` (ticks: `beats × Ticks.Beat`)              |

`gakki` is Audiotool's **soundfont sampler** — it plays real recorded
instruments, so the score sounds like its labels instead of a synth. The
exporter searches the preset library for a matching soundfont and applies it. If
nothing matches a track's query, it falls back to an audible `heisenberg` synth
rather than going silent.

See what a query actually matches (and grab specific preset IDs):

```bash
npm run presets gakki cello
npm run presets gakki "double bass"
npm run presets gakki bells
```

\* The `melody` track only appears when the arranger ran with `ANTHROPIC_API_KEY`
set (Claude writes a sung lead per scene, snapped to the scene's key). Without
it you get the deterministic 3-track arrangement. The exporter handles either.

All creative logic lives in `../audiotool_arranger.py`. This file is a faithful,
music-theory-free writer — change the sound design in the Python arranger, not here.

Built against `@audiotool/nexus@^0.0.8` (pre-1.0 — expect breaking changes).
