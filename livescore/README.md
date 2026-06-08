# Score the Story

> A narrator speaks. MRT2 composes a live score in real time.
> No pre-composed music. No manual controls. The AI listens to the story.

---

## How it works

```
Microphone
   ↓
Voice Analyzer     → extracts energy, pitch, speech rate, brightness
   ↓
Feature Mapper     → translates voice features into MRT2 parameters
   ↓
MRT Controller     → sends parameters to MRT2 (MIDI or Python library)
   ↓
MRT2               → generates live music  →  Speakers
```

---

## Setup

### 1. Install dependencies

```bash
cd score-the-story
pip install -r requirements.txt
```

### 2. Choose your mode

**MIDI mode** (recommended — uses the MRT2 AU plugin you already have):

1. Open **Audio MIDI Setup** → MIDI Studio → create an **IAC Driver** bus named `MRT2 Control`
2. Open your DAW (GarageBand, Logic, Ableton)
3. Load **MRT2 AU** on a MIDI instrument track
4. Set the track's MIDI input to **MRT2 Control**
5. Set your DAW's sample rate to **48,000 Hz**
6. Run: `python main.py`

**Python library mode** (direct generation, no DAW needed):

1. Install: `pip install magenta-rt`
2. Run: `python main.py --mode python`

---

## Running

```bash
python main.py          # MIDI mode
python main.py --mode python   # Python library mode
```

**Keyboard commands while running:**

| Key | Action |
|-----|--------|
| `c` | Hold C major chord |
| `a` | Hold A minor chord |
| `d` | Hold D minor chord |
| `f` | Hold F major chord |
| `g` | Hold G major chord |
| `r` | Release chord |
| `q` | Quit |

---

## Test individual components

```bash
# Test mic + feature extraction only (no MRT2 needed)
python voice_analyzer.py

# Test feature→parameter mapping with simulated input
python feature_mapper.py
```

---

## Tuning the instrument

All the creative mappings live in **`feature_mapper.py`**:

- **`smoothing`** (default 0.70): higher = slower transitions, lower = more reactive
- **`_compute_target()`**: change which voice features drive which MRT2 parameters
- **Prompt A / Prompt B** in `mrt_controller.py`: the two musical poles the voice navigates between

---

## File structure

```
score-the-story/
├── main.py            ← entry point, runs the full pipeline
├── voice_analyzer.py  ← mic capture + feature extraction
├── feature_mapper.py  ← voice features → MRT2 parameters
├── mrt_controller.py  ← sends params to MRT2 (MIDI or Python library)
└── requirements.txt
```
