"""
llm_style_director.py
---------------------
Replaces hardcoded keyword maps with live LLM understanding.

Every few seconds:
  Whisper transcribes the narration
  Claude reads the transcript and outputs two music style descriptions:
    - style_a: darker / tenser pole for this scene
    - style_b: brighter / more triumphant pole for this scene
  MRT2 embeds both and smoothly morphs to them
  The narrator's voice continuously blends between a and b in real time

No keyword maps. No forced wording. Any scene, any genre, any instrument.

Setup:
  export ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
import re
import threading
import time
from collections import deque

SAMPLE_RATE = 48_000


def _clean_style(desc: str) -> str:
    """Tidy a Collider-style preset tag: strip punctuation and any vocal words
    that slip through (the system prompt already forbids them). Keeps the short
    tag format — no longer appends ', instrumental', which bloated the tag."""
    d = desc.strip().rstrip(".")
    vocal_words = ("vocal", "vocals", "singing", "sung", "lyrics",
                   "choir", "chant", "rap", "rapping", "acapella", "a cappella")
    parts = [seg for seg in d.split(",")
             if not any(w in seg.lower() for w in vocal_words)]
    return ", ".join(p.strip() for p in parts if p.strip()) or d


def _extract_json(raw: str) -> dict:
    """Parse the first {...} object out of a string, tolerating surrounding
    prose or markdown fences. Raises json.JSONDecodeError if none is valid."""
    raw = raw.strip()
    # Strip ```json fences if present
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

_SYSTEM_PROMPT = """\
You are a live music director for a generative music synth (MRT2).
You output SHORT style "preset" names — the synth continuously blends between
them based on the narrator's voice. Two poles per scene:
  a — the calmer / darker / more subdued pole
  b — the warmer / brighter / more open pole

STYLE FORMAT — the most important rule:
Each style is a SHORT preset name, 2–4 words, instrument-led and evocative,
Title Case. NOT a sentence, NOT comma-separated adjectives.
  GOOD:  "Mellow Lo-Fi Piano", "Warm Fingerpicked Guitar", "Soft Rhodes Keys",
         "Gentle Felt Piano", "Dusty Chillhop Beat", "Airy Synth Pad"
  BAD:   "low pulsing bass drone, minor key creeping dread, sustained tension"

SOUND — keep it flowing and coherent (this matters MOST):
- Stay in ONE warm world all session — lo-fi / chillhop / intimate acoustic —
  so the whole score feels like one record. But within that world there is a
  WIDE palette of colors. USE IT — don't default to piano + guitar every time.
- PALETTE to draw from (all warm and gentle, all blend cleanly): felt piano,
  Rhodes, Wurlitzer, electric piano, nylon guitar, fingerpicked guitar, warm
  upright bass, cello, viola, harp, music box, celeste, glockenspiel,
  vibraphone, marimba, mellow synth pad, mellotron, brushed drums, soft
  chillhop beat, warm tape ambience, kalimba, dulcimer.
- VARIETY across scene CHANGES: do NOT reuse the instruments you just used.
  Each time you change, bring at least one FRESH color from the palette and
  rotate (e.g. piano → Rhodes → harp → vibraphone → upright bass → music box).
  Keep the MOOD continuous, but let the instrumentation breathe and evolve.
- A and B (WITHIN one scene) must be CLOSE COUSINS — same family/register so
  the blend is seamless. Two shades of one mood, NEVER a hard contrast.
  GOOD: "Mellow Rhodes Keys" / "Warm Vibraphone". Variety lives ACROSS scenes,
  never inside a single A/B pair.
- INSTRUMENTAL only. Never vocals, singing, choir, lyrics, rap.
- MUSICAL, never noisy. Never use: glitchy, distorted, dissonant, atonal,
  harsh, noisy, chaotic, brass, fanfare, war, percussion-only. Convey tension
  with a darker, sparser version of the warm palette — not with noise.
- Favor gentle MOVEMENT (arpeggio, fingerpicked, soft groove, pulse) over a
  static held drone.

SCENE STABILITY — hold a groove, but FOLLOW the emotional arc:
The music should flow and settle, not flicker — but a mood that never responds
feels dead. Change when the FEELING genuinely turns, not on every sentence.
- KEEP → {"keep": true}  while the emotional tone continues (same feeling, same
  intensity). Minor wording changes or added detail → keep.
- CHANGE → {"keep": false, "a": "...", "b": "...", "key": "..."}  on a clear
  emotional TURN — even in the SAME place: calm→unease, tension→relief,
  stillness→triumph, sorrow→hope. Shift the poles to the NEW emotional range
  (warmer/brighter for hope or triumph; darker/sparser for unease or dread).
  Transitions are smooth, so don't be afraid to follow a real turn.

KEY: root note A–G (optional #/b) + " minor" or " major". MINOR for darker/
sadder, MAJOR for warmer/hopeful. Keep the key consistent across nearby scenes.

Output ONLY valid JSON.

Narration: "She sat alone by the window as the rain fell"
{"keep": false, "a": "Sad Felt Piano", "b": "Warm Fingerpicked Guitar", "key": "A minor"}

Narration: "And then, at last, the sun broke warm over the hills"
{"keep": false, "a": "Mellow Lo-Fi Keys", "b": "Bright Warm Piano", "key": "C major"}

Continuation (steady → KEEP):
Current music — a: Mellow Lo-Fi Keys | b: Bright Warm Piano
New narration: "He walked a little further down the quiet path"
{"keep": true}

Continuation (minor detail, still KEEP):
Current music — a: Sad Felt Piano | b: Warm Fingerpicked Guitar
New narration: "She wiped her eyes and picked up the old photograph"
{"keep": true}

Emotional turn in the same place (CHANGE — follow the arc):
Current music — a: Soft Felt Piano | b: Warm Lo-Fi Keys
New narration: "the birds went silent, and a cold unease crept in"
{"keep": false, "a": "Hushed Dark Piano", "b": "Muted Felt Keys", "key": "A minor"}

Then it lifts (CHANGE — follow the arc):
Current music — a: Hushed Dark Piano | b: Muted Felt Keys
New narration: "warm sunlight broke through and triumph filled the air"
{"keep": false, "a": "Warm Rolling Piano", "b": "Bright Uplifting Keys", "key": "C major"}
"""


class LLMStyleDirector:
    """
    Live LLM-driven music direction. Any narration → appropriate style poles.
    Falls back gracefully if ANTHROPIC_API_KEY is not set.
    """

    TRANSCRIBE_INTERVAL = 1.5    # seconds between transcription passes
    AUDIO_WINDOW        = 4.0    # seconds of audio fed to Whisper

    # ── Gates ────────────────────────────────────────────────────────────────
    SPEECH_RMS_GATE  = 0.015    # window must be at least this loud to transcribe
    SPEECH_FRACTION  = 0.20     # ...and at least this fraction of it above silence
    NO_SPEECH_MAX    = 0.55     # reject Whisper segments it flags as non-speech
    MIN_WORDS        = 3        # ignore one/two-word fragments
    DIRECTOR_INTERVAL = 4.0     # min seconds between Claude calls (NOT listening)
    CONTEXT_SNIPPETS  = 3       # recent transcripts kept as narrative context

    def __init__(self, analyzer, controller, *, transcribe_interval=None,
                 audio_window=None, cooldown=None, style_hint="", telemetry=None,
                 signature=None, session_log=None):
        self._analyzer    = analyzer
        self._controller  = controller
        self._telemetry   = telemetry
        self._signature   = signature      # SpeakerSignature (voice fingerprint)
        self._session_log = session_log    # SessionLog (for the keepsake render)
        self._sig_palette = ""             # speaker palette once computed
        self._sig_applied = False
        self._stop_event  = threading.Event()
        self.current_a    = "ambient neutral"
        self.current_b    = "ambient neutral"
        self._active      = False
        self.last_status  = "idle"   # shown in display for visibility
        self._last_call   = 0.0      # monotonic time of last Claude call
        self._recent      = deque(maxlen=self.CONTEXT_SNIPPETS)  # rolling context
        self._recent_scenes = deque(maxlen=4)   # recent style pairs → avoid repeats
        self._mute_event  = threading.Event()   # set = ignore the mic
        self._last_whisper_ms = 0.0  # timing of the most recent transcription
        self._style_hint  = style_hint
        # Per-preset timing overrides (fall back to the class defaults).
        if transcribe_interval is not None:
            self.TRANSCRIBE_INTERVAL = transcribe_interval
        if audio_window is not None:
            self.AUDIO_WINDOW = audio_window
        # Preset "cooldown" now governs how often we consult Claude, not how
        # long we go deaf — listening is always continuous.
        if cooldown is not None:
            self.DIRECTOR_INTERVAL = cooldown

    def start(self):
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("[LLM] No ANTHROPIC_API_KEY — set it with:")
            print("      export ANTHROPIC_API_KEY=sk-ant-...")
            print("[LLM] Running without LLM style direction.")
            return
        self._active = True
        threading.Thread(target=self._run, args=(api_key,), daemon=True).start()

    def inject(self, text: str):
        """Bypass Whisper — feed text directly into the LLM style pipeline.
        A manual inject always forces a scene change (operator intent)."""
        threading.Thread(target=self._process_text, args=(text, True), daemon=True).start()

    def toggle_mute(self) -> bool:
        """Mute/unmute the mic listener. Returns the new muted state.
        While muted, the voice loop ignores the mic — useful for typing
        injections without Whisper transcribing you mid-keystroke."""
        if self._mute_event.is_set():
            self._mute_event.clear()
        else:
            self._mute_event.set()
        return self._mute_event.is_set()

    @property
    def muted(self) -> bool:
        return self._mute_event.is_set()

    def stop(self):
        self._stop_event.set()

    @property
    def current_description(self):
        if not self._active:
            return "no API key"
        return self.current_a[:40]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self, api_key: str):
        import librosa
        import numpy as np
        import whisper
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

        print("[LLM] Loading Whisper base.en...")
        # base.en is markedly more accurate than tiny.en (fewer garbled words
        # feeding Claude) at a still-modest cost. Drop to 'tiny.en' if too slow.
        stt = whisper.load_model('base.en')
        print("[LLM] Ready — narration → Claude → MRT2 style poles")

        last_text = ""

        while not self._stop_event.is_set():
            time.sleep(self.TRANSCRIBE_INTERVAL)

            if self._mute_event.is_set():
                continue   # mic muted — only manual `t` injections get through

            audio_48k = self._analyzer.get_audio_for_transcription(self.AUDIO_WINDOW)
            if len(audio_48k) < SAMPLE_RATE // 2:
                continue

            # ── Gate 1: energy ── skip silence/noise before wasting Whisper on it.
            # Whisper tiny hallucinates filler ("Awesome.", "I'll listen.") when
            # fed near-silent audio, so only transcribe windows that actually
            # contain speech-level loudness. (This is the ONLY thing that pauses
            # listening — and only because there's nothing to hear.)
            if not self._has_speech(audio_48k, np):
                continue

            # Feed the speaker fingerprint with EVERY voiced window — from every
            # person who speaks — so it becomes the blended voice of the whole
            # moment. The live palette locks once (for stable live sound); the
            # keepsake later uses the full blend of everyone.
            if self._signature is not None:
                self._signature.add_audio(audio_48k)
                if not self._sig_applied and self._signature.ready:
                    sig = self._signature.signature()
                    self._sig_palette = sig["palette"]
                    self._sig_applied = True
                    # Personalise the live KEY too: switch the locked key to the
                    # speaker's signature key. Applied at the next scene change so
                    # the re-tune hides inside an expected transition.
                    if hasattr(self._controller, "set_session_key"):
                        self._controller.set_session_key(sig["key"])
                    print(f"[voice] live palette → {self._signature.describe()}")

            audio_16k = librosa.resample(audio_48k, orig_sr=SAMPLE_RATE, target_sr=16_000)

            try:
                w0 = time.monotonic()
                result = stt.transcribe(
                    audio_16k,
                    fp16=False,
                    language='en',
                    temperature=0.0,                 # deterministic, less drift
                    condition_on_previous_text=False, # don't loop on own output
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                )
                self._last_whisper_ms = (time.monotonic() - w0) * 1000.0
            except Exception as e:
                print(f"[LLM] transcription error: {e}")
                continue

            # ── Gate 3: confidence ── trust Whisper's own non-speech flag.
            if self._is_non_speech(result):
                continue

            text = result['text'].strip()
            if (not text or len(text.split()) < self.MIN_WORDS or text == last_text):
                continue

            # We ALWAYS listen and accumulate — this is the continuous ingest.
            last_text = text
            self._recent.append(text)
            print(f"[LLM] heard: \"{text[:80]}\"")
            if self._telemetry:
                self._telemetry.event("heard", text=text[:80])
            if self._session_log is not None:        # the story said, for the keepsake
                self._session_log.add_transcript(text)

            # The DIRECTOR (Claude) is rate-limited only to control cost/churn —
            # listening above never pauses. Scene stability is handled by Claude's
            # keep/change decision, not by going deaf.
            if time.monotonic() - self._last_call < self.DIRECTOR_INTERVAL:
                continue
            self._last_call = time.monotonic()
            self._process_text(" ".join(self._recent))

    def _has_speech(self, audio_48k, np) -> bool:
        """True only if the window has real speech-level energy."""
        rms = float(np.sqrt(np.mean(audio_48k ** 2)))
        if rms < self.SPEECH_RMS_GATE:
            return False
        # Also require a chunk of the window to be voiced, not one loud blip.
        block = 4_800  # 100 ms
        n = len(audio_48k) // block
        if n == 0:
            return rms >= self.SPEECH_RMS_GATE
        loud = 0
        for i in range(n):
            seg = audio_48k[i * block:(i + 1) * block]
            if float(np.sqrt(np.mean(seg ** 2))) >= self.SPEECH_RMS_GATE:
                loud += 1
        return (loud / n) >= self.SPEECH_FRACTION

    def _is_non_speech(self, result) -> bool:
        """True if Whisper itself flags the audio as probably not speech."""
        segments = result.get('segments') or []
        if not segments:
            return True
        probs = [s.get('no_speech_prob', 0.0) for s in segments]
        return (sum(probs) / len(probs)) > self.NO_SPEECH_MAX

    def _process_text(self, text: str, force: bool = False):
        """Send text to Claude and apply new style poles — unless Claude judges
        the scene unchanged. `force=True` (manual `t` inject) always changes.
        """
        import anthropic

        client = getattr(self, '_client', None)
        if client is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                print("[LLM] No API key — cannot process text")
                return
            self._client = anthropic.Anthropic(api_key=api_key)
            client = self._client

        # Build the request. Forced injects always produce a fresh pair; voice
        # narration includes the current music so Claude can decide keep vs change.
        if force:
            user = (f"Narration: {text}\n\n"
                    "The operator deliberately set this scene — always provide a "
                    "fresh instrumental pair.")
            prefill = '{"keep": false, "a": "'
        else:
            recent = " · ".join(self._recent_scenes)
            user = (f"Current music — A: {self.current_a} | B: {self.current_b}\n"
                    + (f"Recently used (if you CHANGE, pick FRESH colors, avoid "
                       f"these): {recent}\n" if recent else "")
                    + f"New narration: {text}\n\n"
                    "Decide: same scene (keep), or a real scene change with FRESH "
                    "instrument colors from the palette?")
            prefill = '{"keep":'   # no trailing space — API rejects it

        self.last_status = "calling Claude..."
        print(f"[LLM] directing: \"{text[:80]}\"")
        raw = ""                          # so error handlers never see it unbound
        try:
            t0 = time.monotonic()
            system_text = _SYSTEM_PROMPT
            if self._style_hint:
                system_text += f"\n\nMODE GUIDANCE (bias every style to this):\n{self._style_hint}"
            if self._sig_palette:
                system_text += (f"\n\nSPEAKER PALETTE — a gentle tonal lean for this "
                                f"telling, derived from the narrator's voice: "
                                f"{self._sig_palette}. Let it color the sound, but "
                                f"freely use whatever instruments each scene needs — "
                                f"vary the instrument families, don't repeat one.")
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,                       # short tags → fewer tokens
                # Cache the (constant) system prompt so repeat calls skip
                # re-processing it — big latency + cost win at our call rate.
                system=[{"type": "text", "text": system_text,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": prefill},
                ],
            )
            claude_ms = (time.monotonic() - t0) * 1000.0
            raw  = prefill + resp.content[0].text
            data = _extract_json(raw)

            # Scene unchanged → hold the music, change nothing.
            if not force and data.get("keep"):
                self.last_status = "same scene — holding"
                print(f"[LLM] same scene — holding music  "
                      f"[perf] whisper {self._last_whisper_ms:.0f}ms · "
                      f"claude {claude_ms:.0f}ms")
                if self._telemetry:
                    self._telemetry.event("hold",
                                          whisper=round(self._last_whisper_ms),
                                          claude=round(claude_ms))
                return

            style_a = _clean_style(data["a"])
            style_b = _clean_style(data["b"])
            key     = str(data.get("key", "")).strip()

            self.current_a   = style_a
            self.current_b   = style_b
            self._recent_scenes.append(f"{style_a} / {style_b}")   # for anti-repeat
            self.last_status = "OK"
            print(f"[LLM] A: {style_a}")
            print(f"[LLM] B: {style_b}")
            print(f"[LLM] key: {key or '(none)'}")
            print(f"[perf] whisper {self._last_whisper_ms:.0f}ms · "
                  f"claude {claude_ms:.0f}ms")
            if self._telemetry:
                self._telemetry.event("scene", a=style_a[:40], b=style_b[:40],
                                      key=key, whisper=round(self._last_whisper_ms),
                                      claude=round(claude_ms))
            if self._session_log is not None:        # record for the keepsake
                self._session_log.add_scene(style_a, style_b, key)
            self._controller.set_prompts(style_a, style_b, key)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if raw:                       # we got a response but couldn't parse it
                self.last_status = f"bad JSON: {raw[:60]}"
                print(f"[LLM] bad JSON from Claude ({e}): {raw[:120]}")
            else:                         # the call itself failed (e.g. bad API key)
                self.last_status = f"call failed: {e}"
                print(f"[LLM] Claude call failed (check ANTHROPIC_API_KEY): {e}")
        except Exception as e:
            self.last_status = f"ERR: {e}"
            print(f"[LLM] API error: {e}")
