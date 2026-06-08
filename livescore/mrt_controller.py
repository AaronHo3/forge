"""
mrt_controller.py
-----------------
Sends control parameters to MRT2 in real time.

TWO APPROACHES — pick the one that works at the hackathon:

  A) MIDI  (default, works right now with the AU plugin in a DAW)
     Python → virtual MIDI port → MRT2 AU (in GarageBand / Logic / Ableton)
     Controls: notes for harmony steering, CC for blend/chaos

  B) Python library  (requires `pip install magenta-rt` + MRT2 Python API)
     Python → magenta_rt.system → generates audio directly
     Full control: text prompts, style blending, chaos

Switch between them by passing mode="midi" or mode="python" to MRTController().

NOTE ── MIDI CC mapping used here:
  CC 1  (Mod Wheel) → prompt blend  (0–127 maps to 0.0–1.0)
  CC 11 (Expression)→ chaos / intensity
  CC 64 (Sustain)   → drums on/off (>64 = on)
These are conventional assignments but can be remapped in MRT2's AU UI.
"""

import threading
import time
from typing import Callable, Protocol, TypedDict

import numpy as np

import config
import dsp
import harmony
from feature_mapper import MRTParams


class HealthStatus(TypedDict):
    """The engine liveness contract returned by PythonMRTController.health()."""
    ok: bool
    fault: str | None
    last_chunk_age_s: float | None
    starves: int
    underruns: int


# ══════════════════════════════════════════════════════════════════════
#  APPROACH A — MIDI Controller
# ══════════════════════════════════════════════════════════════════════

class MIDIMRTController:
    """
    Sends MIDI messages to MRT2's AudioUnit via a virtual MIDI port.

    Setup:
      1. Open Audio MIDI Setup → MIDI Studio → Create a new IAC Driver bus
         and name it "MRT2 Control"
      2. In your DAW, load MRT2 AU on a MIDI track and set its
         input to "MRT2 Control"
      3. Run this script — MIDI flows straight through

    Harmony notes:
      Call `hold_chord(["C3", "E3", "G3"])` to tell MRT2 which harmony
      to follow. The model generates an ensemble around that chord.
      Prompt blend (CC1) and chaos (CC11) then shape the style.
    """

    # Prompts are text labels shown in MRT2's UI.
    # These match typical MRT2 style presets — update to match your setup.
    PROMPT_A = "dark minor strings, tense, cinematic, slow"
    PROMPT_B = "warm bright piano jazz, uplifting, gentle, major"

    def __init__(self, port_name: str = "MRT2 Control"):
        try:
            import rtmidi
            self._midi_out = rtmidi.MidiOut()
            available = self._midi_out.get_ports()
            print(f"[MIDI] Available ports: {available}")

            # Find our virtual port or fall back to the first available
            target = next(
                (i for i, p in enumerate(available) if port_name in p),
                None
            )
            if target is not None:
                self._midi_out.open_port(target)
                print(f"[MIDI] Connected to port: {available[target]}")
            else:
                # Create a virtual port (works on macOS/Linux)
                self._midi_out.open_virtual_port(port_name)
                print(f"[MIDI] Created virtual port: {port_name}")

            self._channel = 0   # MIDI channel 1 (0-indexed)
            self._active_notes = []
            self._ok = True

        except ImportError:
            print("[MIDI] python-rtmidi not installed. Run: pip install python-rtmidi")
            self._ok = False
        except Exception as e:
            print(f"[MIDI] Could not open MIDI port: {e}")
            self._ok = False

    def start(self):
        print(f"[MIDI] Controller ready.")
        print(f"       Prompt A → {self.PROMPT_A}")
        print(f"       Prompt B → {self.PROMPT_B}")

    def update(self, params: MRTParams):
        """Send current MRT2 parameters as MIDI CC messages."""
        if not self._ok:
            return
        blend_cc  = int(params.prompt_blend * 127)
        chaos_cc  = int(params.chaos * 127)
        drums_cc  = 100 if params.drums_on else 0

        self._send_cc(1,  blend_cc)   # Mod wheel  → prompt blend
        self._send_cc(11, chaos_cc)   # Expression → chaos
        self._send_cc(64, drums_cc)   # Sustain    → drums on/off

    def hold_chord(self, notes: list[str], velocity: int = 80):
        """
        Tell MRT2 which harmony to follow.
        Call with a list of note names, e.g. ["C3", "E3", "G3"].
        """
        if not self._ok:
            return
        self._release_all()
        for note_name in notes:
            midi_note = harmony.note_to_midi(note_name)
            self._send_note_on(midi_note, velocity)
            self._active_notes.append(midi_note)

    def release_chord(self):
        self._release_all()

    def stop(self):
        self._release_all()

    # ── Internal ──────────────────────────────────────────────────────

    def _send_cc(self, cc: int, value: int):
        self._midi_out.send_message([0xB0 | self._channel, cc, value])

    def _send_note_on(self, note: int, velocity: int):
        self._midi_out.send_message([0x90 | self._channel, note, velocity])

    def _send_note_off(self, note: int):
        self._midi_out.send_message([0x80 | self._channel, note, 0])

    def _release_all(self):
        for n in self._active_notes:
            self._send_note_off(n)
        self._active_notes = []

    # Note-name parsing lives in harmony.note_to_midi (shared module).


# ══════════════════════════════════════════════════════════════════════
#  APPROACH B — Python Library (magenta-rt)
# ══════════════════════════════════════════════════════════════════════

class MRTBackend(Protocol):
    """The slice of the Magenta RealTime system the live loop depends on. The
    real implementation is MagentaRT2SystemMlxfn; tests inject a fake so the
    loop can run with no model. `embed_style` turns a prompt into a style
    vector; `generate` produces one audio chunk plus the rolling state to
    thread into the next call."""
    # Underscore-named to MATCH the third-party backend's attribute exactly
    # (MagentaRT2SystemMlxfn exposes `_num_notes`); it's the note-mask width.
    _num_notes: int

    def embed_style(self, text_or_audio, **kwargs) -> np.ndarray: ...

    def generate(self, *, style=None, notes=None, drums=None,
                 cfg_musiccoca=None, frames: int = 25, state=None,
                 temperature=None, top_k=None) -> tuple: ...


class OutputStreamLike(Protocol):
    """The audio output stream the loop drives. sounddevice.OutputStream
    satisfies it; a fake implements these no-ops for headless tests."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def _default_backend() -> MRTBackend:
    """Create the real MRT2 MLX backend. Imported lazily and run on the audio
    thread (all MLX work must stay on one thread)."""
    from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
    return MagentaRT2SystemMlxfn(size='mrt2_base')


class PythonMRTController:
    """
    Controls MRT2 directly via magenta-rt (MagentaRT2System, MLX backend).

    Generates stereo 48kHz audio as numpy arrays and plays them via sounddevice.
    GarageBand is not involved — audio goes straight to the system output.

    Voice blend controls style embedding interpolation (A=dark/tense, B=bright/warm).
    Chaos controls cfg_musiccoca guidance strength (higher = more prompt-adherent).
    Drums_on maps to the drums conditioning token.
    """

    PROMPT_A = "dark ambient electronic drone, minimal, atmospheric"
    PROMPT_B = "warm lo-fi hip hop beat, chill, gentle piano"
    SAMPLE_RATE = config.SAMPLE_RATE

    # Classifier-free guidance for the style embedding. Too high (>~3.5) makes
    # MRT2 over-saturate into harsh, noisy, distorted audio. Keep it musical.
    CFG_BASE = config.STYLE_CFG  # guidance FLOOR — high enough that quiet/paused
                            # passages stay ON-STYLE and don't drift to silence
                            # (matches the keepsake's known-good CFG). This is the
                            # main lever against the "fades when I speak softly" decay.
    CFG_SPAN = 1.0          # added at chaos=1 → max guidance 3.0 (still musical)

    # MRT2 defaults (temp 1.3, top_k 40) sample "hot" → wandering, noisy output.
    # Lower = more coherent and musical. These are the single biggest sound
    # quality lever short of harmony conditioning.
    TEMPERATURE = config.TEMPERATURE
    TOP_K = 32              # a little more variety than 24 → less likely to
                           # collapse into the silent attractor when energy is low

    # Harmony note-mask logic (key -> MRT2 notes) lives in harmony.py, shared
    # with the keepsake renderer so the two never drift out of tune.

    TARGET_RMS   = 0.14     # loudness-normalise so all scenes sit at the SAME
                            # perceived level — no jarring jumps between scenes.
    LIMIT_THRESH = 0.80     # soft-limiter knee: samples above this are smoothly
                            # compressed (tanh) toward 1.0, taming transient spikes.

    # Auto-gain safety rails. The loudness normaliser divides by a chunk's RMS,
    # so a near-silent transitional chunk (the model briefly thinning out at a
    # scene change) would send the gain toward infinity and blow up the NEXT
    # chunk into static. We refuse to adapt on sub-musical chunks and clamp the
    # gain to a sane band so it can never explode.
    AGC_MIN_RMS  = 0.02     # below this a chunk is a transient, not music — hold gain
    AGC_GAIN_MIN = 0.30     # gain can never drop below / rise above this band
    AGC_GAIN_MAX = 3.00

    # Decay watchdog. A single autoregressive stream threaded for a long time
    # drifts toward near-silence (silence is a stable attractor for the model).
    # If the RAW generated level stays below DECAY_RMS for DECAY_CHUNKS in a row,
    # we re-seed the state (generate the next chunk fresh from the current style)
    # so the music re-energises instead of fading out over a long telling.
    DECAY_RMS    = config.DECAY_RMS     # below this = "dying", not a soft passage
    DECAY_CHUNKS = config.DECAY_CHUNKS  # ~4s of near-silence before we re-seed

    # Continuous (Collider-style) generation: keep generating forever, threading
    # MRT2's state so the music evolves and never repeats. Scene changes swap the
    # conditioning mid-stream (no restart), so the smaller the lead buffer, the
    # sooner a change is heard.
    #
    # CRITICAL: the buffer must exceed the time to generate ONE chunk, or it
    # empties mid-generation and stutters. We generate small chunks so the buffer
    # (and thus the response latency) can be small without starving.
    CHUNK_FRAMES = 20       # frames per generate() call. 25 = 1.0s; 20 = 0.8s.
                            # Bigger chunks = more COHERENT audio (the model settles
                            # into a groove). The tradeoff is a larger safe buffer.
    LEAD_SECONDS = 1.2      # buffer ahead of playback. Must stay comfortably above
                            # the per-chunk gen time (~0.64s for 20 frames). Bigger
                            # = safer/coherent; this is roughly the change latency.
    TRIM_AT_SEC  = 3.0      # trim already-played audio once this much is behind
    KEEP_BEHIND  = 0.5      # ...keeping this much history

    # Style smoothing — how the conditioning moves when the scene changes.
    # Instead of snapping to a new style embedding (a hard cut that knocked the
    # model out of its groove), the live style GLIDES toward the new target a
    # little each chunk, and every scene is partly tethered to the coherent
    # "bootup" foundation poles so it can never lurch into incoherent territory.
    ANCHOR = 0.35           # fraction of every scene pulled back toward the
                            # foundation, so it modulates AROUND the startup sound.

    def __init__(self, morph_step: float = 0.30,
                 default_a: str | None = None, default_b: str | None = None,
                 default_key: str | None = None, telemetry=None,
                 backend_factory: Callable[[], MRTBackend] | None = None,
                 enable_drums: bool = False,
                 output_factory: Callable[..., OutputStreamLike] | None = None):
        self._telemetry = telemetry
        self._enable_drums = enable_drums   # opt-in; drums default OFF (see _gen_chunk)
        # Output-stream seam: None -> a real sounddevice OutputStream. Tests inject
        # a fake so the generation loop can run with no audio device.
        self._output_factory = output_factory
        # The MRT2 backend is created lazily on the audio thread via this factory,
        # so tests can inject a fake (or a deliberately failing) one without
        # loading the real model or opening an audio device.
        self._backend_factory = (backend_factory if backend_factory is not None
                                 else _default_backend)
        # Fault state: set by the audio-loop supervisor if generation ever dies,
        # so the app can surface a clear error instead of silent dead air.
        # _fault is guarded by _lock; _fault_event lets callers block until a
        # fault instead of polling. _last_chunk_time is a GIL-atomic float.
        self._fault: str | None = None
        self._fault_event = threading.Event()
        self._last_chunk_time = 0.0   # monotonic time the last chunk was generated
        self._out_rms = 0.0          # RMS of the most recent output block
        self._out_block = None       # mono copy of it, for spectral centroid
        self._current_params = MRTParams()
        self._lock = threading.Lock()       # guards params/pending
        self._pb_lock = threading.Lock()    # guards playback buffers
        self._stop_event = threading.Event()
        self._pending_prompts = None        # (str, str) or None
        self._prompt_a = default_a or self.PROMPT_A
        self._prompt_b = default_b or self.PROMPT_B
        self._blend = 0.0                   # live (brightness) → filter cutoff
        self._chaos = 0.0                   # live (energy)     → volume

        # Scene transition length derived from the preset's morph_step:
        # slower presets (meditation) cross-fade scenes over more seconds.
        self._transition_seconds = min(12.0, max(1.5, 1.0 / morph_step))

        # Per-chunk glide rate for the style embedding, derived from that same
        # transition length: slower presets morph more gently. The style chases
        # its target by this fraction each chunk (exponential approach).
        chunk_sec = self.CHUNK_FRAMES / 25.0
        self._style_morph = min(0.5, max(0.05, chunk_sec / self._transition_seconds))
        # Harmony locked for the whole session. If a default key is given it's set
        # from the very first chunk (no mid-stream onset). If None, we fall back to
        # locking on the first key Claude provides (the old behaviour).
        self._session_key = default_key or None
        self._pending_key = None        # signature key, applied at next scene change

        # ── Playback state (shared with the sd callback) ──────────────────
        # Continuous generation, like the Collider: a growing stream of fresh
        # audio for the current scene, and a frozen leftover of the previous
        # scene that fades out during a crossfade. No looping.
        self._cur_stream  = None   # np[?,2] current scene, grows continuously
        self._cur_pos     = 0
        self._prev_stream = None   # np[?,2] previous scene's leftover, fading out
        self._prev_pos    = 0
        self._trans = 1.0          # 1.0 = fully on cur; ramps 0→1 on a scene change
        self._trans_dur = self._transition_seconds   # capped per-transition to the
                                                     # available leftover (no gap)
        self._zi0 = None           # low-pass filter state, left/right channels
        self._zi1 = None

        # Profiling — surfaced in [perf] lines and the `s` status command.
        self._underruns   = 0     # sounddevice-reported output underflows
        self._starves     = 0     # buffer ran dry → we played silence (= stutter)
        self._gen_avg_ms  = 0.0

    def start(self):
        # All MLX operations must happen in one thread — spawn and return.
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def set_prompts(self, prompt_a: str, prompt_b: str, key: str | None = None):
        """Switch musical poles + key (thread-safe). Render happens in audio thread."""
        with self._lock:
            self._pending_prompts = (prompt_a, prompt_b, key)

    def set_session_key(self, key: str):
        """Switch the locked session key — e.g. to the speaker's signature key
        once it's known. Applied at the NEXT scene change so the re-tune hides
        inside a transition the listener already expects, not mid-phrase."""
        if key:
            with self._lock:
                self._pending_key = key

    def _gen_chunk(self, mrt, style, params, state, np, notes=None):
        """Generate one ~1s chunk and update the rolling gen-time average."""
        cfg   = self.CFG_BASE + params.chaos * self.CFG_SPAN   # capped, musical
        # Drums are opt-in. The mapper's auto-threshold can flip them on/off
        # mid-telling, which is distracting for storytelling, so they default OFF.
        # When enable_drums is set, the mapper's drums_threshold actually drives
        # them — no more silently-dead config.
        drums = [1] if (self._enable_drums and params.drums_on) else [0]
        tg = time.monotonic()
        wav, state = mrt.generate(style=style, notes=notes, drums=drums,
                                  cfg_musiccoca=cfg, frames=self.CHUNK_FRAMES,
                                  state=state,
                                  temperature=self.TEMPERATURE, top_k=self.TOP_K)
        gen_ms = (time.monotonic() - tg) * 1000.0
        self._gen_avg_ms = (0.8 * self._gen_avg_ms + 0.2 * gen_ms
                            if self._gen_avg_ms else gen_ms)
        self._last_chunk_time = time.monotonic()   # heartbeat for health()
        return np.ascontiguousarray(wav.samples), state

    def _read_stream(self, buf, pos, n, np):
        """Read n frames from the growing stream buffer; if generation hasn't
        produced enough yet, pad with silence and count it as a starve (the
        audible stutter — sounddevice won't flag it since we hand it valid data)."""
        L = buf.shape[0]
        if pos >= L:
            self._starves += 1
            return np.zeros((n, 2), dtype=np.float32), pos
        end = pos + n
        if end <= L:
            return buf[pos:end], end
        self._starves += 1
        avail = buf[pos:L]
        pad = np.zeros((n - avail.shape[0], 2), dtype=np.float32)
        return np.concatenate([avail, pad]), L

    def _audio_loop(self):
        """Supervisor around the generation loop. If anything inside fails (model
        load, audio device, or a generation error), record it as a fault the rest
        of the app can read via health() — instead of the thread dying silently
        while the app still looks like it is running."""
        try:
            self._run_audio_loop()
        except Exception as e:
            import traceback
            fault = f"{type(e).__name__}: {e}"
            with self._lock:
                self._fault = fault
            self._fault_event.set()   # wake anyone blocked on a fault
            traceback.print_exc()
            print(f"[MRT2] FATAL — generation stopped: {fault}")

    def _run_audio_loop(self):
        """Continuously generate audio (Collider-style) and play it. MRT2's state
        threads from chunk to chunk so the music evolves and never repeats. All
        MLX work stays in this thread; the callback only does numpy + a filter."""
        import sounddevice as sd
        import numpy as np
        from scipy.signal import lfilter

        print("[MRT2] Loading model...")
        mrt = self._backend_factory()
        print("[MRT2] Embedding prompts...")
        style_a = mrt.embed_style(self._prompt_a)
        style_b = mrt.embed_style(self._prompt_b)

        self._zi0 = np.zeros(1, dtype=np.float32)
        self._zi1 = np.zeros(1, dtype=np.float32)

        def _callback(outdata, frames, _time, _status):
            if _status:
                self._underruns += 1
            with self._pb_lock:
                if self._cur_stream is None:
                    outdata.fill(0.0)
                    return
                out, self._cur_pos = self._read_stream(
                    self._cur_stream, self._cur_pos, frames, np)
                out = out.copy()

                # Scene-change crossfade: ramp from the previous scene's leftover.
                if self._prev_stream is not None and self._trans < 1.0:
                    pseg, self._prev_pos = self._read_stream(
                        self._prev_stream, self._prev_pos, frames, np)
                    step = frames / (self._trans_dur * self.SAMPLE_RATE)
                    t1   = min(1.0, self._trans + step)
                    ramp = np.linspace(self._trans, t1, frames, dtype=np.float32)[:, None]
                    out  = ramp * out + (1.0 - ramp) * pseg
                    self._trans = t1
                    if t1 >= 1.0:
                        self._prev_stream = None

                # ── Within-scene voice dynamics ──────────────────────────────
                # Brightness → one-pole low-pass cutoff; energy → volume swell.
                blend = float(self._blend)
                chaos = float(self._chaos)
                alpha = 0.35 + 0.65 * blend          # 0.35 muffled … 1.0 open
                b = [alpha]; a = [1.0, -(1.0 - alpha)]
                l, self._zi0 = lfilter(b, a, out[:, 0], zi=self._zi0)
                r, self._zi1 = lfilter(b, a, out[:, 1], zi=self._zi1)
                gain = 0.8 + 0.2 * chaos
                mixed = np.stack([l, r], axis=1).astype(np.float32) * gain
                mixed = dsp.soft_limit(mixed, self.LIMIT_THRESH)  # tame transient spikes
                outdata[:] = mixed
                mono = mixed.mean(axis=1)
                self._out_rms = float(np.sqrt(np.mean(mono ** 2)))
                self._out_block = mono

        # Factory defaults to the real device; tests inject a fake. Both get the
        # identical kwargs, so the production path is byte-for-byte unchanged.
        make_stream = self._output_factory or sd.OutputStream
        stream = make_stream(
            samplerate=self.SAMPLE_RATE, channels=2, dtype='float32',
            callback=_callback, latency='high',
        )
        stream.start()

        LEAD = int(self.LEAD_SECONDS * self.SAMPLE_RATE)
        TRIM = int(self.TRIM_AT_SEC  * self.SAMPLE_RATE)
        KEEP = int(self.KEEP_BEHIND  * self.SAMPLE_RATE)

        # Current scene's two pole embeddings. The live voice blend interpolates
        # between them every chunk — like dragging the Collider dot between two
        # circles. emb_a = dark pole, emb_b = bright pole.
        # Coherent "bootup" foundation — the preset's default poles, embedded once
        # and NEVER changed. Every scene is partly pulled back toward this so the
        # music always modulates AROUND the sound you hear at startup instead of
        # abandoning it. emb_a/emb_b are the current (Claude-directed) scene poles.
        found_a, found_b = style_a, style_b
        emb_a, emb_b = style_a, style_b
        # Harmony present from the FIRST chunk if a session key is locked, so the
        # note constraint never switches on mid-stream (the first-scene-change
        # stutter). None → free, exactly like before, until Claude sets a key.
        cur_notes = harmony.build_notes(self._session_key, mrt._num_notes)
        state     = None

        def _blend(pa, pb, t):
            return (1.0 - t) * pa + t * pb

        def _target(t):
            """The style this scene should settle on: the current Claude poles
            blended at t, then tethered ANCHOR of the way back to the foundation
            so it can never drift into incoherent territory."""
            scene = _blend(emb_a, emb_b, t)
            base  = _blend(found_a, found_b, t)
            return (1.0 - self.ANCHOR) * scene + self.ANCHOR * base

        # scene_style is what we actually condition on; it GLIDES toward
        # target_style a little each chunk (no hard snap), so a scene change feels
        # like a smooth morph rather than a cut. The voice still reacts instantly
        # via volume + filter (callback) and chaos/drums (params).
        scene_style  = None
        target_style = None

        # Smoothed loudness gain — adapts across scenes WITHOUT a restart, so
        # levels stay consistent even though we never re-seed a fresh chunk.
        cur_g = 1.0
        low_streak = 0          # consecutive near-silent chunks (decay watchdog)

        def _emit(ci):
            """Apply smoothed RMS-normalised gain so every chunk sits at the same
            perceived level. Robust to transitional dips: a chunk that's abnormally
            quiet (the model briefly thinning out at a scene change) does NOT drive
            the gain — otherwise the next normal chunk gets over-amplified into
            static. The gain is also clamped to a sane band so it can never explode."""
            nonlocal cur_g
            rms = float(np.sqrt(np.mean(ci ** 2)))
            if rms > self.AGC_MIN_RMS:                     # only adapt on real music
                cur_g = 0.85 * cur_g + 0.15 * (self.TARGET_RMS / rms)
                cur_g = float(min(self.AGC_GAIN_MAX,
                                  max(self.AGC_GAIN_MIN, cur_g)))
            return (ci * cur_g).astype(np.float32)

        # Seed the first chunk so playback can begin — start fully settled on the
        # foundation target (no morph needed yet).
        with self._lock:
            params = self._current_params
        target_style = _target(float(params.prompt_blend))
        scene_style  = target_style
        c0, state = self._gen_chunk(mrt, scene_style,
                                    params, state, np, cur_notes)
        with self._pb_lock:
            self._cur_stream = _emit(c0)
            self._cur_pos = 0
            self._trans = 1.0
        print(f"[MRT2] Ready (continuous).\n       A → {self._prompt_a}\n       B → {self._prompt_b}")

        while not self._stop_event.is_set():
            # 1) New scene? Just swap the conditioning — NO restart. The running
            #    stream's state keeps flowing and morphs into the new style on the
            #    next chunk, audible within ~LEAD seconds (Collider-style).
            with self._lock:
                pending = self._pending_prompts
                if pending:
                    self._pending_prompts = None
                params = self._current_params

            if pending:
                prompt_a, prompt_b, key = pending
                te = time.monotonic()
                emb_a = mrt.embed_style(prompt_a)
                emb_b = mrt.embed_style(prompt_b)
                # Harmony is locked ONCE per session: the first key we're given is
                # held for the whole telling, so the key never yanks mid-stream
                # (re-masking the keyboard was one of the hard "shocks"). The scene's
                # mood still comes through the style embedding (minor/dark vs warm).
                # Adopt the speaker's signature key (set once it's known) at this
                # scene change — the re-tune hides inside an expected transition.
                with self._lock:
                    pend_key = self._pending_key
                    self._pending_key = None
                if pend_key:
                    self._session_key = pend_key
                    cur_notes = harmony.build_notes(self._session_key, mrt._num_notes)
                elif self._session_key is None and key:
                    self._session_key = key
                    cur_notes = harmony.build_notes(self._session_key, mrt._num_notes)
                # New target — but DON'T snap scene_style. It glides there over the
                # next few chunks, so the transition is a smooth morph, not a cut.
                target_style = _target(float(params.prompt_blend))
                embed_ms = (time.monotonic() - te) * 1000.0
                with self._pb_lock:
                    lead_s = (self._cur_stream.shape[0] - self._cur_pos) / self.SAMPLE_RATE
                print(f"[MRT2] New scene → A: {prompt_a[:40]} | B: {prompt_b[:40]}  key: {self._session_key or '(free)'}")
                print(f"[perf] embed {embed_ms:.0f}ms · heard in ~{lead_s:.1f}s · "
                      f"gen {self._gen_avg_ms:.0f}ms/chunk · starves {self._starves}")
                if self._telemetry:
                    self._telemetry.event("render", ms=round(embed_ms + lead_s * 1000),
                                          chunk=round(self._gen_avg_ms),
                                          underruns=self._underruns)
                continue

            # 2) Keep the buffer ~LEAD seconds ahead — paced continuous generation
            #    at the scene's fixed style (coherent), voice shaping it live.
            with self._pb_lock:
                lead = self._cur_stream.shape[0] - self._cur_pos
            if lead < LEAD:
                with self._lock:
                    params = self._current_params
                # The voice continuously sets WHERE between the two scene poles we
                # aim (dark ↔ warm), and scene_style GLIDES toward it. So even
                # inside ONE Claude scene the music slides with the emotion —
                # granularity from the voice every chunk — while the morph keeps it
                # smooth and the anchor keeps it coherent. (Both poles are now
                # "close cousins", so interpolating between them is seamless.)
                target_style = _target(float(params.prompt_blend))
                scene_style = scene_style + self._style_morph * (target_style - scene_style)

                # Decay watchdog: if the stream has been fading for a while,
                # generate this chunk FRESH (state=None) so the model re-energises
                # from the current style instead of continuing toward silence.
                reseed = low_streak >= self.DECAY_CHUNKS
                ci, state = self._gen_chunk(mrt, scene_style, params,
                                            None if reseed else state, np, cur_notes)
                raw_rms = float(np.sqrt(np.mean(ci ** 2)))
                low_streak = 0 if (reseed or raw_rms >= self.DECAY_RMS) else low_streak + 1
                if reseed:
                    cur_g = 1.0                          # restart the AGC cleanly
                    print("[MRT2] output faded — re-seeded state to re-energise")
                    if self._telemetry:
                        self._telemetry.event("reseed", reason="decay")

                ci = _emit(ci)
                with self._pb_lock:
                    self._cur_stream = np.concatenate([self._cur_stream, ci])
                    if self._cur_pos > TRIM:             # bound memory
                        drop = self._cur_pos - KEEP
                        self._cur_stream = self._cur_stream[drop:]
                        self._cur_pos -= drop
            else:
                time.sleep(0.03)                          # buffer healthy — idle

        stream.stop()
        stream.close()

    SPECTRUM_BANDS = 64   # log-spaced frequency bands for the live equalizer

    def output_stats(self) -> dict:
        """Level + spectral brightness + a binned magnitude spectrum of the audio
        currently playing. Called at ~20Hz; one cheap FFT on the last output block.
        """
        blk = self._out_block
        if blk is None or len(blk) < 2:
            return {"level": 0.0, "bright": 0.0, "spectrum": []}
        level = min(1.0, self._out_rms / 0.3)
        spec  = np.abs(np.fft.rfft(blk))
        freqs = np.fft.rfftfreq(len(blk), 1.0 / self.SAMPLE_RATE)
        total = float(spec.sum())
        centroid = float((freqs * spec).sum() / total) if total > 0 else 0.0
        # Normalise: musical content sits low (~1–3kHz); broadband noise pushes
        # the centroid high → close to 1.0. So a high reading flags "noise".
        bright = min(1.0, centroid / 6000.0)
        return {"level": round(level, 4), "bright": round(bright, 4),
                "spectrum": self._band_spectrum(spec, freqs)}

    def _band_spectrum(self, mag, freqs) -> list[float]:
        """Collapse the FFT magnitudes into SPECTRUM_BANDS log-spaced bands
        (≈40 Hz–16 kHz), normalised so the loudest band = 1.0. Returns a real
        per-frequency spectrum — taller bars = more energy at that pitch range."""
        n = self.SPECTRUM_BANDS
        fmax = min(16000.0, self.SAMPLE_RATE / 2.0)
        edges = np.logspace(np.log10(40.0), np.log10(fmax), n + 1)
        idx = np.searchsorted(freqs, edges)
        bands = np.zeros(n, dtype=np.float32)
        for i in range(n):
            a, b = idx[i], max(idx[i] + 1, idx[i + 1])
            seg = mag[a:b]
            if seg.size:
                bands[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        peak = float(bands.max())
        if peak > 1e-9:
            bands = bands / peak
        return [round(float(x), 3) for x in bands]

    def perf_stats(self) -> dict:
        """Snapshot of generation profiling — used by the `s` status command."""
        # A chunk is CHUNK_FRAMES frames; 25 frames = 1000ms of audio (40ms each).
        chunk_ms = self.CHUNK_FRAMES * 40.0
        return {
            "gen_ms_per_chunk": round(self._gen_avg_ms, 1),
            "underruns": self._underruns,
            "starves": self._starves,            # buffer-empty stutters (the real one)
            # Generation must be faster than the audio it produces, or it can't
            # keep up no matter the buffer size.
            "realtime_ok": self._gen_avg_ms < chunk_ms if self._gen_avg_ms else True,
        }

    def health(self) -> HealthStatus:
        """Liveness snapshot for the app/UI. `ok` goes False if the generation
        thread has faulted (model load, audio device, or a generation error),
        turning the old 'silent dead engine that still looks running' into a
        visible, diagnosable state. `last_chunk_age_s` is None until the first
        chunk is produced."""
        with self._lock:
            fault = self._fault
        age = None
        if self._last_chunk_time:
            age = round(time.monotonic() - self._last_chunk_time, 2)
        return {
            "ok": fault is None,
            "fault": fault,
            "last_chunk_age_s": age,
            "starves": self._starves,
            "underruns": self._underruns,
        }

    def update(self, params: MRTParams):
        """Update target parameters (thread-safe). Within a scene the voice
        shapes the audio: blend → filter brightness, chaos → volume."""
        with self._lock:
            self._current_params = params
        self._blend = params.prompt_blend   # single floats — atomic in callback
        self._chaos = params.chaos

    def generate_chunk(self):
        """Not used — audio loop runs in its own thread."""
        return None

    def stop(self):
        self._stop_event.set()


# ══════════════════════════════════════════════════════════════════════
#  Unified wrapper — picks the right controller automatically
# ══════════════════════════════════════════════════════════════════════

class MRTController:
    """
    Auto-selects MIDI or Python mode.
    Pass mode="midi" or mode="python" to force one.
    """

    def __init__(self, mode: str = "midi", *, morph_step: float = 0.30,
                 default_a: str | None = None, default_b: str | None = None,
                 default_key: str | None = None, telemetry=None,
                 enable_drums: bool = False):
        if mode == "python":
            self._impl = PythonMRTController(
                morph_step=morph_step, default_a=default_a, default_b=default_b,
                default_key=default_key, telemetry=telemetry,
                enable_drums=enable_drums,
            )
        else:
            self._impl = MIDIMRTController()
        self._mode = mode

    def start(self):
        self._impl.start()

    def update(self, params: MRTParams):
        self._impl.update(params)

    def set_prompts(self, prompt_a: str, prompt_b: str, key: str | None = None):
        """Only relevant in Python mode."""
        if isinstance(self._impl, PythonMRTController):
            self._impl.set_prompts(prompt_a, prompt_b, key)

    def set_session_key(self, key: str):
        """Only relevant in Python mode — switch the locked session key."""
        if isinstance(self._impl, PythonMRTController):
            self._impl.set_session_key(key)

    def perf_stats(self) -> dict | None:
        """Generation profiling — Python mode only."""
        if isinstance(self._impl, PythonMRTController):
            return self._impl.perf_stats()
        return None

    def output_stats(self) -> dict | None:
        """Output audio level + brightness — Python mode only."""
        if isinstance(self._impl, PythonMRTController):
            return self._impl.output_stats()
        return None

    def health(self) -> HealthStatus | None:
        """Engine liveness/fault snapshot — Python mode only."""
        if isinstance(self._impl, PythonMRTController):
            return self._impl.health()
        return None

    def hold_chord(self, notes: list[str], velocity: int = 80):
        """Only relevant in MIDI mode."""
        if isinstance(self._impl, MIDIMRTController):
            self._impl.hold_chord(notes, velocity)

    def generate_chunk(self):
        """Only relevant in Python mode."""
        if isinstance(self._impl, PythonMRTController):
            return self._impl.generate_chunk()
        return None

    def stop(self):
        self._impl.stop()
