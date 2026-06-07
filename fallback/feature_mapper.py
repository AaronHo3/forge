"""
feature_mapper.py
-----------------
Translates VoiceFeatures into MRT2 control parameters.

This is the creative layer — the choices here define how the instrument
feels. Tweak the mappings to change the musical personality.

Output parameters:
  - prompt_blend  : 0.0 = Prompt A (dark/tense), 1.0 = Prompt B (bright/warm)
  - chaos         : 0.0 = sparse/simple, 1.0 = dense/complex
  - drums_on      : bool, whether to include percussion

Design choices (change these to change the instrument's feel):
  - Energy drives chaos:   loud/intense narration → complex, layered texture
  - Brightness drives blend: bright/excited voice → warmer, lifted music
  - Silence pulls chaos down: pauses give the music room to breathe
  - Speech rate adds energy to blend: fast delivery nudges toward brighter end
"""

from dataclasses import dataclass
from voice_analyzer import VoiceFeatures


class RunningNormalizer:
    """Auto-gain for a single feature.

    Tracks a running mean and a running spread (mean absolute deviation) of
    the values it sees, then maps each new value to 0–1 *relative to that
    baseline*: equal to the running mean → 0.5, one spread above → ~1.0,
    one spread below → ~0.0.

    The effect: whatever the speaker's natural range is, it gets stretched to
    fill the full 0–1 control range — so the music actually reaches both poles
    instead of smearing in the middle. Self-calibrates to mic and room.

    Tunables:
      alpha     — how slowly the baseline adapts (0.995 ≈ ~10 s memory at 20 Hz).
                  Higher = slower to re-center, holds sustained contrast longer.
      span_k    — how many spreads map to a full swing. Lower = more sensitive.
      min_span  — floor on the spread so a monotone voice doesn't get its tiny
                  variations amplified into jittery full swings.
    """

    def __init__(self, alpha=0.995, span_k=1.5, min_span=0.08,
                 init_mean=0.4, init_spread=0.15):
        self.alpha    = alpha
        self.span_k   = span_k
        self.min_span = min_span
        self.mean     = init_mean
        self.spread   = init_spread

    def normalize(self, x: float) -> float:
        # Update running baseline (only call this on real, non-silent values).
        self.mean   = self.alpha * self.mean   + (1 - self.alpha) * x
        self.spread = self.alpha * self.spread + (1 - self.alpha) * abs(x - self.mean)
        span = max(self.span_k * self.spread, self.min_span)
        return max(0.0, min(1.0, 0.5 + (x - self.mean) / (2.0 * span)))


@dataclass
class MRTParams:
    """Parameters sent to MRT2 each update cycle."""
    prompt_blend: float = 0.5   # 0.0–1.0  (A ↔ B)
    chaos: float = 0.3          # 0.0–1.0
    drums_on: bool = False

    def __repr__(self):
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))
        side = "B (bright)" if self.prompt_blend > 0.5 else "A (dark) "
        return (
            f"blend     {bar(self.prompt_blend)}  {self.prompt_blend:.2f}  → {side}\n"
            f"chaos     {bar(self.chaos)}  {self.chaos:.2f}\n"
            f"drums     {'on' if self.drums_on else 'off'}"
        )


class FeatureMapper:
    """
    Stateful mapper with exponential smoothing so parameters glide
    rather than jumping — avoids jarring cuts in the generated music.

    Smoothing alpha:
        0.0 = completely reactive (follows every syllable)
        0.9 = very slow, almost ignores fast changes
    Recommended: 0.6–0.75 for a natural feel.
    """

    def __init__(self, smoothing: float = 0.70, adaptive: bool = True,
                 drums_threshold: float = 0.5):
        self._alpha = smoothing
        self._adaptive = adaptive
        self._drums_threshold = drums_threshold
        self._params = MRTParams(prompt_blend=0.0, chaos=0.10, drums_on=False)
        # Auto-gain for brightness → blend. Stretches the speaker's natural
        # brightness band to the full pole-to-pole range.
        self._bright_norm = RunningNormalizer()
        # Auto-gain for energy → drums. Lets drums fire on emphasis RELATIVE to
        # the speaker's own loudness (no shouting, robust to mic/room). Initialised
        # on the SPEECH energy scale (~0.05 RMS), not the 0–1 brightness scale, so
        # it's roughly calibrated from the first words instead of needing a minute.
        self._energy_norm = RunningNormalizer(init_mean=0.06, init_spread=0.04,
                                              min_span=0.02, span_k=1.2)

    # ── Public API ────────────────────────────────────────────────────

    def update(self, features: VoiceFeatures) -> MRTParams:
        """Feed in the latest VoiceFeatures; receive smoothed MRTParams."""
        target = self._compute_target(features)
        self._params = self._smooth(self._params, target)
        return self._params

    # ── Internals ─────────────────────────────────────────────────────

    def _compute_target(self, f: VoiceFeatures) -> MRTParams:
        p = MRTParams()

        if f.is_silent:
            # During a pause: hold blend where it is but drop chaos so
            # the music opens up and gives the silence space to breathe.
            p.prompt_blend = self._params.prompt_blend
            p.chaos = 0.05
            p.drums_on = False
            return p

        # ── Chaos: driven by energy ────────────────────────────────
        # Quiet, reflective narration → sparse texture (chaos 0.1–0.3)
        # Loud, dramatic narration   → dense texture (chaos 0.5–0.85)
        p.chaos = 0.10 + f.energy * 0.75

        # ── Blend: driven by brightness + a little speech rate ─────
        # Bright, excited voice → warmer, lifted prompt (B)
        # Dark, slow voice      → tense, minor prompt (A)
        # Raw brightness only spans ~0.24–0.70 for speech, so without
        # auto-gain the blend can never reach either pole. The normalizer
        # stretches the speaker's own range to the full 0–1 swing.
        if self._adaptive:
            bright = self._bright_norm.normalize(f.brightness)
        else:
            bright = f.brightness
        p.prompt_blend = bright + f.speech_rate * 0.20
        p.prompt_blend = max(0.0, min(1.0, p.prompt_blend))

        # ── Drums: kick in on emphasis, RELATIVE to the speaker's own range ──
        # Energy is normalized (like brightness) so the threshold is a 0–1 knob:
        # 0.5 ≈ your average loudness, higher = needs a bigger peak. Fires without
        # shouting and is robust to mic/room. Preset-tunable: low for fitness
        # (drums easily), high for storytelling, >1 to disable entirely.
        energy_rel = self._energy_norm.normalize(f.energy) if self._adaptive else f.energy
        p.drums_on = energy_rel > self._drums_threshold

        return p

    def _smooth(self, current: MRTParams, target: MRTParams) -> MRTParams:
        """Exponential moving average across all float parameters."""
        a = self._alpha
        s = MRTParams()
        s.prompt_blend = a * current.prompt_blend + (1 - a) * target.prompt_blend
        s.chaos        = a * current.chaos        + (1 - a) * target.chaos
        s.drums_on     = target.drums_on  # drums switch immediately — no smoothing
        return s


# ------------------------------------------------------------------
# Quick test — run this file directly with synthetic features
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time, math

    mapper = FeatureMapper(smoothing=0.70)

    print("Simulating a story arc: quiet opening → dramatic climax → silent pause\n")
    for t in range(80):
        f = VoiceFeatures()
        # Sine wave energy: rises to climax around t=40, then fades
        progress = t / 79.0
        f.energy = float(0.1 + 0.8 * math.sin(math.pi * progress))
        f.brightness = float(progress * 0.8)
        f.speech_rate = float(0.3 + progress * 0.5)
        f.is_silent = (t > 65)

        params = mapper.update(f)
        print(f"\033[2J\033[H── Score the Story: Feature Mapper (t={t:02d}) ──\n")
        print("Voice features:")
        print(f)
        print("\nMRT2 parameters:")
        print(params)
        time.sleep(0.08)
