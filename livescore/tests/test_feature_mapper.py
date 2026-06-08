"""Unit tests for feature_mapper: the deterministic voice -> music mapping core.

These pin the *behavior* of the instrument: how loudness becomes density, how
brightness becomes the dark/warm blend, when drums fire, and how the running
auto-gain self-calibrates. Pure logic, no audio device or model needed.
"""

import pytest

from feature_mapper import RunningNormalizer, FeatureMapper
from voice_analyzer import VoiceFeatures


def make_features(*, energy=0.0, pitch=0.0, speech_rate=0.0, brightness=0.0,
                  is_silent=False) -> VoiceFeatures:
    f = VoiceFeatures()
    f.energy = energy
    f.pitch = pitch
    f.speech_rate = speech_rate
    f.brightness = brightness
    f.is_silent = is_silent
    return f


@pytest.mark.unit
class TestRunningNormalizer:
    def test_value_at_mean_maps_to_half(self):
        n = RunningNormalizer(init_mean=0.4, init_spread=0.15)
        assert n.normalize(0.4) == pytest.approx(0.5, abs=1e-6)

    def test_above_mean_maps_above_half(self):
        n = RunningNormalizer(init_mean=0.4, init_spread=0.15)
        assert n.normalize(0.7) > 0.5

    def test_below_mean_maps_below_half(self):
        n = RunningNormalizer(init_mean=0.4, init_spread=0.15)
        assert n.normalize(0.1) < 0.5

    def test_output_always_in_unit_range(self):
        n = RunningNormalizer(init_mean=0.4, init_spread=0.15)
        for x in (-10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 1e6):
            assert 0.0 <= n.normalize(x) <= 1.0

    def test_min_span_floor_prevents_jitter(self):
        # A near-monotone signal with tiny deviations must NOT be amplified into
        # full swings, because the span is floored at min_span.
        n = RunningNormalizer(init_mean=0.5, init_spread=0.0,
                              min_span=0.08, span_k=1.5)
        for d in (0.001, -0.001, 0.002, -0.002):
            assert abs(n.normalize(0.5 + d) - 0.5) < 0.1

    def test_baseline_recenters_on_sustained_shift(self):
        # After many high values the running mean rises, so the same high value
        # eventually reads closer to center: the auto-gain re-calibrates.
        n = RunningNormalizer(alpha=0.9, init_mean=0.4, init_spread=0.15)
        first = n.normalize(0.9)
        for _ in range(50):
            n.normalize(0.9)
        assert n.normalize(0.9) < first


@pytest.mark.unit
class TestFeatureMapper:
    def test_silence_drops_chaos_and_holds_blend(self):
        m = FeatureMapper(smoothing=0.0, adaptive=False)
        voiced = m.update(make_features(brightness=0.8, energy=0.5))
        out = m.update(make_features(is_silent=True))
        assert out.chaos == pytest.approx(0.05, abs=1e-6)
        assert out.drums_on is False
        # blend is held at its pre-silence value (not zeroed or reset)
        assert out.prompt_blend == pytest.approx(voiced.prompt_blend, abs=1e-6)
        assert voiced.prompt_blend > 0.0  # ...and it was a real, non-trivial value

    def test_energy_chaos_formula(self):
        m = FeatureMapper(smoothing=0.0, adaptive=False)
        out = m.update(make_features(energy=0.9, brightness=0.5))
        # matches _compute_target: chaos = 0.10 + energy * 0.75
        assert out.chaos == pytest.approx(0.10 + 0.9 * 0.75, abs=1e-6)

    def test_higher_energy_gives_more_chaos(self):
        # Separate instances so neither call's smoothing state leaks into the other.
        m_lo = FeatureMapper(smoothing=0.0, adaptive=False)
        m_hi = FeatureMapper(smoothing=0.0, adaptive=False)
        lo = m_lo.update(make_features(energy=0.1, brightness=0.5)).chaos
        hi = m_hi.update(make_features(energy=0.9, brightness=0.5)).chaos
        assert hi > lo

    def test_brightness_drives_blend(self):
        m = FeatureMapper(smoothing=0.0, adaptive=False)
        dark = m.update(make_features(brightness=0.1, energy=0.3)).prompt_blend
        bright = m.update(make_features(brightness=0.9, energy=0.3)).prompt_blend
        assert bright > dark

    def test_blend_clamped_to_unit_range(self):
        m = FeatureMapper(smoothing=0.0, adaptive=False)
        out = m.update(make_features(brightness=1.0, speech_rate=1.0, energy=0.3))
        assert 0.0 <= out.prompt_blend <= 1.0
        assert out.prompt_blend == pytest.approx(1.0)  # 1.0 + 1.0*0.2 -> clamp

    def test_drums_fire_above_threshold(self):
        m = FeatureMapper(smoothing=0.0, adaptive=False, drums_threshold=0.5)
        assert m.update(make_features(energy=0.9, brightness=0.5)).drums_on is True
        assert m.update(make_features(energy=0.1, brightness=0.5)).drums_on is False

    def test_smoothing_glides_toward_target(self):
        m = FeatureMapper(smoothing=0.5, adaptive=False)
        target = 0.10 + 1.0 * 0.75  # chaos target for energy=1.0
        first = m.update(make_features(energy=1.0, brightness=0.5)).chaos
        assert first < target  # did not snap straight to target
        last = first
        for _ in range(20):
            last = m.update(make_features(energy=1.0, brightness=0.5)).chaos
        assert last == pytest.approx(target, abs=1e-3)  # converged

    def test_adaptive_blend_responds_and_stays_in_range(self):
        # Exercises the adaptive RunningNormalizer path (the production default),
        # which the other tests bypass with adaptive=False.
        m = FeatureMapper(smoothing=0.0, adaptive=True)
        for b in (0.2, 0.8, 0.3, 0.9, 0.1):
            assert 0.0 <= m.update(make_features(brightness=b, energy=0.3)).prompt_blend <= 1.0
        # After settling on a mid baseline, a bright value reads warmer than a dim one.
        m2 = FeatureMapper(smoothing=0.0, adaptive=True)
        for _ in range(10):
            m2.update(make_features(brightness=0.5, energy=0.3))
        dim = m2.update(make_features(brightness=0.2, energy=0.3)).prompt_blend
        bright = m2.update(make_features(brightness=0.9, energy=0.3)).prompt_blend
        assert bright > dim
