"""Unit tests for keepsake pure helpers: scene scheduling and the offline DSP.

These pin how a captured scene timeline becomes a per-second render schedule
(clamped to a musical range) and the loudness/limiter math used in the render.
"""

import numpy as np
import pytest

from keepsake import (
    _scene_schedule, _tag, _build_notes, _soft_limit, _normalize,
    SCENE_MIN_SEC, SCENE_MAX_SEC,
)


@pytest.mark.unit
class TestSceneSchedule:
    def test_durations_clamped_to_musical_range(self):
        scenes = [{"t": 0.0}, {"t": 2.0}, {"t": 100.0}]
        _schedule, durations = _scene_schedule(scenes)
        # scene 0 gap = 2s -> clamped up to SCENE_MIN; scene 1 gap = 98s ->
        # clamped down to SCENE_MAX; last scene defaults to SCENE_MAX.
        assert durations == pytest.approx([SCENE_MIN_SEC, SCENE_MAX_SEC, SCENE_MAX_SEC])

    def test_schedule_has_one_index_per_second(self):
        scenes = [{"t": 0.0}, {"t": 2.0}, {"t": 100.0}]
        schedule, _durations = _scene_schedule(scenes)
        assert len(schedule) == int(SCENE_MIN_SEC + SCENE_MAX_SEC + SCENE_MAX_SEC)
        assert schedule.count(0) == int(SCENE_MIN_SEC)
        assert schedule.count(2) == int(SCENE_MAX_SEC)

    def test_single_scene_uses_max_duration(self):
        schedule, durations = _scene_schedule([{"t": 5.0}])
        assert durations == pytest.approx([SCENE_MAX_SEC])
        assert len(schedule) == int(SCENE_MAX_SEC)


@pytest.mark.unit
class TestTag:
    def test_blank_falls_back_to_warm_default(self):
        assert _tag("") == "warm gentle piano"
        assert _tag(None) == "warm gentle piano"

    def test_trims_whitespace(self):
        assert _tag("  Felt Piano ") == "Felt Piano"


@pytest.mark.unit
class TestBuildNotes:
    def test_c_major_and_a_minor_match(self):
        assert _build_notes("C major", 128) == _build_notes("A minor", 128)

    def test_none_key_returns_none(self):
        assert _build_notes(None, 128) is None

    def test_out_of_key_pitch_silenced_in_range(self):
        notes = _build_notes("C major", 128)
        assert notes[49] == 0    # C#3 not in C major, in range
        assert notes[48] == -1   # C3 in key, free


@pytest.mark.unit
class TestDsp:
    def test_soft_limit_tames_peaks_below_unity(self):
        x = np.array([0.5, 0.9, 2.0, -2.0], dtype=np.float32)
        out = _soft_limit(x, th=0.85)
        assert out[0] == pytest.approx(0.5)        # below knee, untouched
        assert np.all(np.abs(out) < 1.0 + 1e-6)    # nothing exceeds unity
        assert abs(out[2]) < 2.0                   # the 2.0 peak was compressed

    def test_normalize_hits_target_rms(self):
        buf = np.full(1000, 0.5, dtype=np.float32)
        out = _normalize(buf, target_rms=0.16)
        assert float(np.sqrt(np.mean(out ** 2))) == pytest.approx(0.16, abs=1e-4)

    def test_normalize_leaves_silence_untouched(self):
        buf = np.zeros(1000, dtype=np.float32)
        out = _normalize(buf, target_rms=0.16)
        assert np.all(out == 0.0)  # guard against divide-by-zero blowup
