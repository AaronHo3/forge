"""Unit tests for keepsake pure helpers: scene scheduling and the offline DSP.

These pin how a captured scene timeline becomes a per-second render schedule
(clamped to a musical range) and the loudness/limiter math used in the render.
"""

import pytest

from keepsake import _scene_schedule, _tag, SCENE_MIN_SEC, SCENE_MAX_SEC


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
