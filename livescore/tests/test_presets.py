"""Tests for preset config integrity, including the drums wiring.

`enable_drums` must be on exactly for the presets that intend drums (low
threshold) and off for the intimate/ambient ones, so the drums config is no
longer silently dead.
"""

import pytest

from presets import PRESETS, get


@pytest.mark.unit
def test_drum_presets_enable_drums():
    for name in ("dnd", "stream", "fitness"):
        assert PRESETS[name].enable_drums is True, f"{name} should enable drums"


@pytest.mark.unit
def test_intimate_presets_keep_drums_off():
    for name in ("storytelling", "intimate", "meditation"):
        assert PRESETS[name].enable_drums is False, f"{name} should be drumless"


@pytest.mark.unit
def test_enabled_drums_have_a_reachable_threshold():
    # If a preset enables drums, its threshold must be crossable (< 1.0), else
    # the switch is on but drums can never actually fire.
    for name, preset in PRESETS.items():
        if preset.enable_drums:
            assert preset.drums_threshold < 1.0, f"{name}: drums on but threshold unreachable"


@pytest.mark.unit
def test_get_falls_back_to_storytelling():
    assert get("nonexistent").name == "storytelling"
