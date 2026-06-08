"""Unit tests for paths: the keepsake -> song / arrangement filename derivation.

These transforms preserve a session's timestamp while routing each artifact into
the correct outputs/ subfolder. A regression here silently scatters or
overwrites generated files.
"""

import os

import pytest

import paths


@pytest.mark.unit
class TestSongFor:
    def test_swaps_prefix_and_extension(self):
        out = paths.song_for("keepsake-20260606-0142.json")
        assert os.path.basename(out) == "song-20260606-0142.wav"
        assert os.path.dirname(out) == paths.SONGS

    def test_accepts_a_full_path(self):
        out = paths.song_for("/tmp/whatever/keepsake-99.json")
        assert os.path.basename(out) == "song-99.wav"


@pytest.mark.unit
class TestArrangementFor:
    def test_swaps_prefix(self):
        out = paths.arrangement_for("keepsake-20260606-0142.json")
        assert os.path.basename(out) == "arrangement-20260606-0142.json"
        assert os.path.dirname(out) == paths.ARRANGEMENTS

    def test_falls_back_when_no_prefix_to_swap(self):
        out = paths.arrangement_for("custom.json")
        assert os.path.basename(out) == "custom.arrangement.json"


@pytest.mark.unit
class TestKeepsakePath:
    def test_strips_directory_to_keepsakes_folder(self):
        out = paths.keepsake_path("/some/other/dir/keepsake-1.json")
        assert os.path.dirname(out) == paths.KEEPSAKES
        assert os.path.basename(out) == "keepsake-1.json"
