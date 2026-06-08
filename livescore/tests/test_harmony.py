"""Unit tests for harmony.py: key -> note-mask logic and note-name parsing.

These pure functions are shared by the live controller and the keepsake
renderer, so a single tested implementation keeps them from drifting. The note
parser had latent bugs on multi-digit octaves.
"""

import pytest

import harmony


@pytest.mark.unit
class TestBuildNotes:
    def test_c_major_and_a_minor_share_a_mask(self):
        # C major and A natural minor are the same pitch-class set, so their
        # note masks must be identical.
        assert harmony.build_notes("C major", 128) == harmony.build_notes("A minor", 128)

    def test_in_key_pitch_left_free(self):
        assert harmony.build_notes("C major", 128)[48] == -1  # C3 (pc 0) in key

    def test_out_of_key_pitch_turned_off_in_range(self):
        assert harmony.build_notes("C major", 128)[49] == 0   # C#3 not in key, in range

    def test_out_of_range_pitch_left_free(self):
        notes = harmony.build_notes("C major", 128)
        assert notes[0] == -1    # below NOTE_LO -> never constrained
        assert notes[127] == -1  # above NOTE_HI -> never constrained

    def test_length_matches_request(self):
        assert len(harmony.build_notes("C major", 96)) == 96

    def test_small_num_notes_yields_all_free(self):
        # num_notes below NOTE_LO means no pitch is in the constrained range,
        # so the mask is all -1 (free) regardless of key.
        assert harmony.build_notes("C major", 12) == [-1] * 12

    def test_unknown_key_returns_none(self):
        assert harmony.build_notes("H minor", 128) is None

    def test_none_key_returns_none(self):
        assert harmony.build_notes(None, 128) is None

    def test_sharp_key_builds_correct_mask(self):
        # F# minor pitch classes: {6, 8, 9, 11, 1, 2, 4}. Spot-check the sharp
        # offset on both in- and out-of-scale pitches.
        notes = harmony.build_notes("F# minor", 128)
        assert notes is not None
        assert notes[54] == -1  # F#3 (pc 6) in scale -> free
        assert notes[56] == -1  # G#3 (pc 8) in scale -> free
        assert notes[55] == 0   # G3  (pc 7) out of scale -> silenced


@pytest.mark.unit
class TestNoteToMidi:
    @pytest.mark.parametrize("name,expected", [
        ("C3", 48), ("A4", 69), ("C2", 36), ("C6", 84),
        ("F#4", 66), ("Db3", 49), ("G2", 43), ("B2", 47), ("C4", 60),
    ])
    def test_known_notes(self, name, expected):
        assert harmony.note_to_midi(name) == expected

    def test_case_insensitive(self):
        assert harmony.note_to_midi("c3") == 48

    def test_multi_digit_and_negative_octave_edges(self):
        # Multi-digit and negative octaves now parse correctly, at the exact
        # edges of the valid MIDI range.
        assert harmony.note_to_midi("G9") == 127   # highest valid
        assert harmony.note_to_midi("C-1") == 0    # lowest valid

    def test_out_of_midi_range_raises(self):
        # Regression: "C10" used to be misparsed and "C#10" crashed with int('#').
        # Both now parse cleanly and raise a meaningful out-of-range error.
        with pytest.raises(ValueError):
            harmony.note_to_midi("C10")   # would be 132, out of range
        with pytest.raises(ValueError):
            harmony.note_to_midi("C#10")  # would be 133, out of range

    def test_unparseable_name_raises_valueerror(self):
        with pytest.raises(ValueError):
            harmony.note_to_midi("not-a-note")
