"""Unit tests for the LLM director's pure parsing helpers.

These guard the boundary where Claude's free-form text becomes structured style
direction: stripping vocal instruments the synth can't honor, and extracting
JSON from prose/markdown. Both must never raise on the happy path and must fail
predictably on garbage.
"""

import json

import pytest

from llm_style_director import _clean_style, _extract_json


@pytest.mark.unit
class TestCleanStyle:
    def test_strips_trailing_period_and_whitespace(self):
        assert _clean_style("  Warm Felt Piano.  ") == "Warm Felt Piano"

    def test_removes_vocal_segments(self):
        assert _clean_style("warm piano, female vocals, gentle keys") == \
            "warm piano, gentle keys"

    def test_collapses_whitespace_around_commas(self):
        assert _clean_style("warm piano ,  soft keys") == "warm piano, soft keys"

    def test_falls_back_to_original_when_all_segments_filtered(self):
        # If every segment is a vocal term, return the cleaned original rather
        # than an empty string (so the synth always gets *some* prompt).
        assert _clean_style("vocals") == "vocals"


@pytest.mark.unit
class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"keep": true}') == {"keep": True}

    def test_strips_markdown_json_fence(self):
        assert _extract_json('```json\n{"a": "x"}\n```') == {"a": "x"}

    def test_extracts_json_embedded_in_prose(self):
        assert _extract_json('Sure, here: {"a": "x", "b": "y"} done.') == \
            {"a": "x", "b": "y"}

    def test_raises_on_unparseable(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("there is no json here at all")
