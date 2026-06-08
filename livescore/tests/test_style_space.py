"""
Unit tests for style_space.StyleSpace — the N-pole / axis steering engine.

A deterministic fake embedder (text → seeded vector) stands in for MRT2, so the
blend/axis math is fully tested with no model load.
"""

import hashlib

import numpy as np
import pytest

from style_space import StyleSpace, DEFAULT_AXES

DIM = 768


def fake_embedder(text: str) -> np.ndarray:
    """Deterministic per-text vector, scaled to MRT2-like ~28 norm."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v) * 28.0


@pytest.fixture
def space():
    return StyleSpace(fake_embedder)


@pytest.mark.unit
def test_embed_is_cached(space):
    calls = {"n": 0}

    def counting(text):
        calls["n"] += 1
        return fake_embedder(text)

    s = StyleSpace(counting)
    s.embed("piano"); s.embed("piano"); s.embed("piano")
    assert calls["n"] == 1, "embed must memoise per unique text"


@pytest.mark.unit
def test_single_anchor_blend_returns_that_embedding(space):
    assert np.allclose(space.blend({"warm piano": 1.0}), space.embed("warm piano"))


@pytest.mark.unit
def test_blend_weights_are_normalised(space):
    # Unnormalised weights must equal the same convex combination normalised.
    a, b = space.embed("a"), space.embed("b")
    got = space.blend({"a": 3.0, "b": 1.0})
    expect = 0.75 * a + 0.25 * b
    assert np.allclose(got, expect, atol=1e-4)


@pytest.mark.unit
def test_blend_is_convex_combination_of_three(space):
    embs = {k: space.embed(k) for k in ("x", "y", "z")}
    got = space.blend({"x": 1.0, "y": 1.0, "z": 2.0})
    expect = 0.25 * embs["x"] + 0.25 * embs["y"] + 0.5 * embs["z"]
    assert np.allclose(got, expect, atol=1e-4)


@pytest.mark.unit
def test_nonpositive_total_falls_back_to_uniform(space):
    got = space.blend({"a": 0.0, "b": 0.0})
    expect = 0.5 * space.embed("a") + 0.5 * space.embed("b")
    assert np.allclose(got, expect, atol=1e-4)


@pytest.mark.unit
def test_empty_blend_raises(space):
    with pytest.raises(ValueError):
        space.blend({})


@pytest.mark.unit
def test_lerp_endpoints_and_midpoint(space):
    a, b = space.embed("a"), space.embed("b")
    assert np.allclose(space.lerp("a", "b", 0.0), a, atol=1e-4)
    assert np.allclose(space.lerp("a", "b", 1.0), b, atol=1e-4)
    assert np.allclose(space.lerp("a", "b", 0.5), 0.5 * a + 0.5 * b, atol=1e-4)


@pytest.mark.unit
def test_lerp_clamps_out_of_range(space):
    assert np.allclose(space.lerp("a", "b", 5.0), space.embed("b"), atol=1e-4)
    assert np.allclose(space.lerp("a", "b", -2.0), space.embed("a"), atol=1e-4)


@pytest.mark.unit
def test_axis_dir_is_unit_and_points_pos_minus_neg(space):
    d = space.axis_dir("valence")
    assert abs(np.linalg.norm(d) - 1.0) < 1e-5
    pos, neg = DEFAULT_AXES["valence"]
    raw = space.embed(pos) - space.embed(neg)
    assert np.dot(d, raw) > 0   # same direction


@pytest.mark.unit
def test_from_axes_moves_along_direction(space):
    base = space.embed("neutral pad")
    up = space.from_axes(base, {"arousal": 1.0})
    down = space.from_axes(base, {"arousal": -1.0})
    d = space.axis_dir("arousal")
    # Projection onto the axis must increase for +1 and decrease for −1.
    assert np.dot(up - base, d) > 0
    assert np.dot(down - base, d) < 0
    # A zero coord is a no-op.
    assert np.allclose(space.from_axes(base, {"arousal": 0.0}), base, atol=1e-5)


@pytest.mark.unit
def test_from_axes_ignores_unknown_axis(space):
    base = space.embed("pad")
    assert np.allclose(space.from_axes(base, {"nonsense": 1.0}), base)


@pytest.mark.unit
def test_zero_axis_direction_is_safe_noop():
    # If pos and neg embed identically, the axis direction is zero — from_axes
    # must then add nothing (no crash, no NaN), just return the base.
    const = StyleSpace(lambda text: np.full(8, 3.0, dtype=np.float32))
    base = const.embed("anything")
    out = const.from_axes(base, {"valence": 1.0})
    assert np.allclose(out, base)
    assert not np.isnan(out).any()
