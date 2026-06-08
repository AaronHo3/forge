"""
style_space.py — how the voice navigates MRT2's 768-d style space.

MRT2 conditions on ONE style embedding per generate() call. The original engine
computed it by linearly interpolating TWO poles (A↔B). This module generalises
that into three interchangeable ways to build that one vector, all over the same
space:

  • simplex blend  — mix N named anchors with weights that sum to 1, so the voice
                     navigates a REGION (triangle / tetrahedron / ...) instead of
                     a single A↔B line.
  • axis control   — move along named musical directions (valence, arousal,
                     brightness, density). Each axis is a precomputed unit vector
                     embed(positive) − embed(negative); a coord in [-1, 1] slides
                     along it. This maps voice features to MEANINGFUL directions
                     (energy→arousal, pitch→valence) rather than to opaque poles.
  • lerp           — the 2-pole special case, kept so existing callers are unchanged.

The embedder (text → 768-d vector) is INJECTED, so this whole module is unit-
tested with a deterministic fake and never needs to load MRT2.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from log import get_logger

log = get_logger("style")

# A style embedding is just a 1-D float vector (768-d for MusicCoCa).
Embedding = np.ndarray
Embedder = Callable[[str], Embedding]


# Canonical musical axes: (positive prompt, negative prompt). The direction is
# embed(pos) − embed(neg), normalised. Coord +1 = fully positive, −1 = negative.
DEFAULT_AXES: dict[str, tuple[str, str]] = {
    "valence":    ("bright joyful uplifting major key music",
                   "dark sad heavy mournful minor key music"),
    "arousal":    ("intense driving energetic loud fast music",
                   "calm gentle still quiet slow music"),
    "brightness": ("bright airy shimmering high register music",
                   "warm dark low muffled deep register music"),
    "density":    ("dense full busy layered rich arrangement",
                   "sparse minimal empty single-instrument music"),
}


def _unit(v: Embedding) -> Embedding:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class StyleSpace:
    """Builds MRT2 style embeddings from blends, axes, or poles. Caches embeds."""

    def __init__(self, embedder: Embedder,
                 axes: dict[str, tuple[str, str]] | None = None):
        self._embed_fn = embedder
        self._axes_spec = dict(axes if axes is not None else DEFAULT_AXES)
        self._cache: dict[str, Embedding] = {}
        self._axis_dirs: dict[str, Embedding] = {}   # lazy, cached

    # ── embedding (cached) ────────────────────────────────────────────────────
    def embed(self, text: str) -> Embedding:
        """Embed text once and memoise (MRT2 embeds are not free)."""
        e = self._cache.get(text)
        if e is None:
            e = np.asarray(self._embed_fn(text), dtype=np.float32)
            self._cache[text] = e
        return e

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Stack embeddings for a list of texts → (N, dim)."""
        return np.stack([self.embed(t) for t in texts])

    # ── simplex blend (N poles) ───────────────────────────────────────────────
    def blend(self, weights: dict[str, float]) -> Embedding:
        """Convex combination of named anchors. Weights are normalised to sum to
        1 (a point inside the simplex of the given anchors); non-positive total
        falls back to a uniform mix. Empty → ValueError."""
        if not weights:
            raise ValueError("blend needs at least one anchor")
        names = list(weights)
        w = np.array([max(0.0, float(weights[n])) for n in names], dtype=np.float32)
        total = float(w.sum())
        w = (w / total) if total > 1e-9 else np.full(len(names), 1.0 / len(names),
                                                      dtype=np.float32)
        embs = self.embed_many(names)            # (N, dim)
        return (w[:, None] * embs).sum(axis=0)

    def lerp(self, a: str, b: str, t: float) -> Embedding:
        """2-pole interpolation a↔b at t∈[0,1] — the original A/B behaviour,
        expressed as a 2-point simplex blend."""
        t = float(np.clip(t, 0.0, 1.0))
        return self.blend({a: 1.0 - t, b: t})

    # ── axis control (valence / arousal / ...) ────────────────────────────────
    def axis_dir(self, name: str) -> Embedding:
        """Unit direction for a named axis = normalise(embed(pos) − embed(neg))."""
        d = self._axis_dirs.get(name)
        if d is None:
            pos, neg = self._axes_spec[name]
            d = _unit(self.embed(pos) - self.embed(neg))
            if not d.any():
                # pos/neg embed identically → zero direction. from_axes then adds
                # nothing (safe), but warn so a dead axis isn't a silent mystery.
                log.warning(f"[style] axis '{name}' is a zero direction — its "
                            f"pos/neg prompts embed identically; modulation is a no-op")
            self._axis_dirs[name] = d
        return d

    def axis_names(self) -> list[str]:
        return list(self._axes_spec)

    def from_axes(self, base: Embedding, coords: dict[str, float],
                  scale: float = 1.0) -> Embedding:
        """Push a base embedding along named axes. `coords[name]` in [-1, 1]
        slides `scale * coord * ||base||` along that axis' unit direction, so the
        modulation is proportional to the base magnitude (keeps it on-scale with
        MRT2's ~28-norm embeddings). Unknown axis names are ignored."""
        out = np.asarray(base, dtype=np.float32).copy()
        # MRT2 embeddings are ~28-norm; `or 1.0` only guards a degenerate zero base
        # (never produced by the real embedder) so we don't divide reach by zero.
        mag = float(np.linalg.norm(base)) or 1.0
        for name, c in coords.items():
            if name not in self._axes_spec:
                continue
            c = float(np.clip(c, -1.0, 1.0))
            out = out + (scale * c * mag) * self.axis_dir(name)
        return out
