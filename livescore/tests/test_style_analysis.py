"""
Tests for analyze_style_space — the pure analysis + render pipeline.

Uses synthetic embeddings (no MRT2), so PCA/t-SNE/k-means/heatmap and the RVQ
collision logic are all exercised headlessly. matplotlib runs under Agg.
"""

import json

import numpy as np
import pytest

import analyze_style_space as asp


def _synthetic(n_per=8):
    """A few clearly-separated Gaussian blobs labelled by family."""
    rng = np.random.RandomState(0)
    fams = ["keys", "strings", "synth", "genre", "mood"]
    texts, families, embs = [], [], []
    for fi, fam in enumerate(fams):
        center = rng.randn(768) * 5 + fi * 10
        for j in range(n_per):
            embs.append((center + rng.randn(768)).astype(np.float32))
            texts.append(f"{fam}_{j}")
            families.append(fam)
    embs = np.stack(embs)
    # Tokens: same blob → same tokens for two entries so a collision exists.
    toks = (np.arange(len(texts)) // 2)[:, None].repeat(4, axis=1)
    return {"texts": np.array(texts, dtype=object),
            "families": np.array(families, dtype=object),
            "embeddings": embs, "tokens": toks}


@pytest.mark.unit
def test_rvq_collisions_groups_identical_tokens():
    data = _synthetic()
    groups = asp.rvq_collisions(data["tokens"], data["texts"])
    # We forced pairs (//2) to share tokens → every group has exactly 2 members.
    assert groups and all(len(g) == 2 for g in groups)


@pytest.mark.unit
def test_render_all_writes_pngs(tmp_path, monkeypatch):
    # Redirect outputs into a temp dir.
    monkeypatch.setattr(asp.paths, "analysis_path",
                        lambda name: str(tmp_path / name))
    data = _synthetic()
    outs = asp.render_all(data)
    assert len(outs) == 4
    for o in outs:
        assert (tmp_path / o.split("/")[-1]).exists()
        assert (tmp_path / o.split("/")[-1]).stat().st_size > 0


@pytest.mark.unit
def test_cache_roundtrip_is_pickle_free(tmp_path, monkeypatch):
    npz = str(tmp_path / "e.npz")
    js = str(tmp_path / "l.json")
    monkeypatch.setattr(asp, "CACHE_NPZ", npz)
    monkeypatch.setattr(asp, "CACHE_JSON", js)
    data = _synthetic(n_per=3)
    np.savez(npz, embeddings=data["embeddings"], tokens=data["tokens"])
    with open(js, "w") as f:
        json.dump({"texts": list(data["texts"]), "families": list(data["families"])}, f)
    loaded = asp.load_cache()
    assert loaded is not None
    assert np.allclose(loaded["embeddings"], data["embeddings"])
    assert list(loaded["texts"]) == list(data["texts"])


@pytest.mark.unit
def test_load_cache_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(asp, "CACHE_NPZ", str(tmp_path / "nope.npz"))
    monkeypatch.setattr(asp, "CACHE_JSON", str(tmp_path / "nope.json"))
    assert asp.load_cache() is None
