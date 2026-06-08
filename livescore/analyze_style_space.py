"""
analyze_style_space.py — embed MRT2's whole palette and SEE its style space.

What it does:
  1. Embeds every phrase in palette.all_styles() with the real MRT2 MusicCoCa
     encoder (the same embed_style the live engine uses), and tokenizes each via
     MRT2's RVQ quantizer (the conditioning the model actually receives).
  2. Caches everything to outputs/analysis/style_embeddings.npz so re-rendering
     is instant and needs no model.
  3. Renders, into outputs/analysis/:
       • style_pca.png        — 2-D PCA map, coloured by instrument family
       • style_tsne.png       — 2-D t-SNE map (local neighbourhoods)
       • style_clusters.png   — k-means clusters over the space (what groups
                                together / what MRT2 treats as similar)
       • style_similarity.png — cosine-similarity heatmap, family-ordered
     and prints RVQ "collisions" — palette entries that tokenize identically, i.e.
     styles MRT2 cannot tell apart (the practical edge of the reachable palette).

Usage:
  python3 analyze_style_space.py            # embed (once) + render everything
  python3 analyze_style_space.py --reviz    # re-render from the cache, no model
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

import palette
import paths

# Numeric arrays go in .npz; the text/family labels go in a JSON sidecar. This
# keeps loading pickle-free (np.load needs allow_pickle=True for object/string
# arrays, which we avoid on principle even for our own first-party cache).
CACHE_NPZ = paths.analysis_path("style_embeddings.npz")
CACHE_JSON = paths.analysis_path("style_labels.json")

# A stable, readable colour per family.
_FAMILY_COLORS = {
    "keys": "#f85149", "guitar": "#d29922", "strings": "#3fb950",
    "woodwind": "#58a6ff", "brass": "#bc8cff", "mallet": "#ff7b72",
    "plucked_world": "#39c5cf", "synth": "#db61a2", "bass": "#a371f7",
    "drums_perc": "#e3b341", "ambient_texture": "#6e7681", "voice_like": "#ec6547",
    "genre": "#2f81f7", "mood": "#f0883e",
}


# ── 1. Embedding (needs the model) ────────────────────────────────────────────
def embed_all() -> dict:
    """Embed + tokenize the whole palette with real MRT2. Heavy: loads the model."""
    from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
    print("Loading mrt2_base ...", flush=True)
    mrt = MagentaRT2SystemMlxfn(size="mrt2_base")

    styles = palette.all_styles()
    texts = [t for t, _ in styles]
    families = [f for _, f in styles]
    print(f"Embedding {len(texts)} styles ...", flush=True)

    # tokenize() lives on a private sub-model; guard so a magenta-rt API change
    # degrades to "no RVQ tokens" (collisions report skipped) instead of crashing
    # at the start of an otherwise expensive embedding run.
    style_model = getattr(mrt, "_style_model", None)
    can_tokenize = hasattr(style_model, "tokenize")
    if not can_tokenize:
        print("  (note: mrt._style_model.tokenize unavailable — skipping RVQ tokens)")

    embs, toks = [], []
    for i, t in enumerate(texts):
        e = np.asarray(mrt.embed_style(t), dtype=np.float32)
        embs.append(e)
        if can_tokenize:
            toks.append(np.asarray(style_model.tokenize(e)).flatten())
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)

    data = {
        "texts": np.array(texts, dtype=object),
        "families": np.array(families, dtype=object),
        "embeddings": np.stack(embs),
        # 0-width when tokenize was unavailable → collisions report is skipped.
        "tokens": np.stack(toks) if toks else np.zeros((len(texts), 0), dtype=np.int32),
    }
    np.savez(CACHE_NPZ, embeddings=data["embeddings"], tokens=data["tokens"])
    with open(CACHE_JSON, "w") as f:
        json.dump({"texts": texts, "families": families}, f)
    print(f"cached → {CACHE_NPZ} + {CACHE_JSON}")
    return data


def load_cache() -> dict | None:
    """Load the pickle-free cache (numeric .npz + JSON labels). Text arrays are
    reconstructed in-memory from JSON, so np.load never needs allow_pickle."""
    try:
        z = np.load(CACHE_NPZ)                      # numeric only, no pickle
        with open(CACHE_JSON) as f:
            labels = json.load(f)
        texts, families = labels["texts"], labels["families"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        # Missing OR truncated/corrupt cache (interrupted write, disk full) →
        # treat as "no cache" so the caller re-embeds rather than crashing.
        return None
    return {
        "texts": np.array(texts, dtype=object),
        "families": np.array(families, dtype=object),
        "embeddings": z["embeddings"],
        "tokens": z["tokens"],
    }


# ── 2. Pure analysis helpers (testable, no model) ─────────────────────────────
def rvq_collisions(tokens: np.ndarray, texts: np.ndarray) -> list[list[str]]:
    """Groups of palette entries whose RVQ tokens are identical — styles MRT2
    receives as the SAME conditioning. Each returned group has >= 2 members."""
    seen: dict[tuple, list[str]] = {}
    for tok, txt in zip(tokens, texts):
        seen.setdefault(tuple(int(x) for x in tok), []).append(str(txt))
    return [g for g in seen.values() if len(g) > 1]


def _color_list(families) -> list[str]:
    return [_FAMILY_COLORS.get(str(f), "#999999") for f in families]


def _scatter(ax, xy, families, texts, title, label_every: int = 3):
    ax.set_facecolor("#0d1117")
    ax.scatter(xy[:, 0], xy[:, 1], c=_color_list(families), s=42,
               edgecolors="#0d1117", linewidths=0.5, alpha=0.95)
    for i, t in enumerate(texts):
        if i % label_every == 0:                 # thin labels so it's readable
            ax.annotate(str(t), (xy[i, 0], xy[i, 1]), fontsize=5.5,
                        color="#c9d1d9", alpha=0.8,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_title(title, color="#e6edf3", fontsize=12)
    ax.tick_params(colors="#6e7681", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#21262d")


def _legend(fig, families):
    present = [f for f in palette.families() if f in set(map(str, families))]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=_FAMILY_COLORS.get(f, "#999"),
                          markeredgecolor="none", markersize=7, label=f)
               for f in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(present)),
               frameon=False, fontsize=8, labelcolor="#c9d1d9")


# ── 3. Renderers ──────────────────────────────────────────────────────────────
def plot_pca(data: dict) -> str:
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(data["embeddings"])
    fig, ax = plt.subplots(figsize=(15, 11), facecolor="#0d1117")
    _scatter(ax, xy, data["families"], data["texts"],
             "MRT2 style space — PCA (global structure)")
    _legend(fig, data["families"])
    out = paths.analysis_path("style_pca.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out


def plot_tsne(data: dict) -> str:
    from sklearn.manifold import TSNE
    n = len(data["embeddings"])
    perp = max(5, min(30, n // 4))
    xy = TSNE(n_components=2, perplexity=perp, init="pca",
              random_state=0).fit_transform(data["embeddings"])
    fig, ax = plt.subplots(figsize=(15, 11), facecolor="#0d1117")
    _scatter(ax, xy, data["families"], data["texts"],
             f"MRT2 style space — t-SNE (local neighbourhoods, perplexity {perp})")
    _legend(fig, data["families"])
    out = paths.analysis_path("style_tsne.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out


def plot_clusters(data: dict, k: int = 10) -> str:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    embs = data["embeddings"]
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embs)
    xy = PCA(n_components=2).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(15, 11), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    cmap = matplotlib.colormaps["tab10"].resampled(k)
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, s=44,
               edgecolors="#0d1117", linewidths=0.5)
    for i, t in enumerate(data["texts"]):
        if i % 2 == 0:
            ax.annotate(str(t), (xy[i, 0], xy[i, 1]), fontsize=5.5,
                        color="#c9d1d9", alpha=0.85,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_title(f"MRT2 style space — {k} k-means clusters (what groups together)",
                 color="#e6edf3", fontsize=12)
    ax.tick_params(colors="#6e7681", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#21262d")
    out = paths.analysis_path("style_clusters.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    # Print readable cluster membership.
    print(f"\n── {k} k-means clusters ──")
    for c in range(k):
        members = [str(t) for t, l in zip(data["texts"], labels) if l == c]
        print(f"  cluster {c}: {', '.join(members[:12])}"
              + (" …" if len(members) > 12 else ""))
    return out


def plot_similarity(data: dict) -> str:
    # Family-order the rows so blocks of similar styles are visible.
    order = np.argsort([palette.families().index(str(f)) if str(f) in palette.families()
                        else 99 for f in data["families"]])
    embs = data["embeddings"][order]
    texts = data["texts"][order]
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="#0d1117")
    im = ax.imshow(sim, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Pairwise cosine similarity (family-ordered)",
                 color="#e6edf3", fontsize=12)
    step = max(1, len(texts) // 60)
    ticks = range(0, len(texts), step)
    ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
    ax.set_xticklabels([texts[i] for i in ticks], rotation=90, fontsize=4, color="#8b949e")
    ax.set_yticklabels([texts[i] for i in ticks], fontsize=4, color="#8b949e")
    fig.colorbar(im, ax=ax, fraction=0.025).ax.tick_params(colors="#8b949e", labelsize=7)
    out = paths.analysis_path("style_similarity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out


def render_all(data: dict) -> list[str]:
    outs = [plot_pca(data), plot_tsne(data), plot_clusters(data), plot_similarity(data)]
    if data["tokens"].shape[1] > 0:
        cols = rvq_collisions(data["tokens"], data["texts"])
        print(f"\n── RVQ token collisions (styles MRT2 receives identically): "
              f"{len(cols)} group(s) ──")
        for g in cols:
            print(f"  ≡ {', '.join(g)}")
    print("\nWrote:")
    for o in outs:
        print(f"  {o}")
    return outs


def main(argv: list[str]) -> None:
    reviz = "--reviz" in argv
    data = load_cache()
    if data is None or not reviz:
        if reviz:
            print("no cache found — embedding first.");
        data = data if (reviz and data is not None) else embed_all()
    render_all(data)


if __name__ == "__main__":
    main(sys.argv[1:])
