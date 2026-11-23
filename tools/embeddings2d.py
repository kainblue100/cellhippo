"""Two-dimensional maps of the window embeddings: UMAP and t-SNE.

The pseudotime axis is a one-dimensional summary and, as the results showed, it
stacks the patterns rather than separating them.  A 2D embedding is the honest
companion: it shows whether the patterns are separated islands or one continuous
manifold, which is a different question from how they order along a curve.

Both projections run on the same fused feature block used everywhere else
(annotation fractions + expression components + image embedding), so a pattern's
position here and its position on the pseudotime axis describe the same object.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures/embedding"

PATTERN_COLOR = {0: "#c2410c", 1: "#1d4ed8", 2: "#0f766e", 3: "#7c3aed",
                 4: "#db2777"}


def zscore(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd


def pattern_block(tag, name):
    res = json.load(open(os.path.join(WORK, f"{name}.json")))
    pheno = np.array(res["tile_pheno"])
    slides = np.array(res["slides"])
    off = 0
    for s in res["tags"]:
        n_s = int((slides == s).sum())
        if s == tag:
            break
        off += n_s
    ph = pheno[off:off + int((slides == tag).sum())]

    z = np.load(os.path.join(WORK, f"{tag}_tiles.npz"))
    blocks = [zscore(z["ann"].astype(np.float64)),
              zscore(z["expr"].astype(np.float64))]
    emb = os.path.join(WORK, f"{tag}_cnn.npz")
    if os.path.exists(emb):
        blocks.append(zscore(np.load(emb)["emb"].astype(np.float64)))
    X = np.hstack(blocks)
    meta = z["meta"]
    return X, ph, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["5k", "lc"])
    ap.add_argument("--name", default="fused")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import umap as umap_mod

    payload = {"method": {
        "umap": "n_neighbors=30, min_dist=0.25, metric=euclidean, on the z-scored "
                "fused block",
        "tsne": "perplexity=30, init=pca, learning_rate=auto",
        "features": "annotation fractions + expression components + image embedding",
    }, "slides": {}}

    for tag in a.tags:
        X, ph, meta = pattern_block(tag, a.name)
        print(f"\n=== {tag} ===  {len(X)} windows, {X.shape[1]} features")

        red = umap_mod.UMAP(n_neighbors=30, min_dist=0.25, metric="euclidean",
                            random_state=a.seed, n_jobs=1)
        U = red.fit_transform(X)
        Ts = TSNE(perplexity=30, init="pca", learning_rate="auto",
                  random_state=a.seed).fit_transform(X)
        print(f"  umap range {U.min():.1f}..{U.max():.1f}, "
              f"tsne range {Ts.min():.1f}..{Ts.max():.1f}")

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), facecolor="white")
        ks = sorted(set(int(v) for v in ph))
        for ax, V, title in ((axes[0], U, "UMAP"), (axes[1], Ts, "t-SNE")):
            for k in ks:
                sel = ph == k
                ax.scatter(V[sel, 0], V[sel, 1], s=16, alpha=0.75,
                           c=PATTERN_COLOR.get(k, "#888888"),
                           edgecolors="none", label=f"P{k}")
            ax.set_title(f"{tag} · {title}", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d6d3d1")
        axes[0].legend(frameon=False, fontsize=10, markerscale=1.6)
        fig.suptitle(f"{tag}: window embeddings coloured by pattern",
                     fontsize=14, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(a.outdir, f"{tag}_embedding.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {p}")

        payload["slides"][tag] = {
            "umap": U.round(3).tolist(),
            "tsne": Ts.round(3).tolist(),
            "pattern": ph.tolist(),
            "grid": meta[:, :2].tolist(),
            "n_windows": int(len(X)),
            "n_features": int(X.shape[1]),
        }

    json.dump(payload, open(os.path.join(a.outdir, "embedding.json"), "w"))
    print(f"\nwrote {a.outdir}/embedding.json")


if __name__ == "__main__":
    main()
