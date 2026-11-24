"""Cross-sample integration.

Every earlier section describes one slide at a time.  This one asks the question
that only makes sense with several: given that the patterns were fitted jointly,
how similar are the samples to each other, and which parts of the picture are
reproducible?

Three things are computed, and they are deliberately of different kinds so that
agreement between them means something:

  composition   each sample as a vector of pattern fractions, and the
                correlation between samples.  This is the HIPPO paper's own
                patient-level representation, used here with three pseudo-
                patients instead of 209.

  signature     each pattern's mean cell-type composition.  Says what a pattern
                *is*, independently of where it was found.

  stability     the same composition matrix recomputed on the ablation runs
                (image only, expression only).  If a pattern's profile is the
                same whichever modality found it, that pattern is a property of
                the tissue; if it moves, the pattern is a property of the
                modality.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
DOCS = "/mnt/storage/bzhang/AIVC/cellhippo/docs"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures/cross"

PALETTE = ["#c2410c", "#1d4ed8", "#0f766e", "#7c3aed", "#db2777", "#a16207",
           "#0369a1", "#15803d"]


def load(name):
    return json.load(open(os.path.join(WORK, f"{name}.json"), encoding="utf-8"))


def composition(res):
    """(n_patterns x n_samples) matrix of window fractions."""
    ph = np.array(res["tile_pheno"])
    sl = np.array(res["slides"])
    tags = list(res["tags"])
    G = int(ph.max()) + 1
    M = np.zeros((G, len(tags)))
    for j, t in enumerate(tags):
        sel = sl == t
        n = int(sel.sum())
        for k in range(G):
            M[k, j] = float((ph[sel] == k).sum()) / max(n, 1)
    return M, tags, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fused")
    ap.add_argument("--ablations", nargs="+",
                    default=["ab_img", "ab_exprann", "ab_expr"])
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    fused = load(a.name)
    M, tags, G = composition(fused)
    print(f"=== {a.name} ===  {G} patterns x {len(tags)} samples, "
          f"{fused['n_tiles']} windows on {len(tags)} slides")
    print()
    header = "      " + "".join(f"{t:>9s}" for t in tags)
    print(header)
    for k in range(G):
        print(f"   P{k} " + "".join(f"{M[k, j]*100:8.1f}%" for j in range(len(tags))))

    # ---- sample-sample correlation, from the composition vectors
    C = np.corrcoef(M.T)
    print("\n=== sample-sample correlation of the pattern composition ===")
    print(header)
    for i, t in enumerate(tags):
        print(f"   {t:4s} " + "".join(f"{C[i, j]:9.2f}" for j in range(len(tags))))

    # ---- per-pattern cell-type signature, weighted by cells not windows
    print("\n=== each pattern's cell-type signature (mean fraction over its "
          "windows) ===")
    Z = {}
    for t in tags:
        z = np.load(os.path.join(WORK, f"{t}_tiles.npz"))
        Z[t] = z
    vocab = [str(v) for v in fused["annotations"]]
    idx = {v: i for i, v in enumerate(vocab)}
    sig = np.zeros((G, len(vocab)))
    cnt = np.zeros(G)
    for t in tags:
        z = Z[t]
        A = np.zeros((len(z["ann"]), len(vocab)))
        for j, lab in enumerate(z["annotations"]):
            A[:, idx[str(lab)]] = z["ann"][:, j]
        ph = np.array(fused["tile_pheno"])[
            np.array([s == t for s in fused["slides"]])]
        for k in range(G):
            sel = ph == k
            if sel.sum():
                sig[k] += A[sel].sum(axis=0)
                cnt[k] += sel.sum()
    sig = sig / np.maximum(cnt, 1)[:, None]

    for k in range(G):
        top = np.argsort(-sig[k])[:5]
        print(f"   P{k}  " + ", ".join(f"{vocab[i]} {sig[k, i]*100:.0f}%"
                                       for i in top))

    # ---- stability across modalities.  The ablations find a DIFFERENT number of
    #      patterns (3 vs 4), so the composition matrices cannot be compared
    #      element-wise -- the first version tried and died on the shape
    #      mismatch.  What can be compared is the sample-sample correlation,
    #      which is 3x3 either way: does a different modality agree on which
    #      samples resemble each other?
    print("\n=== does a different modality agree on which samples resemble "
          "each other? ===")
    iu = np.triu_indices(len(tags), 1)
    base = C[iu]
    stab = {}
    for ab in a.ablations:
        if not os.path.exists(os.path.join(WORK, f"{ab}.json")):
            continue
        abr = load(ab)
        M2, tags2, G2 = composition(abr)
        C2 = np.corrcoef(M2.T)
        r = float(np.corrcoef(base, C2[iu])[0, 1])
        pairs = ", ".join(f"{tags[i]}-{tags[j]} {C2[i, j]:+.2f}"
                          for i, j in zip(*iu))
        print(f"   {ab:11s} {G2} patterns   sample-pair correlation with "
              f"fused r = {r:.2f}")
        print(f"               its own sample pairs: {pairs}")
        stab[ab] = {"n_patterns": G2, "sample_pair_r": r,
                    "sample_correlation":
                        [[round(float(v), 4) for v in row] for row in C2]}

    # ---- how separated are the patterns by cell type?  If this comes out near
    #      zero the patterns are not cell-type-defined, which matters a great
    #      deal for a method whose selling point is the annotation modality.
    print("\n=== how separated are the patterns by cell type? ===")
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(sig, metric="cosine"))
    print(f"   cosine distance between pattern signatures: "
          f"min {D[D > 0].min():.3f}, max {D.max():.3f}")
    pd_info = {}
    for k in range(G):
        j = int(np.argmin(np.where(np.arange(G) == k, np.inf, D[k])))
        pd_info[str(k)] = {"closest": j, "distance": round(float(D[k, j]), 4)}
        print(f"   P{k} closest to P{j} at {D[k, j]:.3f} "
              f"(a value near 0 means the two patterns have the same "
              f"cell-type composition)")

    # ---- which sample is each pattern characteristic of
    print("\n=== where each pattern lives ===")
    where = {}
    for k in range(G):
        j = int(np.argmax(M[k]))
        print(f"   P{k}  max in {tags[j]:5s} ({M[k, j]*100:5.1f}%)   "
              f"range across samples {M[k].min()*100:.1f}%-{M[k].max()*100:.1f}%"
              f"   spread {M[k].max()-M[k].min():.3f}")
        where[str(k)] = {"max_sample": tags[j], "max_frac": float(M[k, j]),
                         "min_frac": float(M[k].min()),
                         "spread": float(M[k].max() - M[k].min())}

    # ---- render the composition heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), facecolor="white",
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, fontsize=11)
    ax.set_yticks(range(G))
    ax.set_yticklabels([f"P{k}" for k in range(G)], fontsize=11)
    for k in range(G):
        for j in range(len(tags)):
            v = M[k, j]
            ax.text(j, k, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=10, color="white" if v > 0.5 else "#1b1f24")
    ax.set_title("模式构成：每个样本的窗口落在各模式的比例", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax = axes[1]
    im2 = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, fontsize=11)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=11)
    for i in range(len(tags)):
        for j in range(len(tags)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(C[i, j]) > 0.6 else "#1b1f24")
    ax.set_title("样本之间的构成相关（Pearson）", fontsize=12)
    fig.colorbar(im2, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    p = os.path.join(a.outdir, "cross_composition.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nwrote {p}")

    json.dump({
        "method": "pattern composition per sample, from the joint clustering",
        "note": "samples are three different donors, not patients of one cohort",
        "patterns": G,
        "samples": tags,
        "composition": [[round(float(v), 4) for v in row] for row in M],
        "sample_correlation": [[round(float(v), 4) for v in row] for row in C],
        "signature": {str(k): {vocab[i]: round(float(sig[k, i]), 4)
                               for i in np.argsort(-sig[k])[:8]}
                      for k in range(G)},
        "signature_top": {str(k): [vocab[i] for i in np.argsort(-sig[k])[:5]]
                          for k in range(G)},
        "pattern_distance": pd_info,
        "pattern_distance_summary": {
            "min": round(float(D[D > 0].min()), 4),
            "max": round(float(D.max()), 4)},
        "where": where,
        "stability": stab,
    }, open(os.path.join(a.outdir, "cross.json"), "w"), indent=2)
    print(f"wrote {a.outdir}/cross.json")


if __name__ == "__main__":
    main()
