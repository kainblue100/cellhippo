"""HIPPO's steps 2-4 on the fused tile features.

  fine clusters   k-means on the concatenated modality block (HIPPO: contrastive
                  clustering; here the CNN embeddings come from train_cnn.py, so
                  this is the read-out)
  phenotypes      Pearson between cluster centroids -> diagonal GMM over G by BIC
  slide profiles  phenotype composition per slide

Every modality can be switched off, so the contribution of the image, the
expression and the annotation can be measured rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"


def zscore(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def build_block(tag, emb_path, use_img, use_expr, use_ann, vocab,
                w_img=1.0, w_expr=1.0, w_ann=1.0):
    """Concatenate z-scored modality blocks with explicit relative weights.

    The annotation vocabulary is passed in rather than taken from the file: the
    slides do not share a label set (17 on 5k, 14 on lc), so each slide's block
    must be laid out over the union.  Otherwise the pooled matrix is ragged and
    the blocks silently mean different things in different rows.
    """
    blocks, names = [], []
    if use_img:
        E = np.load(emb_path)["emb"].astype(np.float32)
        blocks.append(zscore(E) * w_img)
        names += [f"img{i}" for i in range(E.shape[1])]
    z = np.load(os.path.join(WORK, f"{tag}_tiles.npz"))
    if use_expr:
        X = z["expr"].astype(np.float32)
        blocks.append(zscore(X) * w_expr)
        names += [f"expr{i}" for i in range(X.shape[1])]
    if use_ann:
        A = np.zeros((len(z["ann"]), len(vocab)), dtype=np.float32)
        idx = {a: i for i, a in enumerate(vocab)}
        for j, a in enumerate(z["annotations"]):
            A[:, idx[str(a)]] = z["ann"][:, j]
        blocks.append(zscore(A) * w_ann)
        names += [f"ann={v}" for v in vocab]
    if not blocks:
        raise SystemExit("no modality selected")
    return np.concatenate(blocks, axis=1), names


def phenotype_chain(X, k=64, seed=0, g_range=(2, 15)):
    """k-means, then Pearson between centroids, then a diagonal GMM by BIC.

    The G upper bound is capped well below the cluster count.  Left uncapped the
    BIC on a 64x64 correlation matrix keeps improving as components are added,
    because 64 points in 64 dimensions can always be fit better by more
    Gaussians -- an expr-only run picked 30 'phenotypes' for 2086 tiles, which is
    curve fitting, not structure.
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    k = int(min(k, max(2, len(X) // 4)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)

    C = km.cluster_centers_.astype(np.float64)
    R = np.corrcoef(C)
    R = np.nan_to_num(R, nan=0.0)

    best, curve = None, {}
    hi = min(g_range[1], len(C) - 1)
    for g in range(g_range[0], hi + 1):
        gm = GaussianMixture(n_components=g, covariance_type="diag",
                             random_state=seed, n_init=3).fit(R)
        b = float(gm.bic(R))
        curve[g] = b
        if best is None or b < best[1]:
            best = (g, b, gm)
    g, bic, gm = best
    return km, km.labels_, g, gm.predict(R), R, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--img", action="store_true")
    ap.add_argument("--expr", action="store_true")
    ap.add_argument("--ann", action="store_true")
    ap.add_argument("--w-img", type=float, default=1.0)
    ap.add_argument("--w-expr", type=float, default=1.0)
    ap.add_argument("--w-ann", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    # union annotation vocabulary across slides, fixed before any block is built
    vocab = sorted({str(a) for t in a.tags
                    for a in np.load(os.path.join(WORK, f"{t}_tiles.npz"))["annotations"]})
    print(f"  annotation vocabulary: {len(vocab)} labels")

    Xs, slides, metas = [], [], []
    for t in a.tags:
        emb = os.path.join(WORK, f"{t}_cnn.npz")
        if a.img and not os.path.exists(emb):
            raise SystemExit(
                f"--img requested but {emb} is missing; run train_cnn.py first")
        X, names = build_block(t, emb, a.img, a.expr, a.ann, vocab,
                               a.w_img, a.w_expr, a.w_ann)
        Xs.append(X)
        metas.append(np.load(os.path.join(WORK, f"{t}_tiles.npz"))["meta"])
        slides += [t] * len(X)
        print(f"  {t}: {X.shape} tiles")

    X = np.concatenate(Xs, axis=0)
    metas = np.concatenate(metas, axis=0)
    slides = np.array(slides)
    ann_names = vocab
    print(f"pooled block {X.shape}  modalities: "
          f"img={a.img} expr={a.expr} ann={a.ann}")

    km, fine, g, cluster_pheno, R, curve = phenotype_chain(X, a.k, a.seed)
    print(f"  fine clusters {km.n_clusters}  ->  phenotypes {g}")

    # tile-level lab (no k-means.predict on the whole block needed: we already
    # have km.labels_ from the fit)
    tile_pheno = cluster_pheno[km.labels_]

    G = int(tile_pheno.max()) + 1
    comp = {}
    for s in np.unique(slides):
        sel = slides == s
        cnt = np.bincount(tile_pheno[sel], minlength=G).astype(float)
        comp[s] = (cnt / cnt.sum()).round(4).tolist()

    out = {
        "name": a.name,
        "tags": a.tags,
        "modalities": {"img": a.img, "expr": a.expr, "ann": a.ann},
        "weights": {"img": a.w_img, "expr": a.w_expr, "ann": a.w_ann},
        "n_tiles": int(len(X)),
        "block_dim": int(X.shape[1]),
        "fine_clusters": int(km.n_clusters),
        "n_phenotypes": int(g),
        "bic_curve": {str(k2): v for k2, v in curve.items()},
        "tile_pheno": tile_pheno.tolist(),
        "fine": km.labels_.tolist(),
        "slides": slides.tolist(),
        "tile_meta": metas.tolist(),
        "annotations": ann_names,
        "phenotype_counts": {
            str(k2): int(v) for k2, v in
            zip(*np.unique(tile_pheno, return_counts=True))},
        "slide_profiles": comp,
    }
    np.save(os.path.join(WORK, f"{a.name}_tilepheno.npy"), tile_pheno)
    np.save(os.path.join(WORK, f"{a.name}_fine.npy"), km.labels_)
    json.dump(out, open(os.path.join(WORK, f"{a.name}.json"), "w"))
    print(f"  phenotypes {G}  counts "
          f"{ {k2: int(v) for k2, v in zip(*np.unique(tile_pheno, return_counts=True))} }")
    print(f"wrote {WORK}/{a.name}.json")


if __name__ == "__main__":
    main()
