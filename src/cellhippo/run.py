"""cellhippo — HIPPO-style phenotype discovery for *annotated* cell tables.

What this is, and what it changes relative to HIPPO:

  HIPPO (Zhang et al., Genome Med 2025;17:98) works on H&E tiles with no labels:
  contrastive clustering gives 64 tile clusters, the 64x64 inter-cluster Pearson
  matrix is merged by a GMM chosen with BIC into ~15 histomorphological
  phenotypes (HIPOs), each patient becomes a vector of HIPO area fractions, and
  consensus clustering of those vectors yields two subtypes (HIPOS-I/II).

  This tool keeps that chain and adds one input dimension: a cell annotation.
  The annotation enters twice -- once as part of the cell state vector (so
  clustering can only separate groups the annotation permits), and once as an
  external yardstick (so the discovered phenotypes can be checked against what a
  pathologist or marker-based annotator already said).

  The second use is the point. HIPPO can only argue that its phenotypes are
  reproducible. With an annotation present, the interesting question becomes
  narrower and answerable: **does the discovered phenotype split an annotated
  cell type, or merely recover it?** A phenotype that cuts one annotation into
  two states is a new pattern; a phenotype that equals one annotation is a
  relabelling, and should be reported as such rather than sold as discovery.

Input contract (long/wide CSV):
  cell_id     required, unique
  annotation  required, the extra dimension
  slide       optional; without it the slide-level step is skipped, not faked
  any other numeric columns are used as features (standardised)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# 1. cell state vectors
# --------------------------------------------------------------------------
def build_state_matrix(df, label_cols, feature_cols):
    """One-hot the label columns, z-score the numeric ones, concatenate.

    One-hot rather than integer codes: an annotation is categorical, and an
    integer code would make "epi vs fib" and "nkt vs mph" distances depend on
    alphabetical order.
    """
    blocks, names = [], []

    for col in label_cols:
        cats = sorted(df[col].astype(str).unique())
        for c in cats:
            blocks.append((df[col].astype(str) == c).to_numpy(dtype=np.float32))
            names.append(f"{col}={c}")

    for col in feature_cols:
        v = df[col].to_numpy(dtype=np.float32)
        sd = v.std()
        blocks.append((v - v.mean()) / (sd if sd > 0 else 1.0))
        names.append(col)

    if not blocks:
        raise ValueError("no usable columns: need at least one label column")
    return np.column_stack(blocks), names


# --------------------------------------------------------------------------
# 2. fine clustering  (HIPPO: contrastive clustering -> ~64 clusters)
# --------------------------------------------------------------------------
def fine_cluster(X, k=64, seed=0, n_init=4):
    """k-means on the state matrix. Returns labels and centroids."""
    from sklearn.cluster import KMeans

    k = int(min(k, max(2, len(X) // 10)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init,
                max_iter=300).fit(X)
    return km.labels_, km.cluster_centers_


# --------------------------------------------------------------------------
# 3. inter-cluster correlation -> GMM -> phenotypes   (HIPPO eq. 1-3, verbatim)
# --------------------------------------------------------------------------
def phenotype_from_clusters(centroids, g_range=(2, 20), seed=0):
    """Pearson between cluster centroids, then GMM chosen by BIC."""
    from sklearn.mixture import GaussianMixture

    C = np.asarray(centroids, dtype=np.float64)
    R = np.corrcoef(C)                     # (k, k), HIPPO eq. 1
    R = np.nan_to_num(R, nan=0.0)

    best = None
    bic_curve = {}
    for g in range(g_range[0], min(g_range[1], len(C) - 1) + 1):
        gm = GaussianMixture(n_components=g, covariance_type="diag",
                             random_state=seed, n_init=2, max_iter=200)
        gm.fit(R)
        bic = gm.bic(R)
        bic_curve[g] = float(bic)
        if best is None or bic < best[1]:
            best = (g, bic, gm)
    g, bic, gm = best
    return g, gm.predict(R), R, bic_curve      # HIPPO eq. 2-3


# --------------------------------------------------------------------------
# 4. slide-level profiles and subtyping
# --------------------------------------------------------------------------
def slide_profiles(df, cell_pheno, slide_col):
    """Fraction of cells per phenotype, per slide (HIPPO eq. 4)."""
    slides = sorted(df[slide_col].astype(str).unique())
    G = int(cell_pheno.max()) + 1
    M = np.zeros((len(slides), G), dtype=np.float64)
    for i, s in enumerate(slides):
        sel = (df[slide_col].astype(str) == s).to_numpy()
        counts = np.bincount(cell_pheno[sel], minlength=G).astype(np.float64)
        M[i] = counts / counts.sum()
    return slides, M


def consensus_subtypes(M, k_range=(2, 5), n_boot=200, frac=0.8, seed=0):
    """Consensus clustering of slide profiles.

    Same shape as HIPPO's use of ConsensusClusterPlus: hierarchical clustering
    with Pearson distance and Ward linkage over bootstrap resamples, choosing k
    by the consensus CDF. Returns None when there are too few slides, because a
    consensus over three points is arithmetic, not evidence.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(M)
    if n < 6:
        return None, {"reason": f"{n} slides is too few for consensus clustering"}

    rng = np.random.default_rng(seed)
    G = M.shape[1]
    cons = {k: np.zeros((n, n)) for k in range(k_range[0], min(k_range[1], n - 1) + 1)}
    counts = {k: np.zeros((n, n)) for k in cons}

    for _ in range(n_boot):
        idx = rng.choice(n, max(3, int(round(n * frac))), replace=False)
        sub = M[idx]
        Z = linkage(pdist(sub, metric="correlation"), method="ward")
        for k in cons:
            lab = fcluster(Z, k, criterion="maxclust")
            same = (lab[:, None] == lab[None, :]).astype(float)
            cons[k][np.ix_(idx, idx)] += same
            counts[k][np.ix_(idx, idx)] += 1

    curves = {}
    for k in cons:
        c = counts[k]
        cons[k] = np.divide(cons[k], c, out=np.zeros_like(cons[k]), where=c > 0)
        iu = np.triu_indices(n, 1)
        curves[k] = float(np.mean(cons[k][iu]))

    # HIPPO reads the CDF plot; here the tabulated equivalent is the mean
    # consensus per k, and we take the smallest k whose mean is within 2% of the
    # best -- preferring fewer subtypes unless more clearly help.
    best_k = max(curves, key=curves.get)
    for k in sorted(curves):
        if curves[k] >= 0.98 * curves[best_k]:
            best_k = k
            break
    Z = linkage(pdist(M, metric="correlation"), method="ward")
    lab = fcluster(Z, best_k, criterion="maxclust")
    return lab, {"curves": curves, "chosen_k": int(best_k),
                 "consensus_matrix": cons[best_k].tolist()}


# --------------------------------------------------------------------------
# 5. the payoff of the extra dimension
# --------------------------------------------------------------------------
def annotation_concordance(df, cell_pheno, ann_col):
    """How do discovered phenotypes relate to the given annotation?

    Reports ARI/NMI (agreement) plus, per annotation, the number of phenotypes
    its cells fall into. High split ratio with moderate ARI is the signature of a
    phenotype that refines an annotation rather than restating it.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = df[ann_col].astype(str).to_numpy()
    p = cell_pheno
    ari = float(adjusted_rand_score(a, p))
    nmi = float(normalized_mutual_info_score(a, p))

    rows = []
    for cat in sorted(set(a)):
        sel = a == cat
        cnt = np.bincount(p[sel])
        frac = cnt / cnt.sum()
        order = np.argsort(-frac)
        major = order[0]
        rows.append({
            "annotation": cat,
            "cells": int(sel.sum()),
            "n_phenotypes": int((frac > 0.05).sum()),
            "dominant_phenotype": int(major),
            "dominant_fraction": float(frac[major]),
            "split": bool(frac[major] < 0.9),
        })
    rows.sort(key=lambda r: -r["cells"])
    return {"ari": ari, "nmi": nmi, "per_annotation": rows}


def phenotype_vs_annotation(df, cell_pheno, ann_col):
    """Cross-tab: for each phenotype, its annotation composition."""
    from collections import Counter

    a = df[ann_col].astype(str).to_numpy()
    out = []
    for g in sorted(set(cell_pheno)):
        sel = cell_pheno == g
        c = Counter(a[sel])
        tot = sum(c.values())
        out.append({"phenotype": int(g), "cells": int(tot),
                    "composition": {k: round(v / tot, 4)
                                    for k, v in c.most_common(6)}})
    return out


def read_table(path):
    """CSV or parquet; the assembly step writes parquet because the 5K dataset
    carries 32 float columns over 278k rows."""
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path)
    import pandas as pd
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(
        description="HIPPO-style phenotype discovery for annotated cell tables")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--annotation", default="group")
    ap.add_argument("--second-label", default=None,
                    help="optional second categorical axis (e.g. a cluster id)")
    ap.add_argument("--slide", default=None)
    ap.add_argument("--features", default="",
                    help="comma-separated numeric columns used as features")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd

    df = read_table(a.csv)
    print(f"cells {len(df):,}  columns {list(df.columns)}", flush=True)

    label_cols = [a.annotation] + ([a.second_label] if a.second_label else [])
    for c in label_cols:
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not in {list(df.columns)}")
    feat_cols = [c for c in a.features.split(",") if c.strip()] if a.features else []

    X, names = build_state_matrix(df, label_cols, feat_cols)
    print(f"state matrix {X.shape}  ({len(label_cols)} label axes, "
          f"{len(feat_cols)} numeric features)", flush=True)

    lab, cent = fine_cluster(X, k=a.k, seed=a.seed)
    print(f"fine clusters: {len(cent)}", flush=True)

    # The GMM labels the *clusters*; a cell inherits its cluster's phenotype.
    g, cluster_pheno, R, bic = phenotype_from_clusters(cent, seed=a.seed)
    cell_pheno = cluster_pheno[lab]
    print(f"phenotypes after GMM/BIC: {g}  (BIC min at {g})", flush=True)

    result = {
        "cells": int(len(df)),
        "state_shape": list(X.shape),
        "state_columns": names,
        "fine_clusters": int(len(cent)),
        "n_phenotypes": int(g),
        "bic_curve": {str(k): v for k, v in bic.items()},
        "phenotype_counts": {str(k): int(v) for k, v in
                             zip(*np.unique(cell_pheno, return_counts=True))},
    }

    conc = annotation_concordance(df, cell_pheno, a.annotation)
    result["annotation_concordance"] = conc
    print(f"\nannotation concordance: ARI {conc['ari']:.3f}  NMI {conc['nmi']:.3f}")
    split = [r for r in conc["per_annotation"] if r["split"]]
    print(f"  annotations split across >1 phenotype: {len(split)}/"
          f"{len(conc['per_annotation'])}")
    for r in conc["per_annotation"][:8]:
        tag = "SPLIT" if r["split"] else "kept"
        print(f"    {r['annotation']:22s} {r['cells']:8,} cells  "
              f"{r['n_phenotypes']:2d} phenotypes  "
              f"dominant {r['dominant_fraction']*100:5.1f}%  {tag}")

    result["phenotype_composition"] = phenotype_vs_annotation(df, cell_pheno, a.annotation)

    if a.slide and a.slide in df.columns:
        slides, M = slide_profiles(df, cell_pheno, a.slide)
        sub, info = consensus_subtypes(M, seed=a.seed)
        result["slides"] = slides
        result["slide_profiles"] = M.round(4).tolist()
        result["subtyping"] = {"subtypes": None if sub is None else sub.tolist(),
                               "info": info if sub is None else
                               {k: v for k, v in info.items()
                                if k != "consensus_matrix"}}
        print(f"\nslides {len(slides)}  profiles {M.shape}")
        if sub is None:
            print(f"  subtyping skipped: {info['reason']}")
        else:
            print(f"  consensus subtypes (k={info['chosen_k']}): {sub.tolist()}")

    out = a.out or os.path.splitext(a.csv)[0] + "_cellhippo.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
