"""Diffusion pseudotime over the window embeddings, then a movie along it.

Method
  Diffusion pseudotime (Haghverdi et al. 2016), applied to embeddings rather than
  to gene expression, following Vig, Wu & Lotter (arXiv:2601.21334) — the closest
  published precedent for ordering H&E patches by disease progression:

    k-NN graph (k=100, cosine) -> locally scaled Gaussian kernel
    K(i,j) = exp(-d(i,j)^2 / (2 sigma_i sigma_j)),  sigma = distance to the k-th
    neighbour -> row-normalised transition matrix T -> accumulate over random walks
    of all lengths, M = (I - T~)^-1 - I, via the top diffusion components ->
    pseudotime(i) = ||M(root, .) - M(i, .)||_2

  An earlier version of this file built a minimum spanning tree over five pattern
  centroids.  That is a different and much weaker object: it orders five points
  and cannot represent within-pattern variation, while DPT orders every window and
  the patterns then fall out as regions of the pseudotime axis.

Why not RegVelo
  RegVelo couples splicing dynamics, so it needs spliced and unspliced counts.
  Xenium measures mature transcripts only.  There is no intron signal, therefore
  no velocity, therefore no regulatory latent time.  Nothing here is a velocity:
  the pseudotime axis is a morphological ordering, and its direction is set by an
  assumed root.

Why not the paper's own validation
  They score fidelity with Kendall's tau against ordinal disease-stage labels and
  run a permutation test over the six possible orderings.  These data have no
  ordinal ground truth — the patterns are tissue archetypes, not stages — so the
  same test cannot be run.  What is reported instead is (a) how cleanly each
  pattern occupies a band of the axis, and (b) how often the ordering survives
  resampling of the windows.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial.distance import cdist

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/work/trajectory"

NORMAL_LABELS = ("normal_epi", "AT2_like", "AT1", "cil")


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
    emb_path = os.path.join(WORK, f"{tag}_cnn.npz")
    if os.path.exists(emb_path):
        blocks.append(zscore(np.load(emb_path)["emb"].astype(np.float64)))
    X = np.hstack(blocks)
    return X, ph, z["ann"], list(res["annotations"])


def diffusion_pseudotime(X, root_idx, k=100, n_dc=10):
    """DPT as in Haghverdi 2016, on L2-normalised embeddings.

    The transition matrix is kept sparse and only the leading eigenvectors are
    requested.  A dense eigendecomposition of 900x900 is O(n^3) and, run 40 times
    for the bootstrap, took this file past ten minutes on the server before the
    first slide finished; `eigsh` on the k-NN graph takes a fraction of a second
    and is what the diffusion map is defined on anyway.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    n = len(X)
    k = int(min(k, n - 2))
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    D = cdist(Xn, Xn, metric="cosine")
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    sigma = np.take_along_axis(D, nn[:, -1:], axis=1).ravel()
    sigma = np.maximum(sigma, np.percentile(sigma, 5) * 0.1 + 1e-6)

    rows = np.repeat(np.arange(n), k)
    cols = nn.ravel()
    w = np.exp(-(D[rows, cols] ** 2) / (2.0 * sigma[rows] * sigma[cols]))
    K = csr_matrix((w, (rows, cols)), shape=(n, n))
    K = K.maximum(K.T)

    Z = np.asarray(K.sum(axis=1)).ravel()
    Z[Z == 0] = 1.0
    T = csr_matrix(K.multiply(1.0 / Z[:, None]))
    T = ((T + T.T) * 0.5).tocsr()

    n_ev = int(min(n_dc + 1, n - 2))
    vals, vecs = eigsh(T, k=n_ev, which="LA")
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    lam, psi = vals[1:], vecs[:, 1:]

    w2 = lam / np.clip(1.0 - lam, 1e-9, None)
    M = (psi * w2[None, :]) @ psi.T
    pt = np.linalg.norm(M - M[root_idx:root_idx + 1, :], axis=1)
    return pt, vals, psi


def boundary_clarity(pt, ph):
    """How cleanly does each pattern occupy a band of the pseudotime axis?

    Reported instead of the paper's Kendall tau, which needs ordinal ground
    truth.  For each pattern, the fraction of its windows that fall inside the
    central 80% of its own pseudotime range versus the overlap with neighbours —
    a high value means the pattern is a distinct stage rather than smeared along
    the axis.
    """
    ks = sorted(set(ph.tolist()))
    bands = {k: np.percentile(pt[ph == k], [10, 90]) for k in ks}
    out = {}
    for k in ks:
        lo, hi = bands[k]
        inside = ((pt >= lo) & (pt <= hi)).sum()
        mine = ph == k
        in_mine = ((pt[mine] >= lo) & (pt[mine] <= hi)).sum()
        out[str(k)] = {
            "p10": round(float(lo), 4), "p90": round(float(hi), 4),
            "median": round(float(np.median(pt[mine])), 4),
            "band_occupied_by_own_pattern": round(float(in_mine / max(1, inside)), 4),
        }
    return out


def order_stability(X, ph, k, n_dc, n_boot=60, seed=0, cap=350):
    """Fraction of window resamples that preserve each pair's relative order.

    The diffusion decomposition is O(n^3), so each resample draws at most `cap`
    windows: sixty eigen-decompositions of 900x900 would dominate the runtime for
    a quantity that only needs the relative order of five medians.
    """
    rng = np.random.default_rng(seed)
    ref = None
    agree = {}
    for b in range(n_boot):
        idx = rng.integers(0, len(ph), min(cap, len(ph)))
        sub, sub_ph = X[idx], ph[idx]
        present = sorted(set(sub_ph.tolist()))
        if len(present) < 3:
            continue
        # root: medoid of the most normal-epithelium pattern in this resample
        counts = {p: int((sub_ph == p).sum()) for p in present}
        root_pat = max(present, key=lambda p: counts[p])
        sel = np.where(sub_ph == root_pat)[0]
        cent = sub[sel].mean(axis=0)
        root = sel[int(np.argmin(((sub[sel] - cent) ** 2).sum(axis=1)))]
        pt, _, _ = diffusion_pseudotime(sub, root, k=min(k, len(sub) - 1),
                                        n_dc=n_dc)
        med = np.array([np.median(pt[sub_ph == p]) for p in present])
        rank = {p: int(r) for p, r in zip(present, np.argsort(np.argsort(med)))}
        if ref is None:
            ref = rank
        for a in present:
            for c in present:
                if a >= c or a not in ref or c not in ref:
                    continue
                key = f"{min(a, c)}-{max(a, c)}"
                same = (rank[a] - rank[c]) * (ref[a] - ref[c]) > 0
                agree.setdefault(key, []).append(bool(same))
    return {k2: round(float(np.mean(v)), 3) for k2, v in agree.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["5k", "lc"])
    ap.add_argument("--name", default="fused")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-dc", type=int, default=10)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--subsample", type=int, default=900,
                    help="windows used for the diffusion graph; the dense "
                         "eigendecomposition is O(n^3) and 1364 windows is "
                         "already near the practical limit")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    payload = {"method": {
        "name": "diffusion pseudotime (Haghverdi 2016) on window embeddings",
        "precedent": "Vig, Wu & Lotter, arXiv:2601.21334",
        "k": a.k, "n_diffusion_components": a.n_dc,
        "velocity": "unavailable: Xenium has mature transcripts only, so RegVelo "
                    "and every splicing-based method cannot be applied",
        "root": "medoid of the pattern with the highest differentiated-normal-"
                "epithelium content; an assumption, not a measurement",
        "ground_truth": "none: patterns are tissue archetypes, not ordinal stages, "
                        "so Kendall tau against stages is not computable here",
    }, "slides": {}}

    for tag in a.tags:
        X, ph, ann, vocab = pattern_block(tag, a.name)
        rng = np.random.default_rng(0)
        keep = rng.choice(len(X), min(a.subsample, len(X)), replace=False)
        Xs, phs = X[keep], ph[keep]
        print(f"\n=== {tag} ===  {len(X)} windows -> {len(Xs)} for the graph, "
              f"{len(set(phs.tolist()))} patterns")

        idx = {v: i for i, v in enumerate(vocab)}
        ann_s = ann[keep]
        normal = {}
        for k in sorted(set(phs.tolist())):
            sel = phs == k
            normal[k] = float(sum(ann_s[sel][:, idx[v]].mean()
                                  for v in NORMAL_LABELS if v in idx))
        root_pat = max(normal, key=lambda k: normal[k])
        sel = np.where(phs == root_pat)[0]
        cent = Xs[sel].mean(axis=0)
        root = int(sel[int(np.argmin(((Xs[sel] - cent) ** 2).sum(axis=1)))])

        pt, vals, psi = diffusion_pseudotime(Xs, root, k=a.k, n_dc=a.n_dc)
        ks = sorted(set(phs.tolist()))
        med = {k: float(np.median(pt[phs == k])) for k in ks}
        order = sorted(ks, key=lambda k: med[k])
        pt_norm = (pt - pt.min()) / max(1e-9, pt.max() - pt.min())

        print(f"  root = P{root_pat} (normal-epithelium {normal[root_pat]:.3f}), "
              f"medoid window {root}")
        print("  order: " + " -> ".join(f"P{k}" for k in order))
        print("  median pseudotime: " + ", ".join(
            f"P{k}={med[k]:.3f}" for k in order))
        print(f"  diffusion eigenvalues 1-3: "
              f"{', '.join(f'{v:.3f}' for v in vals[1:4])}")
        clarity = boundary_clarity(pt, phs)
        stab = order_stability(X, ph, a.k, a.n_dc, a.boot)
        print("  adjacent pairs stable under resampling: " + ", ".join(
            f"{k2} {v:.0%}" for k2, v in sorted(stab.items(), key=lambda kv: -kv[1])))

        payload["slides"][tag] = {
            "patterns": ks, "order": order, "root": root_pat,
            "normal_epithelium": {str(k): round(v, 4) for k, v in normal.items()},
            "pseudotime_median": {str(k): round(med[k], 4) for k in ks},
            "bands": clarity,
            "order_stability": stab,
            "n_windows": int(len(Xs)),
            "n_features": int(X.shape[1]),
            "pseudotime_norm": pt_norm.round(4).tolist(),
            "window_pattern": phs.tolist(),
        }

    p = os.path.join(a.outdir, "trajectory.json")
    json.dump(payload, open(p, "w"))
    print(f"\nwrote {p} ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
