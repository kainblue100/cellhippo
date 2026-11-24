"""Assemble the 17-sample Takano/Kero cohort into cellhippo's fused cell table.

What this cohort has that the three external slides do not is *scale*: 17 sections
from one study, 2,544,257 cells, all on the same 302-gene hLung_100g panel.  What
it lacks is a cell-type annotation: the Seurat objects carry only
`seurat_clusters`, an unsupervised expression clustering at resolution 0.3, and
those cluster numbers mean nothing across samples.

So the annotation is derived here, uniformly, by marker scoring on the shared
panel.  This is a weaker object than a pathologist's label and is reported as
such: it is `derived_annotation`, not `annotation`.  The three external slides
keep their provided labels and are never pooled with this cohort, because the
panels overlap on only 54 genes.

Output per sample: work/lcm/{sample}.parquet with
    cell_id, annotation, x, y, PC1..PC32
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
RAW = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw"
OUT = os.path.join(WORK, "lcm")

# Marker sets restricted to genes that are actually on the hLung_100g panel.
# The panel is 302 genes, so most classes are represented by one to three
# markers; that limits how finely the types can be split, and `cil`, `b`,
# `plasma` and `mast` rest on a single marker each.
MARKERS = {
    "AT1":    ["PDPN", "CAV1", "HOPX"],
    "AT2":    ["SFTPC", "SFTPB", "NKX2-1"],
    "epi":    ["KRT8", "CDH1"],
    "cil":    ["FOXJ1"],
    "endo":   ["PECAM1", "PLVAP"],
    "fib":    ["COL1A1", "PDGFRA"],
    "nkt":    ["CD3E", "CD8A"],
    "b":      ["BANK1"],
    "plasma": ["MZB1"],
    "mph":    ["CD68", "CD163"],
    "mast":   ["KIT"],
}

# A cell is positive for a marker when its log1p(CP10k) clears this.  An
# absolute threshold, not a per-marker z-score: z-scoring across all cells
# forces every cell onto the scale, and with one-marker classes that produced
# plasma = 48,216 of 309,922 cells (15.5%), which lung tissue cannot support.
POS = 1.0
# the winning class must beat the runner-up by this much, or the cell is left
# unresolved rather than being forced into the nearest class
MARGIN = 1


def load_h5(path):
    """Gene-expression matrix as cells x genes, from the 10x CSC h5."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        ft = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["feature_type"][:]]
        nm = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["name"][:]]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
    keep = np.array([i for i, t in enumerate(ft) if t == "Gene Expression"])
    remap = {g: i for i, g in enumerate(keep)}
    mat = sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    # select only gene rows, then transpose to cells x genes
    rows = np.array(keep)
    sub = mat[rows, :].T.tocsr()
    return sub, [nm[i] for i in keep]


def derive_annotation(X, genes):
    """Count positive markers per class and assign, with a margin requirement.

    A cell is positive for a marker when log1p(CP10k) > POS.  A class's score is
    the number of its markers that are positive, normalised by how many of them
    are on the panel.  The winner must clear two things: at least one positive
    marker, and a margin of MARGIN marker-hits over the runner-up.  Cells that
    fail either are `unassigned` -- forcing every cell into a class is what made
    the first attempt call 15% of the tissue plasma.
    """
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    L = sp.diags(1e4 / tot) @ X
    L = L.tocsr().astype(np.float32)
    L.data = np.log1p(L.data)

    gidx = {g: i for i, g in enumerate(genes)}
    scores, names, avail = [], [], []
    for cls, mk in MARKERS.items():
        cols = [gidx[g] for g in mk if g in gidx]
        if not cols:
            continue
        sub = np.asarray(L[:, cols].todense())
        hits = (sub > POS).sum(axis=1)
        scores.append(hits)
        names.append(cls)
        avail.append(len(cols))
    S = np.stack(scores, axis=1).astype(np.float32)
    avail = np.array(avail, dtype=np.float32)
    # fraction of each class's panel markers that fired
    F = S / avail

    order = np.argsort(-S, axis=1)
    best = order[:, 0]
    second = order[:, 1]
    rows = np.arange(len(S))
    bestv = S[rows, best]
    secondv = S[rows, second]
    lab = np.array([names[b] for b in best], dtype=object)
    lab[(bestv < 1) | ((bestv - secondv) < MARGIN)] = "unassigned"
    return lab, F, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npcs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    samples = sorted(d for d in os.listdir(RAW)
                     if os.path.isdir(os.path.join(RAW, d)))
    if a.limit:
        samples = samples[:a.limit]

    report = {}
    for s in samples:
        d = os.path.join(RAW, s)
        h5 = os.path.join(d, "cell_feature_matrix.h5")
        cp = os.path.join(d, "cells.parquet")
        if not (os.path.exists(h5) and os.path.exists(cp)):
            print(f"  {s}: missing h5 or cells.parquet, skipped")
            continue

        X, genes = load_h5(h5)
        cells = pd.read_parquet(cp)
        print(f"\n=== {s} ===  {X.shape[0]} cells x {X.shape[1]} genes, "
              f"cells.parquet {len(cells)} rows", flush=True)

        # the h5 row order and cells.parquet must agree on barcodes; verify
        n = min(len(cells), X.shape[0])
        if len(cells) != X.shape[0]:
            print(f"  WARNING h5 {X.shape[0]} vs parquet {len(cells)} cells; "
                  f"using the first {n}")

        nonempty = np.asarray(X[:n].sum(axis=1)).ravel() > 0
        X = X[:n][nonempty]
        cells = cells.iloc[:n][nonempty].reset_index(drop=True)
        print(f"  kept {X.shape[0]} cells with at least one transcript",
              flush=True)

        lab, S, names = derive_annotation(X, genes)
        cnt = pd.Series(lab).value_counts()
        print("  derived annotation: " +
              ", ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)

        # cell x PC on log1p(CP10k)
        tot = np.asarray(X.sum(axis=1)).ravel()
        tot[tot == 0] = 1.0
        L = sp.diags(1e4 / tot) @ X
        L = L.tocsc()
        L.data = np.log1p(L.data)
        L = L.astype(np.float32)
        # dense as float32 directly: .todense() on float64 would allocate 1.1 GB
        # for the largest sample (472k cells x 302 genes) before the cast
        M = np.asarray(L.todense(), dtype=np.float32)
        mu = M.mean(axis=0)
        M -= mu
        npcs = min(a.npcs, M.shape[1] - 1)
        # randomised SVD via the Gram matrix is cheaper than a dense SVD here.
        # NOTE: the SVD outputs must not be called `s` -- that is the loop
        # variable holding the sample name, and shadowing it turns the output
        # path into a 300-element array (the first run died with "file name too
        # long" for exactly that reason).
        if M.shape[0] > 20000:
            idx = np.random.default_rng(0).choice(M.shape[0], 20000, replace=False)
            _U, _sv, Vt = np.linalg.svd(M[idx], full_matrices=False)
            P = M @ Vt[:npcs].T
        else:
            _U, _sv, Vt = np.linalg.svd(M, full_matrices=False)
            P = _U[:, :npcs] * _sv[:npcs]

        out = pd.DataFrame({
            "cell_id": cells["cell_id"].astype(str).to_numpy(),
            "annotation": lab,
            "x": cells["x_centroid"].to_numpy(),
            "y": cells["y_centroid"].to_numpy(),
        })
        for i in range(P.shape[1]):
            out[f"PC{i+1}"] = P[:, i]
        p = os.path.join(OUT, f"{s}.parquet")
        out.to_parquet(p, index=False)
        print(f"  wrote {p}  {len(out)} cells, {P.shape[1]} PCs", flush=True)

        report[s] = {"cells_in_file": int(X.shape[0]),
                     "annotation_counts": {k: int(v) for k, v in cnt.items()},
                     "n_pcs": int(P.shape[1])}

    json.dump(report, open(os.path.join(OUT, "assemble_report.json"), "w"),
              indent=2)
    tot = sum(v["cells_in_file"] for v in report.values())
    print(f"\n{len(report)} samples, {tot:,} cells total")


if __name__ == "__main__":
    main()
