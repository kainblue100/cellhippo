"""Join the 10x expression matrix, the GHIST annotation, and cell coordinates.

Inputs per dataset:
  tenx/<tag>__<ver>__<name>/<name>_cell_feature_matrix.h5   counts (sparse)
  tenx/<tag>__<ver>__<name>/<name>_cells.parquet            x_centroid, y_centroid
  ghist/.../data_input/<set>/xenium_<set>_by_annotation.csv cell_id, group

Output: work/<tag>_full.parquet  (cell_id, annotation, cluster, x, y, PC1..PCn)

Expression is reduced with a PCA fitted on log1p(CP10k) counts, because the
downstream clustering wants a dense, comparable feature block and the raw panel
is 300-5000 genes wide.  The PCA is computed once per dataset and saved, so the
step is auditable rather than hidden inside the model.
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

GH = ("/mnt/storage/bzhang/AIVC/ghist/"
      "D10ny3u3-GHIST_input-4bf5fa04284b8c794887ef69456cf9588cb9290b")
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
os.makedirs(WORK, exist_ok=True)

MAP = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE",
           "data_input/5k/xenium_5kt_by_annotation.csv",
           "data_input/5k/xenium_5kt_by_seu_clusters_res_0.3.csv"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE",
           "data_input/lc/xenium_lc_by_annotation.csv",
           "data_input/lc/xenium_lc_by_seu_clusters_res_0.3.csv"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE",
             "data_input/luad/01_seurat_process/xenium_luad_by_annotation.csv",
             "data_input/luad/01_seurat_process/xenium_luad_by_seu_clusters_res_0.3.csv"),
}
NPC = 32


def load_h5(path):
    """10x cell_feature_matrix.h5 -> cells x genes CSR.

    The file is CSC with shape (n_features, n_cells): indptr runs over cells,
    so it must be transposed.  Non-expression features (negative control probes,
    blanks, codewords) are dropped, otherwise they enter the PCA as noise.
    """
    with h5py.File(path, "r") as h:
        m = h["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        n_feat, n_cells = (int(x) for x in m["shape"][:])
        barcodes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in m["barcodes"][:]])
        genes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                          for x in m["features"]["name"][:]])
        ftype = np.array([x.decode() if isinstance(x, bytes) else str(x)
                          for x in m["features"]["feature_type"][:]])

    assert indptr.shape[0] == n_cells + 1, (indptr.shape, n_cells)
    X_csc = sp.csc_matrix((data, indices, indptr), shape=(n_feat, n_cells))
    keep = np.where(ftype == "Gene Expression")[0]
    X = X_csc[keep, :].T.tocsr()
    return X, barcodes, genes[keep]


summary = {}
for tag, (ver, name, ann_rel, clu_rel) in MAP.items():
    d = os.path.join(TENX, f"{tag}__{ver}__{name}")
    h5 = os.path.join(d, f"{name}_cell_feature_matrix.h5")
    pq = os.path.join(d, f"{name}_cells.parquet")
    print(f"\n=== {tag} ({ver}/{name}) ===")

    X, bc, genes = load_h5(h5)
    print(f"  matrix {X.shape}  genes {len(genes)}  nnz {X.nnz:,}")

    ann = pd.read_csv(os.path.join(GH, ann_rel)).rename(columns={"group": "annotation"})
    clu = pd.read_csv(os.path.join(GH, clu_rel)).rename(columns={"group": "cluster"})
    lab = ann[["cell_id", "annotation"]].merge(
        clu[["cell_id", "cluster"]], on="cell_id", how="left")

    coords = pd.read_parquet(pq)
    xcol = next((c for c in coords.columns if "x_centroid" in c), None)
    ycol = next((c for c in coords.columns if "y_centroid" in c), None)
    coords = coords[["cell_id", xcol, ycol]].rename(
        columns={xcol: "x", ycol: "y"})

    # align everything to the matrix barcode order
    idx = pd.DataFrame({"cell_id": bc, "row": np.arange(len(bc))})
    df = idx.merge(lab, on="cell_id", how="inner").merge(coords, on="cell_id", how="left")
    cover = len(df) / len(bc)
    print(f"  barcodes {len(bc):,}  matched to annotation {len(df):,} "
          f"({cover*100:.2f}%)")
    assert cover > 0.95, f"{tag}: only {cover*100:.1f}% of barcodes carry a label"

    Xs = X[df["row"].to_numpy()]
    # log1p(CP10k) then PCA/SVD; float32 keeps the 78M-nnz 5K matrix in memory
    tot = np.asarray(Xs.sum(axis=1)).ravel()
    tot[tot == 0] = 1
    Xn = (sp.diags(1e4 / tot) @ Xs).astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    from sklearn.decomposition import PCA, TruncatedSVD
    if Xn.shape[1] <= 512:
        Z = PCA(n_components=min(NPC, Xn.shape[1] - 1),
                random_state=0).fit_transform(Xn.toarray())
        var = None
    else:
        sv = TruncatedSVD(n_components=NPC, random_state=0).fit(Xn)
        Z = sv.transform(Xn)
        var = float(sv.explained_variance_ratio_.sum())
    pcs = [f"PC{i+1}" for i in range(Z.shape[1])]
    out = pd.concat([df.reset_index(drop=True),
                     pd.DataFrame(Z, columns=pcs)], axis=1)
    p = os.path.join(WORK, f"{tag}_full.parquet")
    out.to_parquet(p, index=False)
    print(f"  wrote {p}  shape {out.shape}  "
          f"annotations {out.annotation.nunique()}  clusters {out.cluster.nunique()}")
    summary[tag] = {"matrix": list(X.shape), "cells_used": int(len(out)),
                    "coverage": round(cover, 4), "features": pcs,
                    "svd_var": var}

json.dump(summary, open(os.path.join(WORK, "assemble_summary.json"), "w"), indent=2)
print("\n=== done ===")
for k, v in summary.items():
    print(f"  {k:5s} {v['cells_used']:8,} cells  {v['matrix'][1]} genes  "
          f"coverage {v['coverage']*100:.2f}%")
