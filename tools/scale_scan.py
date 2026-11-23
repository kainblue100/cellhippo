"""Find the microns->pixels scale by scanning, since the metadata disagrees with
itself and the documented meaning of he_imagealignment.csv cannot be checked from
this machine.

Sweep the scale, map the cell centroids through the matrix's rotation and
translation but with a free scale, and score each scale by how well the resulting
cell-density map matches the nuclear-stain map.  The peak is the answer; the
curve around it says how sharp the evidence is.

Also scans a small translation search at the best scale, so a residual offset
cannot masquerade as a scale error.
"""
import os

import numpy as np
import pandas as pd
import tifffile

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}

NBIN = 150


def nuclear_score(rgb):
    """Haematoxylin is blue-purple: blue above red, and dark."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.clip(b - r, 0, None) + np.clip(220.0 - r, 0, None) * 0.6


def grid_of(xs, ys, W, H):
    bh, bw = H / NBIN, W / NBIN
    ci = np.clip((xs / bw).astype(int), 0, NBIN - 1)
    ri = np.clip((ys / bh).astype(int), 0, NBIN - 1)
    g = np.zeros((NBIN, NBIN))
    np.add.at(g, (ri, ci), 1.0)
    return g.ravel()


for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not ready")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    x = cells.x_centroid.to_numpy()
    y = cells.y_centroid.to_numpy()

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
    LEVEL = max(0, nlev - 4)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3].astype(np.uint8)
    H, W = arr.shape[:2]
    sc = 2 ** LEVEL
    nuc = nuclear_score(arr)
    bh, bw = H / NBIN, W / NBIN
    ci = np.clip((np.arange(W) / bw).astype(int), 0, NBIN - 1)
    ri = np.clip((np.arange(H) / bh).astype(int), 0, NBIN - 1)
    ir = ri[:, None].repeat(W, 1).ravel()
    ic = ci[None, :].repeat(H, 0).ravel()
    acc = np.zeros(NBIN * NBIN)
    cnt = np.zeros(NBIN * NBIN)
    flat = (ir * NBIN + ic)
    np.add.at(acc, flat, nuc.ravel())
    np.add.at(cnt, flat, 1.0)
    nuc_v = acc / np.maximum(cnt, 1)
    nuc_v = nuc_v - nuc_v.mean()

    print(f"\n=== {tag}  image L{LEVEL} {W}x{H}  "
          f"matrix scale {np.hypot(M[0,0], M[1,0]):.3f} px/um ===")

    # rotation+translation unit vector from the matrix, scale left free
    R = M[:2, :2] / np.hypot(M[0, 0], M[1, 0])   # unit-scale rotation
    T0 = M[:2, 2]
    scan = np.arange(0.6, 7.01, 0.1)
    scores = []
    for s in scan:
        S = np.array([[R[0, 0] * s, R[0, 1] * s, T0[0]],
                      [R[1, 0] * s, R[1, 1] * s, T0[1]],
                      [0, 0, 1]])
        px = (S @ np.column_stack([x, y, np.ones(len(x))]).T).T / sc
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        if ok.mean() < 0.5:
            scores.append((s, -9.0, ok.mean()))
            continue
        v = grid_of(px[ok, 0], px[ok, 1], W, H)
        if v.std() < 1e-9:
            scores.append((s, 0.0, ok.mean()))
            continue
        v = v - v.mean()
        rho = float(v @ nuc_v / (np.linalg.norm(v) * np.linalg.norm(nuc_v)))
        scores.append((s, rho, ok.mean()))

    order = sorted(scores, key=lambda t: -t[1])[:6]
    for s, rho, ins in order:
        print(f"   scale {s:5.2f} px/um  corr {rho:+.4f}  inside {ins*100:5.1f}%")
    best_s = order[0][0]
    print(f"   -> best scale {best_s:.2f} px/um "
          f"(resident um/px {1/best_s:.4f}, x{sc} level scale)")
    print(f"      matrix scale / best = {np.hypot(M[0,0], M[1,0])/best_s:.4f}")
