"""Correlate mapped cell density against nuclear stain density across the whole
slide.

Xenium detects cells; haematoxylin stains nuclei.  So a correct transform makes
the cell-density map and the nucleus-density map peak in the same places, and a
wrong one decorrelates them.  Unlike a per-crop check this uses the entire slide,
and unlike an "is it on tissue" check it has a real null: the tissue covers half
the frame, so almost any point lands on tissue, but almost no wrong transform
lands its cells on the nuclei specifically.

Candidates: the matrix as given, its inverse, and both rescaled to the OME
physical pixel size.  Also a shuffled control.
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

NBIN = 120


def nucleus_score(rgb):
    """Haematoxylin is blue-purple and dark; eosin is pink.  Score nuclei as
    darkness in the blue channel relative to the red channel."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.clip((r - b), 0, None) + np.clip(255.0 - r, 0, None) * 0.5


for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not ready")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    pts = np.column_stack([cells.x_centroid.to_numpy(),
                           cells.y_centroid.to_numpy(),
                           np.ones(len(cells))])

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
    LEVEL = max(0, nlev - 5)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3].astype(np.uint8)
    H, W = arr.shape[:2]
    sc = 2 ** LEVEL

    nuc = nucleus_score(arr)
    # bin the nucleus score into the same grid we will bin cells into
    bh, bw = H / NBIN, W / NBIN
    bins_r = np.clip((np.arange(H) / bh).astype(int), 0, NBIN - 1)
    bins_c = np.clip((np.arange(W) / bw).astype(int), 0, NBIN - 1)
    nuc_grid = np.zeros((NBIN, NBIN), dtype=np.float64)
    np.add.at(nuc_grid, (bins_r[:, None].repeat(W, 1).ravel(),
                         bins_c[None, :].repeat(H, 0).ravel()), nuc.ravel())
    cnt_grid = np.zeros((NBIN, NBIN), dtype=np.float64)
    np.add.at(cnt_grid, (bins_r[:, None].repeat(W, 1).ravel(),
                         bins_c[None, :].repeat(H, 0).ravel()), 1.0)
    nuc_mean = nuc_grid / np.maximum(cnt_grid, 1)
    nuc_v = nuc_mean.ravel()
    print(f"\n=== {tag}  L{LEVEL} {W}x{H}  nucleus score mean {nuc.mean():.1f} ===")

    side = np.hypot(M[0, 0], M[1, 0])
    omepx = None
    import re
    with tifffile.TiffFile(he) as t:
        ome = t.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
    if m:
        omepx = float(m.group(1))

    cands = {"M direct": M, "M inverse": np.linalg.inv(M)}
    if omepx:
        f = (1.0 / omepx) / side
        cands["M direct x physsize"] = M @ np.diag([f, f, 1.0])
        cands["M inverse x physsize"] = np.linalg.inv(M) @ np.diag([f, f, 1.0])

    results = []
    for label, S in cands.items():
        px = (S @ pts.T).T
        c = px[:, 0] / sc
        r = px[:, 1] / sc
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        ci = np.clip((c[ok] / bw).astype(int), 0, NBIN - 1)
        ri = np.clip((r[ok] / bh).astype(int), 0, NBIN - 1)
        cg = np.zeros((NBIN, NBIN))
        np.add.at(cg, (ri, ci), 1.0)
        v = cg.ravel()
        if v.std() < 1e-9:
            continue
        rho = float(np.corrcoef(v, nuc_v)[0, 1])
        results.append((label, ok.mean(), rho))
        print(f"  {label:24s} inside {ok.mean()*100:6.2f}%  "
              f"corr(cell density, nuclei) = {rho:+.4f}")

    # shuffled null: keep the mapped distribution but destroy the pairing
    rng = np.random.default_rng(0)
    px = (M @ pts.T).T
    c = px[:, 0] / sc
    r = px[:, 1] / sc
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    nulls = []
    for _ in range(5):
        cs = rng.permutation(c[ok])
        rs = rng.permutation(r[ok])
        ci = np.clip((cs / bw).astype(int), 0, NBIN - 1)
        ri = np.clip((rs / bh).astype(int), 0, NBIN - 1)
        cg = np.zeros((NBIN, NBIN))
        np.add.at(cg, (ri, ci), 1.0)
        nulls.append(float(np.corrcoef(cg.ravel(), nuc_v)[0, 1]))
    print(f"  shuffled null             corr = {np.mean(nulls):+.4f} "
          f"(sd {np.std(nulls):.4f})")
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"  -> best: {best[0]}  rho {best[2]:+.4f}")
