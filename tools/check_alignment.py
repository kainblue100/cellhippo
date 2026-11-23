"""Decide empirically how he_imagealignment.csv maps microns to pixels.

The two candidate conventions disagree by a factor of ~2.84 on the 5k slide:

  A  raw matrix, scale 1.288 px/um, applied to level-0 pixels
  B  raw matrix rescaled so that one pixel is the OME PhysicalSizeX

Neither can be settled by reading documentation.  Two measurements settle it:
  1. what fraction of cell centroids land inside the image
  2. how dark the image is where they land -- tissue is dark, glass is bright.
     A wrong affine puts cells on background or off the image entirely.
"""
import gzip
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}


def ome_phys(path):
    with tifffile.TiffFile(path) as t:
        ome = t.ome_metadata or ""
    mx = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
    return float(mx.group(1)) if mx else None


for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not downloaded yet")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    um = ome_phys(he)

    print(f"\n=== {tag} ===")
    print(f"  alignment matrix:\n{np.array2string(M, precision=6)}")
    print(f"  OME PhysicalSizeX = {um} um/px  -> {1/um:.4f} px/um")
    print(f"  matrix scale      = {np.hypot(M[0,0], M[1,0]):.4f} px/um")
    print(f"  ratio             = {(1/um)/np.hypot(M[0,0], M[1,0]):.4f}")

    with tifffile.TiffFile(he) as t:
        lv = t.series[0].levels
        shapes = [x.shape for x in lv]
    H0, W0 = shapes[0][0], shapes[0][1]
    print(f"  level 0 {W0}x{H0} px = {W0*um:.0f}x{H0*um:.0f} um")
    print(f"  cells.parquet x {cells.x_centroid.min():.0f}..{cells.x_centroid.max():.0f} um, "
          f"y {cells.y_centroid.min():.0f}..{cells.y_centroid.max():.0f} um")

    x = cells.x_centroid.to_numpy()
    y = cells.y_centroid.to_numpy()
    pts = np.column_stack([x, y, np.ones(len(x))])

    for label, S in (("A raw as level-0 px", M),
                     ("B rescaled to PhysicalSize",
                      M @ np.diag([1.0 / (np.hypot(M[0, 0], M[1, 0]) * um),
                                   1.0 / (np.hypot(M[0, 0], M[1, 0]) * um), 1.0]))):
        px = (S @ pts.T).T
        col, row = px[:, 0], px[:, 1]
        inside = ((col >= 0) & (col < W0) & (row >= 0) & (row < H0))
        frac = inside.mean()
        span = (col.max() - col.min(), row.max() - row.min())
        print(f"  {label:28s} inside {frac*100:6.2f}%  "
              f"col range {col.min():9.0f}..{col.max():9.0f}  "
              f"row {row.min():9.0f}..{row.max():9.0f}")

    # intensity test at a coarse level, so it is cheap
    lvl_i = max(0, len(shapes) - 4)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[lvl_i].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    g = arr[..., :3].astype(np.float32).mean(axis=2)
    Hs, Ws = g.shape
    sc = 2 ** lvl_i
    print(f"  intensity sampled at L{lvl_i} {Ws}x{Hs}, scale {sc}")

    for label, S in (("A raw", M),
                     ("B rescaled",
                      M @ np.diag([1.0 / (np.hypot(M[0, 0], M[1, 0]) * um),
                                   1.0 / (np.hypot(M[0, 0], M[1, 0]) * um), 1.0]))):
        px = (S @ pts.T).T
        c = (px[:, 0] / sc).astype(int)
        r = (px[:, 1] / sc).astype(int)
        ok = (c >= 0) & (c < Ws) & (r >= 0) & (r < Hs)
        vals = g[r[ok], c[ok]]
        print(f"  {label:12s} mean intensity at cells {vals.mean():6.1f}  "
              f"(global mean {g.mean():.1f})  n={ok.sum():,}")
