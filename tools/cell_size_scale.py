"""Pin the scale by cell size, which is measurable on both sides.

Left side: cells.parquet gives cell_area in um^2, so the equivalent diameter in
microns is known exactly.  Multiply it by whatever px/um a candidate transform
implies and you get the outline diameter that transform will draw.

Right side: threshold the H&E for nuclear stain, label the blobs, and measure the
median nucleus diameter in the same pixels.

The correct transform is the one where those two numbers agree.  A scale that is
wrong by 2.8x is off by a factor no blob measurement can hide.
"""
import os
import re

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}

for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not ready")
        continue
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
        ome = t.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
    omepx = float(m.group(1)) if m else None

    area = cells["cell_area"].to_numpy(dtype=float)
    area = area[np.isfinite(area) & (area > 0)]
    d_um = 2 * np.sqrt(np.median(area) / np.pi)
    print(f"\n=== {tag} ===")
    print(f"  median cell_area {np.median(area):.1f} um2  "
          f"-> equivalent diameter {d_um:.2f} um")
    if omepx:
        print(f"  OME pixel size {omepx:.4f} um/px  "
              f"-> {1/omepx:.3f} px/um  -> that diameter is "
              f"{d_um/omepx:.1f} px at level 0")

    side = np.hypot(M[0, 0], M[1, 0])
    for label, pxum in (("raw matrix", side),
                        ("scaled to OME pixel size", (1.0 / omepx) if omepx else None)):
        if pxum is None:
            continue
        print(f"  {label:26s} {pxum:7.3f} px/um -> outline diameter "
              f"{d_um*pxum:6.1f} px at level 0")

    # measure actual nuclei in the image
    LEVEL = max(0, nlev - 2)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3].astype(np.float32)
    sc = 2 ** LEVEL
    # haematoxylin: blue channel dominates and the pixel is dark
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    nuc = (b - r) + (255.0 - r) * 0.4
    # take a window with tissue in it
    H, W = nuc.shape[:2]
    win = nuc[H // 4:H // 2, W // 4:W // 2]
    thr = np.percentile(win, 92)
    msk = win > thr
    lab, n = ndimage.label(msk)
    if n:
        sizes = ndimage.sum(msk, lab, range(1, n + 1))
        sizes = sizes[sizes >= 4]
        if len(sizes):
            d_px = 2 * np.sqrt(np.median(sizes) / np.pi)
            print(f"  measured image: L{LEVEL} (scale {sc}), "
                  f"{len(sizes)} nuclear blobs, median equivalent diameter "
                  f"{d_px:.2f} px = {d_px*sc:.1f} px at level 0 "
                  f"= {d_px*sc*omepx if omepx else float('nan'):.2f} um")
