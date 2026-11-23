"""Settle the transform by shape, not by scalars.

Render the mapped cell centroids as a point cloud on top of the H&E image.  A
correct transform makes the cloud cover the tissue and only the tissue; a wrong
one scatters points onto glass or compresses them into part of the section.
This is decisive in a way that per-crop intensity is not, because it uses the
whole slide at once.
"""
import os

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
os.makedirs(OUT, exist_ok=True)

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}

CAND_PANELS = []
for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"{tag}: no H&E yet")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    x = cells.x_centroid.to_numpy()
    y = cells.y_centroid.to_numpy()
    pts = np.column_stack([x, y, np.ones(len(x))])

    # a small, readable level
    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
    LEVEL = max(0, nlev - 5)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    H, W = arr.shape[:2]
    sc = 2 ** LEVEL
    print(f"{tag}: L{LEVEL} {W}x{H} scale {sc}")

    scale_given = np.hypot(M[0, 0], M[1, 0])

    variants = {
        "M_direct": M,
        "M_inverse": np.linalg.inv(M),
    }
    for label, S in variants.items():
        px = (S @ pts.T).T
        col = px[:, 0] / sc
        row = px[:, 1] / sc
        inside = ((col >= 0) & (col < W) & (row >= 0) & (row < H))
        base = Image.fromarray(arr).convert("RGB")
        dr = ImageDraw.Draw(base)
        cc = col[inside]
        rr = row[inside]
        # subsample so the dots make a density map rather than a solid blob
        step = max(1, len(cc) // 40000)
        for a, b in zip(cc[::step], rr[::step]):
            dr.point((a, b), fill=(255, 0, 0))
            dr.point((a + 1, b), fill=(255, 0, 0))
        head = f"{tag}  {label}  inside {inside.mean()*100:.1f}%"
        dr.rectangle([0, 0, W, 16], fill=(255, 255, 255))
        dr.text((4, 3), head, fill=(0, 0, 0))
        CAND_PANELS.append((head, base))
        print(f"   {label:10s} inside {inside.mean()*100:6.2f}%")

if CAND_PANELS:
    cols = 2
    rows = (len(CAND_PANELS) + cols - 1) // cols
    tw = 760
    panels = []
    for h, im in CAND_PANELS:
        r = tw / im.width
        panels.append((h, im.resize((tw, max(1, int(im.height * r))),
                                    Image.BILINEAR)))
    th = max(p[1].height for p in panels)
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 24) + 8),
                      "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(panels):
        rr, cc = divmod(i, cols)
        x0 = 8 + cc * (tw + 8)
        y0 = 8 + rr * (th + 24)
        dr.text((x0 + 2, y0 + 4), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, y0 + 20))
    sheet.save(os.path.join(OUT, "cloud_overlay.png"))
    print(f"\nwrote {OUT}/cloud_overlay.png")
