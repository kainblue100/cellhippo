"""Render the registration as a coverage tint, not as specks.

Single-pixel dots at slide scale are unresolvable, which is why the previous
panels could not be judged.  A density raster drawn as a translucent overlay
shows coverage as a continuous wash, and the tissue outline is visible through it,
so agreement or mismatch is obvious at a glance.

Also renders the tissue mask itself alongside, so a bad mask cannot be mistaken
for a bad transform.
"""
import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"

TAGS = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
        "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE")}

REG = json.load(open(os.path.join(OUT, "registration_final.json")))


def tissue_mask(rgb):
    a = rgb.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    m = ~((mx > 215) & (sat < 0.06))
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=6)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.where(sizes > 0.0005 * m.size)[0] + 1)
    return m


panels = []
for tag in TAGS:
    if tag not in REG:
        continue
    ver, name = TAGS[tag]
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    r = REG[tag]
    LEVEL = r["level"]
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    rgb = arr[..., :3].astype(np.uint8)
    H, W = rgb.shape[:2]

    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    R = np.array(r["R"]); t_ = np.array(r["t"]); s = r["scale_L0"] / (2 ** LEVEL)
    src = np.column_stack([cells.x_centroid.to_numpy(),
                           cells.y_centroid.to_numpy()]).astype(float)
    pts = (s * (R @ src.T)).T + t_

    # density raster, smoothed, then used as the alpha of a red tint
    dens = np.zeros((H, W), dtype=np.float32)
    ok = ((pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H))
    xi = pts[ok, 0].astype(int)
    yi = pts[ok, 1].astype(int)
    np.add.at(dens, (yi, xi), 1.0)
    dens = ndimage.gaussian_filter(dens, 6)
    dmax = np.percentile(dens[dens > 0], 99) if (dens > 0).any() else 1.0
    alpha = np.clip(dens / max(dmax, 1e-6), 0, 1) ** 0.6

    out = rgb.astype(np.float32).copy()
    red = np.array([255, 30, 30], dtype=np.float32)
    a3 = alpha[..., None] * 0.62
    out = out * (1 - a3) + red * a3
    out = out.astype(np.uint8)

    from PIL import Image, ImageDraw
    im = Image.fromarray(out).convert("RGB")
    tm = tissue_mask(rgb)
    # outline of the tissue mask, so mask error is visible too
    edge = tm ^ ndimage.binary_erosion(tm, iterations=3)
    ea = np.array(im)
    ea[edge] = [0, 200, 255]
    im = Image.fromarray(ea)
    dr = ImageDraw.Draw(im)
    tw = 760
    im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    panels.append((f"{tag}  red = mapped cell density    cyan = tissue mask edge",
                   im))

if panels:
    tw = panels[0][1].width
    th = max(p[1].height for p in panels)
    sheet = Image.new("RGB", (len(panels) * (tw + 8) + 8, th + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(panels):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "registration_coverage.png"))
    print(f"wrote {OUT}/registration_coverage.png")
