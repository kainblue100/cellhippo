"""Fix rotation and scale from independent evidence, then search translation only.

Rotation: the alignment matrix's off-diagonal terms dominate, so the transform is
a ~90 degree rotation.  A free IoU search agreed (+80 deg on 5k, -85 deg on lc).

Scale: measured, not assumed.  cells.parquet cell_area gives a cell diameter in
microns; the H&E's own nuclei give one in pixels; at the OME pixel size those two
agree to 6% on lc, and 1.288 px/um would imply a 34 um cell, which is not a lung
cell.  So scale = 1 / PhysicalSizeX.

With shape and size fixed, "shrink to fit" is no longer available, and maximizing
the fraction of the cell cloud that lands on tissue is a well-posed objective.
"""
import os
import re

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
os.makedirs(OUT, exist_ok=True)

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}

NB = 256
RES = {}


def to_grid(xs, ys, W, H, dilate=1, fill=True):
    g = np.zeros((NB, NB), dtype=bool)
    ci = np.clip((xs / W * NB).astype(int), 0, NB - 1)
    ri = np.clip((ys / H * NB).astype(int), 0, NB - 1)
    g[ri, ci] = True
    if dilate:
        g = ndimage.binary_dilation(g, iterations=dilate)
    if fill:
        g = ndimage.binary_fill_holes(g)
    return g


for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not ready")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
        ome = t.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
    omepx = float(m.group(1)) if m else 0.2738

    LEVEL = max(0, nlev - 4)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    rgb = arr[..., :3].astype(np.uint8)
    H, W = rgb.shape[:2]
    sc = 2 ** LEVEL
    gray = rgb.astype(np.float32).mean(axis=2)

    from skimage.filters import threshold_otsu
    thr = threshold_otsu(gray)
    tmask_full = gray < thr
    # downsample by reshaping into blocks
    bh, bw = H // NB, W // NB
    tmask = tmask_full[:bh * NB, :bw * NB].reshape(NB, bh, NB, bw).mean((1, 3)) > 0.5
    print(f"\n=== {tag} L{LEVEL} {W}x{H}  otsu {thr:.0f}  "
          f"tissue {tmask.mean()*100:.1f}%  scale {1/omepx:.3f} px/um ===")

    # rotation, in level-0 px per um, from the matrix's linear part direction
    lin = M[:2, :2]
    s_mat = np.hypot(lin[0, 0], lin[1, 0])
    rot = lin / s_mat                       # unit rotation
    s_true = (1.0 / omepx) / sc             # px per um, at this level
    print(f"  matrix rotation det {np.linalg.det(rot):+.3f}  "
          f"matrix scale {s_mat:.3f} -> using {s_true:.3f} px/um at L{LEVEL}")

    x = cells.x_centroid.to_numpy()
    y = cells.y_centroid.to_numpy()
    # the search only needs the cloud's shape, not every point; 25k is plenty and
    # keeps the grid sweep to minutes rather than an hour
    rs = np.random.default_rng(0)
    ix = rs.choice(len(x), min(25000, len(x)), replace=False)
    base = (rot @ np.vstack([x[ix], y[ix]])).T
    base -= base.mean(axis=0)

    def score(s_um, dx, dy):
        """Fraction of the cell cloud that lands on tissue, with points outside
        the frame counting against it.  No hard failure, so the search can move
        through placements where the cloud overhangs the edge -- which it does at
        the OME scale, suggesting that scale is a few per cent too large."""
        px = base * s_um
        xs = px[:, 0] + W / 2 + dx * (W / NB)
        ys = px[:, 1] + H / 2 + dy * (H / NB)
        inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if inside.mean() < 0.5:
            return -1.0
        g = to_grid(xs[inside], ys[inside], W, H)
        return float(tmask[g].mean() * inside.mean())

    # narrow scale band: OME value and a few per cent below it
    coarse = None
    for s_um in np.arange(s_true * 0.86, s_true * 1.02, s_true * 0.01):
        for dy in range(-70, 71, 6):
            for dx in range(-70, 71, 6):
                v = score(s_um, dx, dy)
                if coarse is None or v > coarse[0]:
                    coarse = (v, s_um, dx, dy)
    best = coarse
    for step in (3, 1):
        b0 = best
        for s_um in np.arange(b0[1] - s_true * 0.012, b0[1] + s_true * 0.012,
                              s_true * 0.003):
            for dy in range(b0[3] - 4, b0[3] + 5, step):
                for dx in range(b0[2] - 4, b0[2] + 5, step):
                    v = score(s_um, dx, dy)
                    if v > best[0]:
                        best = (v, s_um, dx, dy)
    v, s_best, dx, dy = best
    print(f"  best: {v*100:.2f}% of cloud on tissue at scale {s_best:.4f} px/um "
          f"({1/s_best*sc:.4f} um/px at L{LEVEL}, OME says {omepx:.4f}) "
          f"shift ({dx},{dy})")
    print(f"  scale vs OME: {s_best*sc:.4f} vs {1/omepx:.4f} px/um "
          f"-> ratio {s_best*sc*omepx:.4f}")
    # null: random shifts of the same cloud at the chosen scale
    rng = np.random.default_rng(0)
    nulls = [score(s_best, int(a), int(b))
             for a, b in rng.integers(-70, 71, (40, 2))]
    nulls = [x for x in nulls if x >= 0]
    if nulls:
        print(f"  shuffled shifts: mean {np.mean(nulls)*100:.2f}%  "
              f"max {np.max(nulls)*100:.2f}%  (n={len(nulls)})")

    RES[tag] = {"scale_px_per_um_L": s_best, "scale_px_per_um_L0": s_best * sc,
                "ome_px_per_um": 1 / omepx, "rot": rot.tolist(),
                "shift_grid": [dx, dy], "score": v,
                "null_mean": float(np.mean(nulls)) if nulls else None,
                "level": LEVEL, "omepx": omepx}

    # visual
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    px = base * s_best
    xs = px[:, 0] + W / 2 + dx * (W / NB)
    ys = px[:, 1] + H / 2 + dy * (H / NB)
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    step = max(1, int(inside.sum()) // 12000)
    for a, b in zip(xs[inside][::step], ys[inside][::step]):
        dr.ellipse([a - 3, b - 3, a + 3, b + 3], fill=(255, 0, 0, 200))
    tw = 900
    im = im.resize((tw, int(im.height * tw / im.width)), Image.BILINEAR)
    im.save(os.path.join(OUT, f"reg_{tag}.png"))
    print(f"  wrote figures/reg_{tag}.png")

import json
json.dump(RES, open(os.path.join(OUT, "registration.json"), "w"), indent=2)
