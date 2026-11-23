"""Score each candidate transform by where the cells land.

Build a tissue mask from the H&E, then measure the fraction of mapped cell
centroids that fall on tissue.  Cells are inside cells; if a transform is right,
essentially every centroid lands on stained tissue.  If it is wrong, the same
fraction collapses toward the tissue fraction of the whole image, which is what
"randomly placed" would give.

The null is explicit: shuffle the point positions and remeasure.
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


def tissue_mask(rgb):
    """Tissue is anything that is not near-white glass."""
    g = rgb.astype(np.float32)
    mx = g.max(axis=2)
    mn = g.min(axis=2)
    sat = mx - mn
    dark = mx < 235
    return (sat > 18) | dark


panels = []
for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"{tag}: H&E not ready")
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
    mask = tissue_mask(arr)
    base_rate = mask.mean()
    print(f"\n=== {tag}  L{LEVEL} {W}x{H}  tissue {base_rate*100:.1f}% of frame ===")

    inv = np.linalg.inv(M)
    side = np.hypot(M[0, 0], M[1, 0])
    cands = {
        "M direct (1.288 px/um)": M,
        "M inverse": inv,
    }
    rows = []
    for label, S in cands.items():
        px = (S @ pts.T).T
        c = px[:, 0] / sc
        r = px[:, 1] / sc
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        ii = np.where(ok)[0]
        cc = c[ii].astype(int)
        rr = r[ii].astype(int)
        on = mask[rr, cc].mean()
        # null: random positions inside the image
        rng = np.random.default_rng(0)
        nr = rng.integers(0, H, len(rr))
        nc = rng.integers(0, W, len(cc))
        null = mask[nr, nc].mean()
        # a second null that is fair: random points inside the mapped bbox
        b0, b1 = cc.min(), cc.max()
        r0, r1 = rr.min(), rr.max()
        nc2 = rng.integers(b0, b1 + 1, len(cc))
        nr2 = rng.integers(r0, r1 + 1, len(cc))
        null_bbox = mask[nr2, nc2].mean()
        rows.append((label, ok.mean(), on, null, null_bbox))
        print(f"  {label:24s} inside {ok.mean()*100:6.2f}%  "
              f"on-tissue {on*100:6.2f}%  "
              f"(random-in-image {null*100:5.2f}%, random-in-bbox {null_bbox*100:5.2f}%)")

    best = max(rows, key=lambda x: x[2])
    print(f"  -> best: {best[0]}")

    # a legible visual for the best one
    Sc = dict(cands)[best[0]]
    px = (Sc @ pts.T).T
    c = px[:, 0] / sc
    r = px[:, 1] / sc
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    im = Image.fromarray(arr).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    step = max(1, int(ok.sum()) // 15000)
    for a, b in zip(c[ok][::step], r[ok][::step]):
        dr.ellipse([a - 2, b - 2, a + 2, b + 2], fill=(255, 0, 0, 190))
    tw = 900
    im = im.resize((tw, int(im.height * tw / im.width)), Image.BILINEAR)
    panels.append((f"{tag}  best={best[0]}  on-tissue {best[2]*100:.1f}% "
                   f"vs random {best[3]*100:.1f}%", im))

if panels:
    tw = panels[0][1].width
    th = max(p[1].height for p in panels)
    sheet = Image.new("RGB", (len(panels) * (tw + 8) + 8, th + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(panels):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "cloud_scored.png"))
    print(f"\nwrote {OUT}/cloud_scored.png")
