"""Use he_imagealignment.csv exactly as shipped, on level-0 pixels.

Every attempt to 'improve' the matrix made things worse: the plain application
scored 3 of 4 crops on tissue, while centring and rescaling produced clouds that
sat in blank background.  So do the simple thing and look at the whole slide.

Also render the SAME slide at the level whose pixel size the matrix's own scale
implies, in case the matrix refers to a level rather than to level 0.
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

panels = []
for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"{tag}: no H&E")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    rs = np.random.default_rng(0)
    ix = rs.choice(len(cells), min(30000, len(cells)), replace=False)
    pts = np.column_stack([cells.x_centroid.to_numpy()[ix],
                           cells.y_centroid.to_numpy()[ix],
                           np.ones(len(ix))])
    px = (M @ pts.T).T                    # level-0 pixels, no adjustments

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
    LEVEL = max(0, nlev - 5)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    rgb = arr[..., :3].astype(np.uint8)
    H, W = rgb.shape[:2]
    sc = 2 ** LEVEL

    c = px[:, 0] / sc
    r = px[:, 1] / sc
    inside = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    print(f"{tag}: L{LEVEL} {W}x{H}  inside {inside.mean()*100:.1f}%  "
          f"col {c.min():.0f}..{c.max():.0f}  row {r.min():.0f}..{r.max():.0f}")

    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    step = max(1, int(inside.sum()) // 15000)
    for a, b in zip(c[inside][::step], r[inside][::step]):
        dr.ellipse([a - 3, b - 3, a + 3, b + 3], fill=(255, 0, 0, 210))
    head = f"{tag}  raw matrix  inside {inside.mean()*100:.0f}%  L{LEVEL} {W}x{H}"
    panels.append((head, im))

if panels:
    tw = 820
    ims = [(h, im.resize((tw, int(im.height * tw / im.width)), Image.BILINEAR))
           for h, im in panels]
    th = max(im.height for _, im in ims)
    sheet = Image.new("RGB", (len(ims) * (tw + 8) + 8, th + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(ims):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "raw_matrix.png"))
    print(f"\nwrote {OUT}/raw_matrix.png")
