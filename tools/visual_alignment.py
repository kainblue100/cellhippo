"""Look at the pixels. Crop patches at mapped cell positions under each
candidate transform and save a contact sheet, so the correct convention is
decided by eye rather than by an intensity summary that both hypotheses pass.

A correct mapping puts each crop on stained tissue.  A wrong one lands it on
white glass, and that is obvious in a picture.
"""
import os

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
os.makedirs(OUT, exist_ok=True)

TAG, ver, name = "5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"
d = os.path.join(BASE, f"{TAG}__{ver}__{name}")
he = os.path.join(d, f"{name}_he_image.ome.tif")
M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))

import re
with tifffile.TiffFile(he) as t:
    ome = t.ome_metadata or ""
um = float(re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome).group(1))

# work at L2 (~1.1 um/px) for a readable crop
LEVEL = 2
with tifffile.TiffFile(he) as t:
    arr = t.series[0].levels[LEVEL].asarray()
if arr.ndim == 3 and arr.shape[0] == 3:
    arr = np.moveaxis(arr, 0, -1)
if arr.shape[-1] == 4:
    arr = arr[..., :3]
H, W = arr.shape[:2]
sc = 2 ** LEVEL
print(f"level {LEVEL}: {W}x{H}, scale {sc}, ~{um*sc:.3f} um/px")

rng = np.random.default_rng(0)
ix = rng.choice(len(cells), 4, replace=False)
sel = cells.iloc[ix]
print(sel[["cell_id", "x_centroid", "y_centroid"]].to_string())

scaleA = np.hypot(M[0, 0], M[1, 0])                 # px per um as given
scaleB = 1.0 / um
CANDS = {
    "A_raw_1.288pxum": M,
    "B_phys_3.653pxum": M @ np.diag([scaleB / scaleA, scaleB / scaleA, 1.0]),
}

PS = 320
tiles = []
labels = []
for label, S in CANDS.items():
    pts = np.column_stack([sel.x_centroid.to_numpy(), sel.y_centroid.to_numpy(),
                           np.ones(len(sel))])
    px = (S @ pts.T).T
    for k in range(len(sel)):
        c = int(px[k, 0] / sc)
        r = int(px[k, 1] / sc)
        y0 = min(max(r - PS // 2, 0), max(H - PS, 0))
        x0 = min(max(c - PS // 2, 0), max(W - PS, 0))
        tiles.append(Image.fromarray(arr[y0:y0 + PS, x0:x0 + PS]))
        labels.append(f"{label}\ncell {sel.cell_id.iloc[k]}\npx ({c},{r})")

pad = 8
sheet = Image.new("RGB", (4 * (PS + pad) + pad, 2 * (PS + 46) + pad), "white")
dr = ImageDraw.Draw(sheet)
for i, (im, lab) in enumerate(zip(tiles, labels)):
    row, col = divmod(i, 4)
    x = pad + col * (PS + pad)
    y = pad + row * (PS + 46)
    sheet.paste(im, (x, y + 42))
    dr.text((x + 2, y + 4), lab.replace("\n", " | ", 1), fill=(20, 20, 20))
    dr.text((x + 2, y + 20), lab.split("\n")[2], fill=(90, 90, 90))
sheet.save(os.path.join(OUT, "alignment_check.png"))
print(f"\nwrote {OUT}/alignment_check.png")
print("top row = hypothesis A (raw 1.288 px/um), bottom = B (OME 3.653 px/um)")
