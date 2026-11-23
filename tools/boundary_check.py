"""Register cell outlines, not centroids.

A centroid landing on "tissue" proves nothing when 55% of the frame is tissue.
Cell boundary polygons trace real cell shapes, so if a transform is right the
outlines sit on nuclei and follow the visible cell pattern.  That is a signal a
wrong transform cannot fake.

Renders, for a zoomed sub-region, the H&E with the transformed boundary outlines
drawn on top, for each candidate transform.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
from PIL import Image, ImageDraw

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
os.makedirs(OUT, exist_ok=True)

TAG, ver, name = "5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"
d = os.path.join(BASE, f"{TAG}__{ver}__{name}")
he = os.path.join(d, f"{name}_he_image.ome.tif")
M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")

bd = pq.read_table(os.path.join(d, f"{name}_cell_boundaries.parquet"))
print("cell_boundaries columns:", bd.column_names)
bdf = bd.to_pandas()
print(bdf.head(3).to_string())
print("n cells with boundaries:", bdf["cell_id"].nunique(),
      " rows:", len(bdf), " -> long format, one row per vertex")

# group vertices into polygons
polys = {}
gx = bdf.groupby("cell_id")["vertex_x"]
gy = bdf.groupby("cell_id")["vertex_y"]
for cid in gx.groups:
    polys[cid] = np.column_stack([gx.get_group(cid).to_numpy(dtype=float),
                                  gy.get_group(cid).to_numpy(dtype=float)])
print("polygons built:", len(polys),
      " median vertices:", int(np.median([len(v) for v in polys.values()])))

cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))

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
print(f"L{LEVEL} {W}x{H} scale {sc}")


def poly_of(row):
    """Xenium stores boundary vertices as arrays; be liberal about the layout."""
    if "vertex_x" in row.index and "vertex_y" in row.index:
        return np.column_stack([np.asarray(row["vertex_x"], dtype=float),
                                np.asarray(row["vertex_y"], dtype=float)])
    for cx, cy in (("x", "y"), ("boundary_x", "boundary_y")):
        if cx in row.index:
            return np.column_stack([np.asarray(row[cx], dtype=float),
                                   np.asarray(row[cy], dtype=float)])
    return None


# pick cells near a reference point so the crop has some cell density
inv = np.linalg.inv(M)
cands = {}

for label, S in (("M_direct", M), ("M_inverse", inv)):
    # choose the crop centre from the mapped centroid distribution
    pts = np.column_stack([cells.x_centroid.to_numpy(),
                           cells.y_centroid.to_numpy(),
                           np.ones(len(cells))])
    px = (S @ pts.T).T
    cc, rr = px[:, 0] / sc, px[:, 1] / sc
    med_c, med_r = np.median(cc), np.median(rr)
    cands[label] = (S, med_c, med_r)
    print(f"  {label}: mapped median ({med_c:.0f},{med_r:.0f})")

PS = 900
sheet_panels = []

for label, (S, mc, mr) in cands.items():
    x0 = int(min(max(mc - PS / 2, 0), max(W - PS, 0)))
    y0 = int(min(max(mr - PS / 2, 0), max(H - PS, 0)))
    crop = Image.fromarray(arr[y0:y0 + PS, x0:x0 + PS]).convert("RGB")
    dr = ImageDraw.Draw(crop, "RGBA")
    drawn = 0
    for cid, p in polys.items():
        if len(p) < 3:
            continue
        pp = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
        q = (S @ pp.T).T
        xs = q[:, 0] / sc - x0
        ys = q[:, 1] / sc - y0
        if xs.max() < 0 or xs.min() > PS or ys.max() < 0 or ys.min() > PS:
            continue
        dr.polygon(list(zip(xs, ys)), outline=(0, 255, 255, 255))
        drawn += 1
        if drawn > 2000:
            break
    print(f"  {label}: drew {drawn} outlines in crop at ({x0},{y0})")
    sheet_panels.append((f"{label}  {drawn} outlines  crop ({x0},{y0})", crop))

if sheet_panels:
    tw = 900
    sheet = Image.new("RGB", (len(sheet_panels) * (tw + 8) + 8, tw + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(sheet_panels):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "boundary_check.png"))
    print(f"wrote {OUT}/boundary_check.png")
