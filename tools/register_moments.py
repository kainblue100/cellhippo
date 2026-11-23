"""Register by second moments.

Every objective I tried before was gameable: IoU and "fraction on tissue" both
reward shrinking the cell cloud, so both slid to the lower edge of the scale
range.  Moment matching is not gameable -- a cloud's covariance ellipse is a
fixed description of its shape, and matching two ellipses determines scale,
rotation and translation uniquely.

The rotation this recovers is also an independent check on he_imagealignment.csv,
whose linear part says ~90 degrees.
"""
import json
import os

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
os.makedirs(OUT, exist_ok=True)

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}


def moments(pts, w=None):
    w = np.ones(len(pts)) if w is None else w
    w = w / w.sum()
    mu = (pts * w[:, None]).sum(axis=0)
    d = pts - mu
    C = (d * w[:, None]).T @ d
    return mu, C


def similarity_from_moments(mu_s, C_s, mu_t, C_t):
    """Similarity transform mapping source moments onto target moments."""
    vs, Es = np.linalg.eigh(C_s)
    vt, Et = np.linalg.eigh(C_t)
    order_s = np.argsort(-vs)
    order_t = np.argsort(-vt)
    vs, Es = vs[order_s], Es[:, order_s]
    vt, Et = vt[order_t], Et[:, order_t]
    scale = float(np.sqrt(vt.sum() / vs.sum()))
    R = Et @ Es.T
    if np.linalg.det(R) < 0:            # forbid a reflection
        Et[:, -1] *= -1
        R = Et @ Es.T
    t = mu_t - scale * (R @ mu_s)
    return scale, R, t


panels = []
REPORT = {}
for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"{tag}: no H&E")
        continue
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    x = cells.x_centroid.to_numpy()
    y = cells.y_centroid.to_numpy()

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
    gray = rgb.astype(np.float32).mean(axis=2)

    # tissue mask; keep the largest connected component so stray specks and
    # fiducial marks cannot drag the moments
    from scipy import ndimage
    thr = threshold_otsu(gray)
    tm = gray < thr
    lab, n = ndimage.label(tm)
    if n:
        sizes = ndimage.sum(tm, lab, range(1, n + 1))
        tm = lab == (1 + int(np.argmax(sizes)))
    tm = ndimage.binary_fill_holes(ndimage.binary_opening(tm, iterations=2))
    ys, xs = np.nonzero(tm)
    tissue_pts = np.column_stack([xs, ys]).astype(float)

    src = np.column_stack([x, y]).astype(float)
    mu_s, C_s = moments(src)
    mu_t, C_t = moments(tissue_pts)

    scale, R, t = similarity_from_moments(mu_s, C_s, mu_t, C_t)
    print(f"\n=== {tag} L{LEVEL} {W}x{H} tissue {tm.mean()*100:.1f}% ===")
    print(f"  moment match: scale {scale:.4f} px/um at L{LEVEL} "
          f"= {scale*sc:.4f} at L0  ({1/(scale*sc):.4f} um/px)")
    ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    ling = M[:2, :2]
    ang_m = np.degrees(np.arctan2(ling[1, 0], ling[0, 0]))
    print(f"  rotation from moments {ang:+.1f} deg; "
          f"from the matrix {ang_m:+.1f} deg "
          f"(difference {((ang-ang_m+180) % 360)-180:+.1f})")
    print(f"  matrix scale 1.288 px/um vs moment scale {scale*sc:.3f} at L0 "
          f"-> ratio {1.288/(scale*sc):.4f}")

    pts_px = (scale * (R @ src.T)).T + t
    inside = ((pts_px[:, 0] >= 0) & (pts_px[:, 0] < W) &
              (pts_px[:, 1] >= 0) & (pts_px[:, 1] < H))
    on_tissue = tm[pts_px[inside][:, 1].astype(int),
                   pts_px[inside][:, 0].astype(int)].mean()
    print(f"  {inside.mean()*100:.1f}% of cells inside the frame, "
          f"{on_tissue*100:.1f}% of those on tissue")

    REPORT[tag] = {"level": LEVEL, "scale_L": scale, "scale_L0": scale * sc,
                   "um_per_px_L0": 1 / (scale * sc), "rot_deg": float(ang),
                   "matrix_rot_deg": float(ang_m),
                   "inside": float(inside.mean()), "on_tissue": float(on_tissue),
                   "R": R.tolist(), "t": t.tolist(), "image_wh": [W, H]}

    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    step = max(1, int(inside.sum()) // 15000)
    for a, b in zip(pts_px[inside][::step, 0], pts_px[inside][::step, 1]):
        dr.ellipse([a - 3, b - 3, a + 3, b + 3], fill=(255, 0, 0, 200))
    tw = 820
    im = im.resize((tw, int(im.height * tw / im.width)), Image.BILINEAR)
    panels.append((f"{tag}  moment match  on-tissue {on_tissue*100:.0f}%  "
                   f"{scale*sc:.2f} px/um", im))

json.dump(REPORT, open(os.path.join(OUT, "registration_moments.json"), "w"),
          indent=2)
if panels:
    tw = panels[0][1].width
    th = max(p[1].height for p in panels)
    sheet = Image.new("RGB", (len(panels) * (tw + 8) + 8, th + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(panels):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "moment_registration.png"))
    print(f"\nwrote {OUT}/moment_registration.png")
