"""Fix the tissue mask, which has been the bottleneck all along.

Otsu on the whole frame found only 30% tissue where the eye sees 50-60%, because
pale stroma sits close to the glass intensity.  Stained tissue is not merely
dark, it is *coloured*: haematoxylin and eosin both put saturation into pixels
that glass leaves near zero.  So threshold saturation, not luminance, and keep
every component rather than the largest one.

Then re-run the moment registration and re-measure containment; with a correct
mask the numbers finally mean something.
"""
import json
import os

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


def tissue_mask(rgb):
    """Coloured-or-dark, with holes filled and small debris dropped."""
    a = rgb.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    # glass: bright AND unsaturated
    glass = (mx > 215) & (sat < 0.06)
    m = ~glass
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=6)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.where(sizes > 0.0005 * m.size)[0] + 1
        m = np.isin(lab, keep)
    return m


def moments(pts):
    mu = pts.mean(axis=0)
    d = pts - mu
    return mu, (d.T @ d) / len(pts)


def similarity_from_moments(mu_s, C_s, mu_t, C_t):
    vs, Es = np.linalg.eigh(C_s)
    vt, Et = np.linalg.eigh(C_t)
    os_ = np.argsort(-vs); ot = np.argsort(-vt)
    vs, Es = vs[os_], Es[:, os_]
    vt, Et = vt[ot], Et[:, ot]
    scale = float(np.sqrt(vt.sum() / vs.sum()))
    R = Et @ Es.T
    if np.linalg.det(R) < 0:
        Et[:, -1] *= -1
        R = Et @ Es.T
    return scale, R, mu_t - scale * (R @ mu_s)


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

    tm = tissue_mask(rgb)
    ys, xs = np.nonzero(tm)
    print(f"\n=== {tag} L{LEVEL} {W}x{H} ===")
    print(f"  tissue mask (saturation rule): {tm.mean()*100:.1f}% of frame")

    src = np.column_stack([cells.x_centroid.to_numpy(),
                           cells.y_centroid.to_numpy()]).astype(float)
    mu_s, C_s = moments(src)
    mu_t, C_t = moments(np.column_stack([xs, ys]).astype(float))
    scale, R, t = similarity_from_moments(mu_s, C_s, mu_t, C_t)

    ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    ang_m = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    print(f"  moment scale {scale*sc:.4f} px/um at L0 ({1/(scale*sc):.4f} um/px)")
    print(f"  rotation moments {ang:+.1f}  matrix {ang_m:+.1f}  "
          f"diff {((ang-ang_m+180)%360)-180:+.1f} deg")

    pts = (scale * (R @ src.T)).T + t
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
              (pts[:, 1] >= 0) & (pts[:, 1] < H))
    ii = np.where(inside)[0]
    on = tm[pts[ii, 1].astype(int), pts[ii, 0].astype(int)]
    rate = on.mean()
    # null: random points inside the frame, and random points inside the tissue
    rng = np.random.default_rng(0)
    null = tm[rng.integers(0, H, len(ii)), rng.integers(0, W, len(ii))].mean()
    print(f"  {inside.mean()*100:.1f}% inside; on tissue {rate*100:.2f}% "
          f"vs random-in-frame {null*100:.2f}% "
          f"(n={len(ii):,})  enrichment x{rate/null:.2f}")

    REPORT[tag] = {"level": LEVEL, "scale_L0": scale * sc,
                   "um_per_px": 1 / (scale * sc), "rot_deg": float(ang),
                   "matrix_rot_deg": float(ang_m), "on_tissue": float(rate),
                   "random_in_frame": float(null),
                   "tissue_frac": float(tm.mean()),
                   "R": R.tolist(), "t": t.tolist(), "image_wh": [W, H]}

    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    step = max(1, int(inside.sum()) // 25000)
    for a, b in zip(pts[ii][::step, 0], pts[ii][::step, 1]):
        dr.ellipse([a - 2, b - 2, a + 2, b + 2], fill=(255, 0, 0, 230))
    tw = 820
    im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    panels.append((f"{tag}  moment  on-tissue {rate*100:.0f}% "
                   f"(random {null*100:.0f}%)  {scale*sc:.2f} px/um", im))

json.dump(REPORT, open(os.path.join(OUT, "registration_final.json"), "w"), indent=2)
if panels:
    tw = panels[0][1].width
    th = max(p[1].height for p in panels)
    sheet = Image.new("RGB", (len(panels) * (tw + 8) + 8, th + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (h, im) in enumerate(panels):
        x0 = 8 + i * (tw + 8)
        dr.text((x0 + 2, 5), h, fill=(0, 0, 0))
        sheet.paste(im, (x0, 22))
    sheet.save(os.path.join(OUT, "registration_final.png"))
    print(f"\nwrote {OUT}/registration_final.png")
