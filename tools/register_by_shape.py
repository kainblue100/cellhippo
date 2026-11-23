"""Register by shape, not by texture.

Stain-texture correlation gave a weak peak (+0.12) because the translation baked
into he_imagealignment.csv belongs to a different scale.  Shape is a much larger
signal: the Xenium region of interest is one specific blob of tissue, and its
outline must land on one specific blob in the H&E.

So: rasterise the cell cloud as a mask, rasterise the tissue as a mask, and search
a similarity transform (scale, rotation, translation) for maximum IoU.  Coarse
grid, then refine.  The rotation is searched too, so a wrong rotation assumption
cannot hide a right scale.
"""
import os

import numpy as np
import pandas as pd
import tifffile

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

TAGS = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}

NB = 256        # working grid for masks


def tissue_mask_gray(gray):
    """Otsu on the inverted intensity: tissue is darker than glass."""
    from skimage.filters import threshold_otsu
    t = threshold_otsu(gray)
    return gray < t


def rasterize(xs, ys, W, H):
    """Percentile-normalised raster of a point set."""
    g = np.zeros((NB, NB), dtype=np.float32)
    ci = np.clip((xs / W * NB).astype(int), 0, NB - 1)
    ri = np.clip((ys / H * NB).astype(int), 0, NB - 1)
    np.add.at(g, (ri, ci), 1.0)
    return g


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


for tag, (ver, name) in TAGS.items():
    d = os.path.join(BASE, f"{tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    if not os.path.exists(he):
        print(f"\n{tag}: H&E not ready")
        continue
    M = np.loadtxt(os.path.join(d, f"{name}_he_imagealignment.csv"), delimiter=",")
    cells = pd.read_parquet(os.path.join(d, f"{name}_cells.parquet"))
    # subsample: the outline of the point cloud is what matters, not every point
    rs = np.random.default_rng(0)
    idx = rs.choice(len(cells), min(40000, len(cells)), replace=False)
    x = cells.x_centroid.to_numpy()[idx]
    y = cells.y_centroid.to_numpy()[idx]
    x = x - x.mean()
    y = y - y.mean()

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
    LEVEL = max(0, nlev - 4)
    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    rgb = arr[..., :3].astype(np.float32)
    H, W = rgb.shape[:2]
    sc = 2 ** LEVEL
    gray = rgb.mean(axis=2)
    tmask = tissue_mask_gray(gray)
    # downsample the tissue mask to the working grid
    tsmall = np.zeros((NB, NB), dtype=bool)
    bh, bw = H / NB, W / NB
    for i in range(NB):
        for j in range(NB):
            blk = tmask[int(i * bh):int((i + 1) * bh),
                        int(j * bw):int((j + 1) * bw)]
            tsmall[i, j] = blk.mean() > 0.5 if blk.size else False
    print(f"\n=== {tag}  L{LEVEL} {W}x{H}  tissue {tsmall.mean()*100:.1f}% of grid ===")

    # point cloud basis: unit rotation from the matrix, scale free, angle free
    best = None
    rots = np.arange(-180, 181, 5)
    scales = np.arange(0.5, 5.01, 0.1)
    # coarse: no translation, search rotation and scale
    cxs = np.array([0.0])
    coarse = []
    for ang in rots:
        th = np.deg2rad(ang)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for s in scales:
            px = (R @ np.vstack([x, y])).T * s
            xs = px[:, 0] + W / 2
            ys = px[:, 1] + H / 2
            if (xs.min() < -W * 0.5) or (xs.max() > W * 1.5):
                continue
            g = rasterize(xs, ys, W, H)
            m = g > 0
            mf = np.zeros((NB, NB), dtype=bool)
            ys2, xs2 = np.where(m)
            # dilate by 1 cell: use a filled convolution
            from scipy import ndimage
            mf = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
            v = iou(mf, tsmall)
            coarse.append((v, ang, s))
    coarse.sort(key=lambda t: -t[0])
    print("  coarse top:")
    for v, ang, s in coarse[:5]:
        print(f"    IoU {v:.4f}  rot {ang:+4d} deg  scale {s:.2f} px/um")
    v0, ang0, s0 = coarse[0]

    # refine translation at the best rotation/scale
    th = np.deg2rad(ang0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    from scipy import ndimage
    best_t = (v0, 0, 0)
    for dy in range(-30, 31, 3):
        for dx in range(-30, 31, 3):
            px = (R @ np.vstack([x, y])).T * s0
            xs = px[:, 0] + W / 2 + dx * (W / NB)
            ys = px[:, 1] + H / 2 + dy * (H / NB)
            g = rasterize(xs, ys, W, H)
            m = ndimage.binary_fill_holes(ndimage.binary_closing(g > 0, iterations=2))
            v = iou(m, tsmall)
            if v > best_t[0]:
                best_t = (v, dx, dy)
    print(f"  refined: IoU {best_t[0]:.4f}  shift ({best_t[1]},{best_t[2]}) cells")
    print(f"  -> transform: rotate {ang0:+d} deg, scale {s0:.2f} px/um "
          f"(= {1/s0:.4f} um/px), shift ({best_t[1]},{best_t[2]}) grid cells")
    print(f"     OME physical size implies {1/0.2738:.3f} px/um if size is 0.2738")
