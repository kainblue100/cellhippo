"""Cut HIPPO-style tiles and fuse three modalities per tile.

Imaging conventions are taken from the reference implementations rather than
invented:

  HIPOS  (the paper's own code)
    224x224 px tiles at roughly 5x magnification, non-overlapping
    Otsu threshold, keep tiles with >= 60% tissue coverage
    Reinhard stain normalisation
  GHIST+
    expects an aligned H&E plus a nuclei segmentation mask and per-nucleus
    metadata alongside the expression matrix -- the same three inputs used here

Registration (see tools/register_final.py): rotation from
he_imagealignment.csv, scale from the OME PhysicalSizeX, translation from second
moment matching.  The three agree with each other and the overlay was checked.

Per tile the three modalities are kept separate so each can be ablated later:
  img   224x224x3 uint8 H&E, Reinhard-normalised
  expr  mean of the log1p(CP10k)+PCA values of the cells in the tile
  ann   fraction of the tile's cells in each annotated cell type
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"

MAP = {
    "5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
}
TILE = 224


# ---------------------------------------------------------------- stain norm
def reinhard(rgb, src_mean, src_std, tgt_mean, tgt_std):
    """Reinhard normalisation in LAB, done properly.

    The first version z-scored the a and b channels tile by tile, which sets each
    tile's colour mean to zero -- it strips the stain out of the picture, and the
    gallery came back looking black and white.  The real method is a per-channel
    affine map from the *slide's* statistics to the reference's:
        out = (x - mean_src) / std_src * std_tgt + mean_tgt

    Source statistics come from the whole slide, so relative colour within a slide
    is preserved; only the slide-to-slide offset is removed.  That is the point --
    normalising each slide to itself would erase the differences the clustering is
    meant to find.
    """
    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(rgb / 255.0)
    out = np.empty_like(lab)
    for c in range(3):
        out[..., c] = ((lab[..., c] - src_mean[c]) /
                       max(src_std[c], 1e-6) * tgt_std[c] + tgt_mean[c])
    return (np.clip(lab2rgb(out), 0, 1) * 255).astype(np.uint8)


def tissue_fraction(rgb_tile):
    """Stained-tissue area fraction.

    Otsu on luminance was the first attempt and it is the wrong tool here: on a
    tile that is almost entirely tissue it still splits the histogram near the
    middle and reports ~50%, so a 60% filter rejects nearly everything.  Stained
    tissue is characterised by colour, not darkness -- haematoxylin and eosin
    both put saturation into pixels that glass leaves grey.  Same rule that fixed
    the whole-slide mask.
    """
    a = rgb_tile.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    stained = ~((mx > 215) & (sat < 0.06))
    return float(stained.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--level", type=int, default=None,
                    help="pyramid level to tile; default picks ~1 um/px")
    ap.add_argument("--tissue-min", type=float, default=0.60)
    ap.add_argument("--min-cells", type=int, default=10)
    a = ap.parse_args()

    ver, name = MAP[a.tag]
    d = os.path.join(TENX, f"{a.tag}__{ver}__{name}")
    he = os.path.join(d, f"{name}_he_image.ome.tif")
    cells_pq = os.path.join(d, f"{name}_cells.parquet")
    full_pq = os.path.join(WORK, f"{a.tag}_full.parquet")
    reg = json.load(open(os.path.join(FIGS, "registration_final.json")))[a.tag]

    with tifffile.TiffFile(he) as t:
        nlev = len(t.series[0].levels)
        shapes = [lv.shape for lv in t.series[0].levels]
    LEVEL = reg["level"] if a.level is None else a.level
    sc = 2 ** LEVEL
    um_per_px = reg["um_per_px"] * sc
    print(f"=== {a.tag} ===  level {LEVEL} of {nlev}  {shapes[LEVEL]}  "
          f"{um_per_px:.3f} um/px  tile {TILE}px = {TILE*um_per_px:.0f} um")

    with tifffile.TiffFile(he) as t:
        arr = t.series[0].levels[LEVEL].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    rgb = arr[..., :3].astype(np.uint8)
    H, W = rgb.shape[:2]
    print(f"  frame {W}x{H}  ({rgb.nbytes/1e6:.0f} MB)")

    # register cells into this level's pixel frame
    R = np.array(reg["R"]); tvec = np.array(reg["t"])
    s_lvl = reg["scale_L0"] / sc
    df = pd.read_parquet(full_pq)
    src = np.column_stack([df["x"].to_numpy(), df["y"].to_numpy()]).astype(float)
    pts = (s_lvl * (R @ src.T)).T + tvec
    col, row = pts[:, 0], pts[:, 1]
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    print(f"  {inside.mean()*100:.2f}% of cells land in the frame")
    assert inside.mean() > 0.85, "registration disagrees with this level"

    # grid
    tc = (col // TILE).astype(int)
    tr = (row // TILE).astype(int)
    key = tr * 100000 + tc
    uniq, inv = np.unique(key, return_inverse=True)
    n_tiles_all = len(uniq)

    pcs = [c for c in df.columns if c.startswith("PC")]
    anns = sorted(df["annotation"].astype(str).unique())
    ann_i = {v: i for i, v in enumerate(anns)}
    aidx = df["annotation"].astype(str).map(ann_i).to_numpy()
    E = df[pcs].to_numpy(dtype=np.float32)

    # pass 1: tissue filter and gather the tiles that survive
    keep, counts, imgs = [], [], []
    for u in range(n_tiles_all):
        sel = inv == u
        n_cells = int(sel.sum())
        if n_cells < a.min_cells:
            continue
        r, c = divmod(int(uniq[u]), 100000)
        y0, x0 = r * TILE, c * TILE
        if y0 + TILE > H or x0 + TILE > W:
            continue
        tile = rgb[y0:y0 + TILE, x0:x0 + TILE]
        if tissue_fraction(tile) < a.tissue_min:
            continue
        keep.append(u)
        counts.append(n_cells)
        imgs.append(tile)
    n = len(keep)
    print(f"  tiles: {n_tiles_all} on the cell bounding box, {n} pass the "
          f">= {a.tissue_min:.0%} tissue and >= {a.min_cells}-cell filters")

    # Reinhard statistics.  Source: this slide, measured over the kept tiles and
    # pooled.  Target: a fixed reference, taken from the 5k slide on the first
    # run and reused, so every slide is pulled to the same place instead of each
    # being normalised to itself.
    from skimage.color import rgb2lab
    labs = np.stack([rgb2lab(t / 255.0) for t in imgs[:: max(1, n // 300)][:300]])
    src_mean = [float(labs[..., c].mean()) for c in range(3)]
    src_std = [float(labs[..., c].std()) for c in range(3)]
    ref_path = os.path.join(WORK, "reinhard_reference.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        tgt_mean, tgt_std = ref["mean"], ref["std"]
        print(f"  Reinhard target: loaded reference (L mean {tgt_mean[0]:.2f})")
    else:
        tgt_mean, tgt_std = src_mean, src_std
        json.dump({"mean": tgt_mean, "std": tgt_std}, open(ref_path, "w"),
                  indent=2)
        print(f"  Reinhard target: this slide becomes the reference "
              f"(L mean {tgt_mean[0]:.2f} sd {tgt_std[0]:.2f})")
    print(f"  this slide: L mean {src_mean[0]:.2f} sd {src_std[0]:.2f}")

    Ximg = np.zeros((n, TILE, TILE, 3), dtype=np.uint8)
    Xexpr = np.zeros((n, len(pcs)), dtype=np.float32)
    Xann = np.zeros((n, len(anns)), dtype=np.float32)
    meta = np.zeros((n, 4), dtype=np.int64)
    remap = {u: i for i, u in enumerate(keep)}

    for u, i in remap.items():
        sel = inv == u
        Xexpr[i] = E[sel].mean(axis=0)
        cnt = np.bincount(aidx[sel], minlength=len(anns)).astype(np.float32)
        Xann[i] = cnt / cnt.sum()
        r, c = divmod(int(uniq[u]), 100000)
        meta[i] = (c, r, int(sel.sum()), int(uniq[u]))
        Ximg[i] = reinhard(imgs[i], src_mean, src_std, tgt_mean, tgt_std)

    out = os.path.join(WORK, f"{a.tag}_tiles.npz")
    np.savez_compressed(out, img=Ximg, expr=Xexpr, ann=Xann, meta=meta,
                        annotations=np.array(anns), pc_names=np.array(pcs),
                        um_per_px=np.array([um_per_px]), tile_px=np.array([TILE]),
                        level=np.array([LEVEL]))
    print(f"  wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)  "
          f"img {Ximg.shape} expr {Xexpr.shape} ann {Xann.shape}")

    json.dump({"tag": a.tag, "level": LEVEL, "um_per_px": um_per_px,
               "tile_px": TILE, "n_tiles": n, "n_tiles_all": n_tiles_all,
               "tissue_min": a.tissue_min, "min_cells": a.min_cells,
               "reinhard_src_mean": src_mean, "reinhard_src_std": src_std,
               "reinhard_tgt_mean": tgt_mean, "reinhard_tgt_std": tgt_std,
               "image_shape": [H, W], "cells_in_frame": float(inside.mean()),
               "annotations": anns},
              open(os.path.join(WORK, f"{a.tag}_tiles_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
