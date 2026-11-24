"""Cut tiles for the 17-sample Takano cohort.

Differences from extract_tiles.py, and why:

  image source   the Xenium morphology MIP, not H&E.  These 17 samples ship no
                 he_imagealignment.csv; the H&E tiffs have no OME metadata and
                 their implied scale is 1.383 x 1.335 px/um, 3.5% anisotropic.
                 The morphology stack is in the cell coordinate frame by
                 construction at 0.2125 um/px, so nothing has to be registered.
                 It is a single grayscale channel, replicated to three for the
                 encoder.

  level          level 2 of the morphology pyramid, 0.85 um/px, so a 224 px tile
                 is 190 um -- comparable to the 243 um tiles of the 5k slide.

  tissue         uint16 intensity, not RGB saturation.  Otsu on the level-2 image
                 gives the threshold; the glass is the low-intensity mode.

  annotation     12 derived classes including `unassigned` (56% of cells).  It is
                 carried as an explicit category rather than dropped, so every
                 window has a well-defined profile.

  Reinhard       not applied.  Reinhard is a stain normalisation for H&E; there is
                 no stain here.  Per-sample intensity scaling is used instead.
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
LCM = os.path.join(WORK, "lcm")
RAW = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/work/lcm_tiles"

TILE = 224
LEVEL = 2
UM_PER_PX = 0.2125 * (2 ** LEVEL)      # 0.85 um/px


def tissue_mask(g):
    """Otsu on the uint16 morphology image; glass is the low mode."""
    v = g.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros(g.shape, bool)
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * omega - mu) ** 2 / (omega * (1 - omega))
    sb = np.nan_to_num(sb)
    thr = edges[int(np.argmax(sb))]
    m = v > thr
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.where(sizes > 0.0002 * m.size)[0] + 1
        m = np.isin(lab, keep)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=TILE,
                    help="tile side in pixels; the physical side is tile*um_per_px. "
                         "224 at level 2 is 190 um. Varying this is the scale "
                         "sensitivity test: if the patterns change with the window "
                         "size, they were a property of the window, not the tissue.")
    ap.add_argument("--samples", nargs="+", default=None)
    ap.add_argument("--tissue-min", type=float, default=0.08,
                    help="minimum nucleus-area coverage of the tile. Nuclei cover "
                         "only ~10-30%% of tissue area, so this is not the 0.6 "
                         "used for H&E tile staining")
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    TL = a.tile
    os.makedirs(a.outdir, exist_ok=True)

    samples = a.samples or sorted(
        f[:-8] for f in os.listdir(LCM) if f.endswith(".parquet"))
    print(f"{len(samples)} samples to cut", flush=True)

    meta_all = {}
    for s in samples:
        cp = os.path.join(LCM, f"{s}.parquet")
        morph = os.path.join(RAW, s, "morphology_mip.ome.tif")
        if not (os.path.exists(cp) and os.path.exists(morph)):
            print(f"  {s}: missing inputs, skipped")
            continue

        cells = pd.read_parquet(cp)
        with tifffile.TiffFile(morph) as tf:
            g = tf.series[0].levels[LEVEL].asarray()
        H, W = g.shape
        print(f"\n=== {s} ===  {len(cells)} cells, morphology L{LEVEL} "
              f"{W}x{H} at {UM_PER_PX:.3f} um/px", flush=True)

        px = cells["x"].to_numpy() / UM_PER_PX
        py = cells["y"].to_numpy() / UM_PER_PX
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        print(f"  {inside.mean()*100:.1f}% of cells inside the frame", flush=True)

        px_i = np.clip(px.astype(np.int64), 0, W - 1)
        py_i = np.clip(py.astype(np.int64), 0, H - 1)
        tile_id = (py_i // TL) * 100000 + (px_i // TL)

        tm = tissue_mask(g)
        anns = sorted(cells["annotation"].unique())
        aidx = {v: i for i, v in enumerate(anns)}
        an = cells["annotation"].map(aidx).to_numpy()
        pcs = [c for c in cells.columns if c.startswith("PC")]
        E = cells[pcs].to_numpy(dtype=np.float32)

        # Tissue coverage must come from the cells, not from pixel intensity.
        # The morphology MIP is a max-projection of nuclei: 62% of its pixels are
        # exactly 0 and the median is 11, so Otsu leaves a 6% mask broken into
        # 9,945 specks and every tile reads as 16% "tissue" even when it is full
        # of cells.  Summing each cell's measured nucleus_area per tile and
        # dividing by the tile's physical area is both meaningful and measured.
        #
        # nucleus_area is NOT in the assembled table -- that carries only
        # cell_id/annotation/x/y/PCs -- so it is joined back from the raw
        # cells.parquet.  The first attempt fell back to zeros silently, which
        # made every tile's coverage 0 and filtered the whole sample out.
        tile_um2 = (TL * UM_PER_PX) ** 2
        raw_cells = pd.read_parquet(os.path.join(RAW, s, "cells.parquet"))
        if "nucleus_area" not in raw_cells.columns:
            raise SystemExit(f"{s}: cells.parquet has no nucleus_area column")
        area = dict(zip(raw_cells["cell_id"].astype(str),
                        raw_cells["nucleus_area"].fillna(0).to_numpy()))
        nuc = np.array([area.get(str(c), np.nan) for c in cells["cell_id"]],
                       dtype=np.float64)
        miss = int(np.isnan(nuc).sum())
        if miss:
            print(f"  WARNING {miss} cells had no nucleus_area, set to 0")
        nuc = np.nan_to_num(nuc, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  nucleus_area: median {np.median(nuc[nuc > 0]) if (nuc > 0).any() else 0:.1f} um2, "
              f"tile area {tile_um2:.0f} um2", flush=True)

        uniq, inv = np.unique(tile_id[inside], return_inverse=True)
        keep, imgs, ncell = [], [], []
        keep_idx = []
        cov_all = []
        n_short = n_bounds = n_cover = 0
        cov_seen = []
        # `inv` holds INDICES into `uniq`, not tile ids.  Comparing it to `u`
        # (a tile id like 170028) never matches, which is why the first runs
        # reported every tile as having fewer than min_cells cells.  Group with
        # a stable argsort instead, so each tile gets the indices of its cells.
        order = np.argsort(inv, kind="stable")
        inv_sorted = inv[order]
        lo_i = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="left")
        hi_i = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="right")
        nuc_in = nuc[inside]
        for ui, u in enumerate(uniq):
            idx = order[lo_i[ui]:hi_i[ui]]
            n = int(idx.size)
            if n < a.min_cells:
                n_short += 1
                continue
            r, c = divmod(int(u), 100000)
            y0, x0 = r * TL, c * TL
            if y0 + TL > H or x0 + TL > W:
                n_bounds += 1
                continue
            cov = float(nuc_in[idx].sum()) / tile_um2
            cov_seen.append(cov)
            if cov < a.tissue_min:
                n_cover += 1
                continue
            keep.append(u)
            ncell.append(n)
            cov_all.append(cov)
            keep_idx.append(idx)

        if cov_seen:
            cs = np.array(cov_seen)
            print(f"  filter: {n_short} below {a.min_cells} cells, "
                  f"{n_bounds} out of bounds, {n_cover} below coverage "
                  f"{a.tissue_min}", flush=True)
            print(f"  coverage of the {len(cs)} tiles with enough cells: "
                  f"min {cs.min():.3f} median {np.median(cs):.3f} "
                  f"max {cs.max():.3f}", flush=True)
        else:
            print(f"  filter: {n_short} below {a.min_cells} cells, "
                  f"{n_bounds} out of bounds, 0 reached the coverage test",
                  flush=True)

        if not keep:
            print(f"  no tiles passed the filters", flush=True)
            continue

        Ximg = np.zeros((len(keep), TL, TL, 3), dtype=np.uint8)
        Xexpr = np.zeros((len(keep), len(pcs)), dtype=np.float32)
        Xann = np.zeros((len(keep), len(anns)), dtype=np.float32)
        meta = np.zeros((len(keep), 4), dtype=np.int64)

        # display scaling from the whole image, subsampled: a level-2 morphology
        # plane is ~130 M pixels and a full percentile on it costs seconds per
        # sample for no visible difference
        tv = g.reshape(-1)[::32]
        lo, hi = float(np.percentile(tv, 0.5)), float(np.percentile(tv, 99.9))
        hi = max(hi, lo + 1)

        remap = {int(u): i for i, u in enumerate(keep)}
        for i, u in enumerate(keep):
            idx = keep_idx[i]
            r, c = divmod(int(u), 100000)
            y0, x0 = r * TL, c * TL
            t = g[y0:y0 + TL, x0:x0 + TL].astype(np.float32)
            t = np.clip((t - lo) / (hi - lo), 0, 1) * 255
            Ximg[i] = np.repeat(t[..., None].astype(np.uint8), 3, axis=2)
            Xexpr[i] = E[inside][idx].mean(axis=0)
            cnt = np.bincount(an[inside][idx], minlength=len(anns)).astype(np.float32)
            Xann[i] = cnt / cnt.sum()
            meta[i] = (c, r, int(idx.size), int(u))

        out = os.path.join(a.outdir, f"{s}_tiles.npz")
        np.savez_compressed(out, img=Ximg, expr=Xexpr, ann=Xann, meta=meta,
                            annotations=np.array(anns),
                            um_per_px=np.array([UM_PER_PX]),
                            tile_px=np.array([TL]), level=np.array([LEVEL]))
        print(f"  {len(keep)} tiles kept of {len(uniq)} on the cell grid "
              f"({len(keep)/max(len(uniq),1)*100:.0f}%)", flush=True)
        print(f"  wrote {out} ({os.path.getsize(out)/1e6:.0f} MB) "
              f"img {Ximg.shape} expr {Xexpr.shape} ann {Xann.shape}", flush=True)

        meta_all[s] = {"n_tiles": len(keep), "n_candidates": int(len(uniq)),
                       "level": LEVEL, "um_per_px": UM_PER_PX,
                       "tile_px": TL, "tile_um": TL * UM_PER_PX,
                       "tissue_min": a.tissue_min, "min_cells": a.min_cells,
                       "cells_in_frame": float(inside.mean()),
                       "annotations": anns}

    json.dump(meta_all, open(os.path.join(a.outdir, "tiles_meta.json"), "w"),
              indent=2)
    tot = sum(v["n_tiles"] for v in meta_all.values())
    print(f"\n{len(meta_all)} samples, {tot:,} tiles total")


if __name__ == "__main__":
    main()
