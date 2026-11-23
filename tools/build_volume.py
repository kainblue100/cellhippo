"""Build the per-pattern 3D cell model, following Annotile's construction.

This replaces the CellSqueeze3D-style inversion used before, which was the wrong
tool.  Annotile's approach is simpler and better behaved:

  radius   a cell-type prior (epi 8.5 um, endo 5.5, nkt 4.5, ...), not an
           inversion of the segmented 2D area.  Inverting the area needed an
           identifiability argument and left residual overlap that had to be
           reported as a caveat; a type prior needs neither.
  depth    centres start uniform in a 10 um section, then are pushed apart along
           z until the spheres stop interpenetrating.  No optimisation, no PBC.
  patch    the densest 40 um square, widened to 80 and then 120 if it holds
           fewer than 20 cells, capped at 160 cells.
  layers   each cell is a membrane, a cytoplasm and a nucleus.

The result feeds tools/render_cells.py, which is Annotile's own Blender script.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/work/volume"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"

THICKNESS_UM = 10.0
PATCH_UM = 40.0
MAX_CELLS = 160
Z_SCALE = 4.0

# cell-type radius prior, from Annotile's volume.py.  Kept identical so the two
# projects read on the same scale.
RADIUS_UM = {
    "epi": 8.5, "epi?": 8.0, "normal_epi": 7.5, "AT1": 6.5, "AT2_like": 7.0,
    "cil": 7.5, "endo": 5.5, "endo?": 5.5, "lymph_endo": 5.0, "fib": 7.0,
    "fib_activated": 7.5, "fib_SYNPO2": 6.5, "nkt": 4.5,
    "Tfh_CXCL13_region": 5.0, "b": 4.2, "plasma": 5.0, "mph": 7.0, "mast": 5.5,
}

# Annotile's cell-type palette, verbatim.  An earlier version used invented
# muted pinks and the render came out a dark plum: grape() mixes 62% toward one
# magenta, so a desaturated source lands below the target's value instead of
# around it.  Starting from these saturated hues is what gives the reference its
# rose/lavender range.
TYPE_COLOR = {
    "epi": "#c2410c", "epi?": "#fb923c", "normal_epi": "#fdba74",
    "AT1": "#fde68a", "AT2_like": "#eab308", "cil": "#0f766e",
    "endo": "#1d4ed8", "endo?": "#60a5fa", "lymph_endo": "#312e81",
    "fib": "#65a30d", "fib_activated": "#166534", "fib_SYNPO2": "#a3e635",
    "nkt": "#7c3aed", "Tfh_CXCL13_region": "#c4b5fd", "b": "#db2777",
    "plasma": "#f9a8d4", "mph": "#b45309", "mast": "#78716c",
}


def radius_of(label: str) -> float:
    return RADIUS_UM.get(label, 6.0)


# The annotation notes mark these groups as mixed, unused or too uncertain.  Not
# filtering them was a real defect: the densest 40 um patch then landed on a
# low-quality cell run or an immune aggregate, which is not what the pattern is
# named for.  epi? and endo? are kept because the source notes say those two were
# checked against the image and retained.
DROP_EXACT = frozenset({"low_feat_cell", "mph_CXCL9_region", "fib?",
                        "nkt?CCL19", "lymph_endo?CCL19"})
KEEP_MARKED = frozenset({"epi?", "endo?"})


def keep_label(label: str) -> bool:
    if label in DROP_EXACT:
        return False
    if "?" in label and label not in KEEP_MARKED:
        return False
    return True


def keep_label_mask(cells: pd.DataFrame):
    return cells["annotation"].astype(str).map(keep_label).to_numpy()


def densest(cells: pd.DataFrame, size: float, step: float = 8.0) -> pd.DataFrame:
    """Most populated `size`-micron square, found with an integral image.

    The brute-force version (a boolean mask per candidate origin) is
    O(windows x cells); on 253k cells that ran past ten minutes.  Binning once
    and summing with a cumulative array makes it O(cells + windows).
    """
    xs = cells["x"].to_numpy()
    ys = cells["y"].to_numpy()
    x_min, y_min = float(xs.min()), float(ys.min())
    nx = int(np.ceil((xs.max() - x_min) / step)) + 2
    ny = int(np.ceil((ys.max() - y_min) / step)) + 2
    hist = np.zeros((ny, nx), dtype=np.int64)
    ix = np.clip(((xs - x_min) / step).astype(int), 0, nx - 1)
    iy = np.clip(((ys - y_min) / step).astype(int), 0, ny - 1)
    np.add.at(hist, (iy, ix), 1)
    integ = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    integ[1:, 1:] = hist.cumsum(0).cumsum(1)

    win = max(1, int(round(size / step)))
    # vectorised sliding-window sum over the integral image
    sums = (integ[win:, win:] - integ[:-win, win:]
            - integ[win:, :-win] + integ[:-win, :-win])
    if sums.size == 0:
        return cells
    oy, ox = np.unravel_index(int(np.argmax(sums)), sums.shape)
    x0 = x_min + ox * step
    y0 = y_min + oy * step
    m = ((cells["x"] >= x0) & (cells["x"] < x0 + win * step) &
         (cells["y"] >= y0) & (cells["y"] < y0 + win * step))
    return cells[m]


def patch_of(cells: pd.DataFrame):
    for size in (PATCH_UM, 80.0, 120.0):
        chosen = densest(cells, size)
        if len(chosen) >= 20 or size == 120.0:
            if len(chosen) > MAX_CELLS:
                cx, cy = chosen["x"].median(), chosen["y"].median()
                d = (chosen["x"] - cx) ** 2 + (chosen["y"] - cy) ** 2
                chosen = chosen.loc[d.nsmallest(MAX_CELLS).index]
            return chosen.copy(), size
    return cells.copy(), PATCH_UM


def place(xy: np.ndarray, radius: np.ndarray, thickness=THICKNESS_UM,
          seed=42) -> np.ndarray:
    """Push centres apart along depth until the spheres stop intersecting."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, thickness, size=len(radius))
    for _ in range(60):
        delta = z[:, None] - z[None, :]
        flat = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt(np.sum(flat * flat, axis=2) + delta * delta) + 1e-6
        overlap = np.maximum(radius[:, None] + radius[None, :] - dist, 0.0)
        np.fill_diagonal(overlap, 0.0)
        push = (overlap * np.sign(delta)).sum(axis=1)
        hits = np.maximum((overlap > 0).sum(axis=1), 1)
        z = np.clip(z + 0.45 * push / hits, 0.0, thickness)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["5k", "lc"])
    ap.add_argument("--name", default="fused")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    res = json.load(open(os.path.join(WORK, f"{a.name}.json")))
    pheno = np.array(res["tile_pheno"])
    meta = np.array(res["tile_meta"])
    slides = np.array(res["slides"])
    G = int(pheno.max()) + 1
    tile_px = None

    reg = json.load(open(os.path.join(FIGS, "registration_final.json")))
    models = []
    for t in a.tags:
        lvl = json.load(open(os.path.join(WORK, f"{t}_tiles_meta.json")))
        tile_px = lvl["tile_px"]
        s_lvl = reg[t]["scale_L0"] / (2 ** lvl["level"])
        R = np.array(reg[t]["R"]); tv = np.array(reg[t]["t"])

        off = 0
        for s in res["tags"]:
            n_s = int((slides == s).sum())
            if s == t:
                break
            off += n_s
        n_tag = int((slides == t).sum())
        ph_tag = pheno[off:off + n_tag]
        mt_tag = meta[slides == t]

        cells = pd.read_parquet(os.path.join(WORK, f"{t}_full.parquet"))
        # cells["x"], cells["y"] are micron slide coordinates, which is what the
        # patch geometry needs.  The registered pixel positions are used only to
        # work out which tile (hence which phenotype) each cell belongs to.
        src = np.column_stack([cells["x"].to_numpy(),
                               cells["y"].to_numpy()]).astype(float)
        pts = (s_lvl * (R @ src.T)).T + tv
        key2ph = {int(k): int(p) for k, p in zip(mt_tag[:, 3], ph_tag)}
        col = (pts[:, 0] // tile_px).astype(int)
        row = (pts[:, 1] // tile_px).astype(int)
        cells = cells.assign(
            pattern=[key2ph.get(int(rw) * 100000 + int(cl), -1)
                     for rw, cl in zip(row, col)])
        cells = cells[keep_label_mask(cells)]
        cells = cells[cells["pattern"] >= 0]
        print(f"=== {t} ===  {len(cells):,} cells after label filtering, "
              f"{cells['pattern'].nunique()} patterns")

        for g in range(G):
            sub = cells[cells["pattern"] == g]
            if len(sub) < 30:
                print(f"  P{g}: only {len(sub)} cells, skipped")
                continue
            chosen, span = patch_of(sub)
            x0, y0 = chosen["x"].min(), chosen["y"].min()
            xy = np.column_stack([chosen["x"] - x0, chosen["y"] - y0])
            labels = chosen["annotation"].astype(str).tolist()
            radius = np.array([radius_of(v) for v in labels], dtype=float)
            depth = place(xy, radius)
            models.append({
                "pattern": f"P{g}", "slice": t, "patch_um": span,
                "cells_in_patch": int(len(chosen)),
                "x0_um": float(x0), "y0_um": float(y0),
                "cells": [
                    {"x": round(float(xy[i, 0]), 2),
                     "y": round(float(xy[i, 1]), 2),
                     "z": round(float(depth[i]), 2),
                     "r": round(float(radius[i]), 2),
                     "type": labels[i],
                     "color": TYPE_COLOR.get(labels[i], "#a8a29e")}
                    for i in range(len(labels))],
                "backdrop": f"backdrops/P{g}_{t}.png",
            })
            print(f"  P{g}: {len(chosen)} cells in {span:.0f} um, "
                  f"types {sorted(set(labels))[:6]}")

    payload = {"thickness_um": THICKNESS_UM, "z_scale": Z_SCALE,
               "patterns": models}
    p = os.path.join(a.outdir, "scene.json")
    json.dump(payload, open(p, "w"), indent=1)
    print(f"\nwrote {p}  ({len(models)} patterns)")


if __name__ == "__main__":
    main()
