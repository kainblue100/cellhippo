"""Infer a 3D cell arrangement from the 2D section, then colour it by phenotype.

Two ideas are combined, and both are load-bearing:

  CellSqueeze3D (Kong & Lu, Adv Sci 2026;13(10), doi 10.1002/advs.202518706)
    A single H&E section still contains the third dimension.  Sectioning a sphere
    of radius R at distance z from its centre leaves a disc of radius
    a = sqrt(R^2 - z^2).  The apparent 2D area is therefore not the cell's size;
    it is the size of the slice.  Recovering (R, z) is an inverse problem, which
    that paper solves with particle swarm optimisation under biomechanical
    constraints.

  Alon et al. 2021 (arXiv:2101.11600)
    Build the 3D result as explicit primitives -- spheres with a position, a
    radius and a colour -- rather than a voxel volume.  That is what makes the
    learned pattern visible in the model instead of buried in an occupancy grid.
    Their generator is constrained so shapes stay biologically feasible.

Identifiability, stated plainly, because it drives the whole design:
  A single number per cell (its 2D area) cannot determine two unknowns (R and z).
  What breaks the degeneracy is that **cells of one annotated type share a true
  radius distribution**, so one cell's small apparent area is read as a cut away
  from its equator rather than as a small cell.  The largest apparent radius in a
  group is the least-cut cell, so it bounds that group's true radius from below.

So the model is:
  R_group   = high quantile of the group's apparent radii   (near-equatorial cut)
  z_i       = +-sqrt(R_group^2 - a_i^2),  sign chosen to reduce overlap
  given     x, y from the registration;  apparent area from Xenium cell_area
  assumed   cells are spheres; every cell of a type shares that type's radius;
            the section is a slab of finite thickness

Nothing here measures z.  It is the least-assumption arrangement consistent with
the 2D observation, and the residual overlap is reported rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

MAP = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
       "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
       "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE")}


def assign_z(a2d, xy, r_group, slab_um, seed=0, sweeps=80, grow=0.05):
    """Pick a sign for each cell's z, and re-fit R, to reduce overlap.

    Magnitude is fixed by the sectioning relation |z| = sqrt(R^2 - a^2); only the
    sign is free.  The first version used np.sign(z) to carry the sign, which is 0
    at z = 0 -- so every cell stayed exactly on the section plane and the overlap
    was never touched.  The sign is a separate variable now.
    """
    rng = np.random.default_rng(seed)
    n = len(a2d)
    R = np.maximum(r_group, a2d).astype(float)
    sign = rng.choice([-1.0, 1.0], size=n)
    d2d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)

    def pen_for(sg, R):
        z_mag = np.sqrt(np.clip(R ** 2 - a2d ** 2, 0, None))
        z = sg * np.minimum(z_mag, slab_um)
        dist = np.sqrt(d2d ** 2 + (z[:, None] - z[None, :]) ** 2)
        p = np.clip((R[:, None] + R[None, :]) - dist, 0, None)
        np.fill_diagonal(p, 0.0)
        return z, p

    # Both sign choices are evaluated for every cell at once; the per-cell sweep
    # version was O(n^3) and did not finish on 3,465 cells.
    prev = None
    for s in range(sweeps):
        z, pen = pen_for(sign, R)
        total = float(pen.sum())
        if prev is not None and total >= prev - 1e-9:
            break
        prev = total
        _, pen_alt = pen_for(-sign, R)
        flip = pen_alt.sum(axis=1) < pen.sum(axis=1) - 1e-9
        if not flip.any():
            # flipping cannot help: the residual is a radius problem, so let the
            # radii relax a little, as CellSqueeze3D's constraints do
            R = R + grow
            continue
        sign = np.where(flip, -sign, sign)

    z, pen = pen_for(sign, R)
    return z, R, pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="5k")
    ap.add_argument("--name", default="fused")
    ap.add_argument("--block", type=int, default=1,
                    help="reconstruct a (2*block+1)^2 tile block")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--slab-um", type=float, default=6.0,
                    help="half-thickness of the section, microns")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="quantile of a group's apparent radii taken as its true "
                         "radius; 0.9 says the least-cut decile is near-equatorial")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = json.load(open(os.path.join(WORK, f"{a.name}.json")))
    pheno = np.array(res["tile_pheno"])
    meta = np.array(res["tile_meta"])
    slides = np.array(res["slides"])
    G = int(pheno.max()) + 1

    cells = pd.read_parquet(os.path.join(WORK, f"{a.tag}_full.parquet"))
    ver, nm = MAP[a.tag]
    raw = pd.read_parquet(os.path.join(TENX, f"{a.tag}__{ver}__{nm}",
                                       f"{nm}_cells.parquet"))
    cells = cells.join(raw.set_index("cell_id")[["cell_area", "nucleus_area"]],
                       on="cell_id")
    lvl = json.load(open(os.path.join(WORK, f"{a.tag}_tiles_meta.json")))
    reg = json.load(open(os.path.join(OUT, "registration_final.json")))[a.tag]

    off = 0
    for t in res["tags"]:
        n_t = int((slides == t).sum())
        if t == a.tag:
            break
        off += n_t
    ph_tag = pheno[off:off + int((slides == a.tag).sum())]
    mt_tag = meta[slides == a.tag]
    tile_px = lvl["tile_px"]
    s_lvl = reg["scale_L0"] / (2 ** lvl["level"])
    extent_um = tile_px / s_lvl
    print(f"=== {a.tag} ===  tile {tile_px}px = {extent_um:.0f} um, "
          f"slab +-{a.slab_um} um")

    # pick the centre tile of the most phenotype-diverse neighbourhood
    pos = {(int(c), int(r)): i for i, (c, r, n, k) in enumerate(mt_tag)}
    if a.tile is None:
        best, best_t = -1, 0
        for i, (c, r, n, k) in enumerate(mt_tag):
            seen = {int(ph_tag[j]) for dc in range(-a.block, a.block + 1)
                    for dr in range(-a.block, a.block + 1)
                    if (j := pos.get((int(c) + dc, int(r) + dr))) is not None}
            if len(seen) > best:
                best, best_t = len(seen), i
        sel = best_t
        print(f"  centre tile index {sel} with {best} phenotypes in the block")
    else:
        sel = a.tile

    tc, tr = int(mt_tag[sel, 0]), int(mt_tag[sel, 1])
    cen_px = np.array([(tc + 0.5) * tile_px, (tr + 0.5) * tile_px])
    rad_px = (a.block + 0.5) * tile_px

    R = np.array(reg["R"])
    tvec = np.array(reg["t"])
    src = np.column_stack([cells["x"].to_numpy(), cells["y"].to_numpy()]).astype(float)
    pts = (s_lvl * (R @ src.T)).T + tvec
    keep = ((np.abs(pts[:, 0] - cen_px[0]) < rad_px) &
            (np.abs(pts[:, 1] - cen_px[1]) < rad_px))
    print(f"  field = {2*a.block+1}x{2*a.block+1} tiles, "
          f"{extent_um*(2*a.block+1):.0f} um across, {keep.sum()} cells")

    # phenotype per cell from its tile
    col = (pts[keep, 0] // tile_px).astype(int)
    row = (pts[keep, 1] // tile_px).astype(int)
    key2ph = {int(k): int(p) for k, p in zip(mt_tag[:, 3], ph_tag)}
    cell_ph = np.array([key2ph.get(int(rw) * 100000 + int(cl), -1)
                        for rw, cl in zip(row, col)])
    ok = cell_ph >= 0
    print(f"  {ok.sum()} cells carry a phenotype; present: "
          f"{sorted(set(cell_ph[ok].tolist()))}")

    x_um = (pts[keep, 0] - cen_px[0]) / s_lvl
    y_um = (pts[keep, 1] - cen_px[1]) / s_lvl
    area = cells["cell_area"].to_numpy(dtype=float)[keep]
    narea = cells["nucleus_area"].to_numpy(dtype=float)[keep]
    bad = ~np.isfinite(area)
    if bad.any():
        area = np.where(bad, np.nanmedian(area), area)
    a2d = np.sqrt(np.clip(area, 0.5, None) / np.pi)

    xy = np.column_stack([x_um[ok], y_um[ok]])
    a2 = a2d[ok]
    ph = cell_ph[ok]

    # One true radius per phenotype group, but capped by what the observed density
    # can actually accommodate.  N spheres of radius R in area A and slab 2t
    # occupy N*(4/3)pi R^3, which cannot exceed the packing fraction phi of the
    # slab; solving for R gives the largest self-consistent radius.  Without this
    # cap the group radius (a high quantile of apparent radii) lands above the
    # limit and the solver is asked to do something geometrically impossible.
    phi = 0.60
    A = (2 * (a.block + 0.5) * (tile_px / s_lvl)) ** 2
    r_pack = (3 * phi * A * (2 * a.slab_um) / (4 * np.pi * max(1, len(a2)))) ** (1 / 3)
    print(f"  packing-limited radius {r_pack:.2f} um "
          f"(A={A/1e6:.2f} mm2, slab {2*a.slab_um:.0f} um, n={len(a2)})")

    r_group = np.zeros(len(a2))
    for g in sorted(set(ph.tolist())):
        m = ph == g
        q = float(np.quantile(a2[m], a.quantile))
        r_group[m] = min(q, r_pack)
    print("  group radii: " + ", ".join(
        f"P{g}={r_group[ph==g][0]:.2f}" for g in sorted(set(ph.tolist()))))

    z, R3, pen = assign_z(a2, xy, r_group, a.slab_um)
    n_over = int((pen > 0).sum())
    print(f"  z {z.min():+.2f}..{z.max():+.2f} um (sd {z.std():.2f})")
    print(f"  apparent radius sd {a2.std():.3f} -> true radius sd {R3.std():.3f} um")
    print(f"  residual 3D overlap: {n_over} pairs, max penetration "
          f"{pen.max():.2f} um, mean {pen[pen>0].mean() if n_over else 0:.2f} um")
    d2d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    pen2d = np.clip((R3[:, None] + R3[None, :]) - d2d, 0, None)
    np.fill_diagonal(pen2d, 0.0)
    n2d = int((pen2d > 0).sum())
    frac = n2d / max(1, len(R3) * (len(R3) - 1))
    print(f"  same radii, z ignored: {n2d} overlapping pairs "
          f"(so {n_over/max(1,n2d)*100:.0f}% of the 2D conflicts survive the "
          f"3D adjustment)")
    # packing fraction actually realised, against the phi the radius cap assumed
    vol = (4 / 3) * np.pi * (R3 ** 3).sum()
    print(f"  realised packing fraction {vol/(A*2*a.slab_um):.3f} (cap used {phi})")

    payload = {
        "tag": a.tag, "tile": [tc, tr], "block": a.block,
        "n_cells": int(ok.sum()), "n_phenotypes": G, "slab_um": a.slab_um,
        "extent_um": float(extent_um),
        "x": x_um[ok].round(3).tolist(), "y": y_um[ok].round(3).tolist(),
        "z": z.round(3).tolist(),
        "r_apparent": a2.round(3).tolist(), "r_true": R3.round(3).tolist(),
        "phenotype": ph.tolist(),
        "apparent_radius_sd": float(a2.std()),
        "true_radius_sd": float(R3.std()),
        "overlap_pairs": n_over, "max_penetration_um": float(pen.max()),
        "overlap_2d_pairs": n2d,
        "overlap_resolved_frac": float(1 - n_over / max(1, n2d)),
        "packing_limit_um": float(r_pack),
        "packing_realised": float(vol / (A * 2 * a.slab_um)),
        "mean_penetration_um": float(pen[pen > 0].mean()) if n_over else 0.0,
    }
    p = a.out or os.path.join(OUT, f"model3d_{a.tag}.json")
    json.dump(payload, open(p, "w"))
    print(f"  wrote {p} ({os.path.getsize(p)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
