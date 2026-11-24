"""A real serial-section volume, from the Pentimalli 2025 NSCLC block.

Six 5 um sections of one non-small-cell lung cancer block were imaged with CosMx
at 30 um spacing and registered into a shared frame with STIM (Zenodo
15240431).  Unlike the per-pattern blocks, which are synthesised from a single
2D section, this stack is a measurement in the z direction: the cells really do
come from successive slices of one tumour.

Two things are produced.

  density volume   the cell centroids rasterised into a 3D occupancy grid and
                   rendered by emission-absorption ray marching.  Nothing is
                   inferred: every voxel is a count of registered centroids.

  registration check   the 2D density projections of adjacent layers are
                   correlated.  If STIM had aligned the sections badly, the same
                   structures would not land on top of each other and the
                   correlation would collapse.  This is the evidence that the
                   volume is coherent rather than a stack of unrelated pictures.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

SER = "/mnt/storage/bzhang/AIVC/cellhippo/data/serial/stimwrap_files"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures/serial"

# The publication's own conversion: 5.6 px = 1 um.
PX_PER_UM = 5.6
SECTIONS = [4, 10, 16, 22, 28, 34]
SPACING_UM = 30.0


def load(sec, aligned=True):
    suf = "_aligned" if aligned else ""
    p = os.path.join(SER, f"section_{sec}{suf}.csv")
    a = np.loadtxt(p, delimiter=",")
    return a / PX_PER_UM            # -> microns


def raster(pts, origin, size_um, vox_um):
    """One 2D layer as an occupancy count grid."""
    n = int(round(size_um / vox_um))
    g, _, _ = np.histogram2d(
        (pts[:, 1] - origin[1]) / vox_um,
        (pts[:, 0] - origin[0]) / vox_um,
        bins=n, range=[[0, n], [0, n]])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox", type=float, default=8.0,
                    help="in-plane voxel size for the RENDER, microns")
    ap.add_argument("--corr-vox", type=float, default=80.0,
                    help="in-plane bin size for the registration check, microns. "
                         "At 8 um the grid is 78%% empty with a maximum count of 4 "
                         "per voxel, so Pearson is pure Poisson noise and measures "
                         "0.04 -- which is what the first run produced. At 80 um "
                         "the same data measures 0.49-0.69, the range the source "
                         "publication reports.")
    ap.add_argument("--smooth", type=float, default=1.1)
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--no-turntable", action="store_true",
                    help="skip the 48-frame turntable (48 x ~2 min); the four "
                         "static views are usually enough for the page")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print("=== loading the six aligned sections ===")
    raw = {s: load(s, aligned=False) for s in SECTIONS}
    ali = {s: load(s, aligned=True) for s in SECTIONS}
    for s in SECTIONS:
        print(f"  section {s:3d}: {len(raw[s]):6d} cells   "
              f"raw x {raw[s][:,0].min():7.0f}..{raw[s][:,0].max():7.0f}  "
              f"aligned x {ali[s][:,0].min():7.0f}..{ali[s][:,0].max():7.0f}")

    allpts = np.vstack([ali[s] for s in SECTIONS])
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    span = hi - lo
    print(f"\n  common footprint {span[0]/1000:.2f} x {span[1]/1000:.2f} mm, "
          f"stack {len(SECTIONS)} x {SPACING_UM:.0f} um = "
          f"{(len(SECTIONS)-1)*SPACING_UM:.0f} um")

    # side length is the square that holds the whole footprint
    side = float(max(span))
    centre = (lo + hi) / 2.0
    origin = centre - side / 2.0
    n = int(round(side / a.vox))
    nz = len(SECTIONS)
    print(f"  grid {n} x {n} x {nz} voxels at {a.vox:.1f} x {a.vox:.1f} x "
          f"{SPACING_UM:.0f} um = {side/1000:.2f} mm cube, "
          f"{(nz-1)*SPACING_UM:.0f} um deep")

    # ---- registration check on the 2D density projections
    print(f"\n=== registration check: adjacent-layer density correlation "
          f"({a.corr_vox:.0f} um bins) ===")
    nc = int(round(side / a.corr_vox))
    dens2d = {}
    for s in SECTIONS:
        dens2d[s] = raster(ali[s], origin, side, a.corr_vox)
    print(f"  {nc} x {nc} bins, mean count per bin "
          f"{np.mean([dens2d[s].mean() for s in SECTIONS]):.1f}, "
          f"empty bins {np.mean([(dens2d[s] == 0).mean() for s in SECTIONS])*100:.0f}%")
    corrs, corrs_raw = [], []
    for i in range(len(SECTIONS) - 1):
        u = dens2d[SECTIONS[i]].ravel()
        v = dens2d[SECTIONS[i + 1]].ravel()
        c = float(np.corrcoef(u, v)[0, 1])
        gu = raster(raw[SECTIONS[i]], origin, side, a.corr_vox).ravel()
        gv = raster(raw[SECTIONS[i + 1]], origin, side, a.corr_vox).ravel()
        cr = float(np.corrcoef(gu, gv)[0, 1])
        corrs.append(c)
        corrs_raw.append(cr)
        print(f"  {SECTIONS[i]:3d} -> {SECTIONS[i+1]:3d}   "
              f"aligned r = {c:.2f}    unaligned r = {cr:.2f}")

    # the render grid is finer than the correlation bins, so build it separately
    vol = np.stack([raster(ali[s], origin, side, a.vox) for s in SECTIONS],
                   axis=2).astype(np.float32)
    print(f"\n  render volume {vol.shape}, {vol.sum():.0f} cell centroids placed, "
          f"max counts per voxel {vol.max():.0f}")

    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(vol, sigma=(a.smooth, a.smooth, 0.34))
    nog = (vol == 0).mean()
    print(f"  empty voxel fraction {nog*100:.1f}% before smoothing")

    # ---- render
    XR = {
        "label": "cell density",
        "rgb": (0.30, 0.72, 0.95),
        "a": 0.030,
        "em": 0.85,
    }
    H, W, D = sm.shape
    rgb = np.zeros((H, W, D, 3), np.float32)
    alpha = np.zeros((H, W, D), np.float32)
    emis = np.zeros((H, W, D), np.float32)

    peak = float(np.percentile(sm[sm > 0], 99)) if (sm > 0).any() else 1.0
    t = np.clip(sm / max(peak, 1e-6), 0, 1) ** 0.6
    rgb[...] = np.array(XR["rgb"], np.float32)
    alpha = (t * XR["a"]).astype(np.float32)
    emis = (t * XR["em"]).astype(np.float32)

    def render(size, az, el, step=0.5, gain=3.0, bg=(0, 0, 0)):
        a_ = np.deg2rad(az)
        e_ = np.deg2rad(el)
        fwd = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_),
                        np.sin(e_)], np.float32)
        up0 = np.array([0, 0, 1], np.float32)
        right = np.cross(fwd, up0)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)
        ctr = np.array([H / 2, W / 2, D / 2], np.float32)
        pts = np.array([[i, j, k] for i in (0, H - 1) for j in (0, W - 1)
                        for k in (0, D - 1)], np.float32)
        rel = pts - ctr
        pr, pu, pf = rel @ right, rel @ up, rel @ fwd
        half = max(np.abs(pr).max(), np.abs(pu).max()) * 0.94
        scale = size / (2 * half)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        xr = (xx - size / 2) / scale
        yu = -(yy - size / 2) / scale
        origin = ctr - fwd * (pf.max() + 2)
        nsteps = int((pf.max() - pf.min() + 6) / step)
        C = np.zeros((size, size, 3), np.float32)
        T = np.ones((size, size), np.float32)
        L = np.array([-0.42, -0.66, 0.62], np.float32)
        L /= np.linalg.norm(L)
        L2 = np.array([0.6, 0.25, -0.4], np.float32)
        L2 /= np.linalg.norm(L2)
        from scipy.ndimage import sobel
        gm = np.linalg.norm(np.stack([sobel(alpha, axis=i) for i in range(3)],
                                     axis=-1), axis=-1, keepdims=True) + 1e-6
        g = np.stack([sobel(alpha, axis=i) for i in range(3)], axis=-1) / gm
        for si in range(nsteps):
            tt = si * step
            px = origin[0] + xr * right[0] + yu * up[0] + tt * fwd[0]
            py = origin[1] + xr * right[1] + yu * up[1] + tt * fwd[1]
            pz = origin[2] + xr * right[2] + yu * up[2] + tt * fwd[2]
            ok = ((px >= 0) & (px < H - 1) & (py >= 0) & (py < W - 1) &
                  (pz >= 0) & (pz < D - 1))
            if not ok.any():
                continue
            xi = np.clip(px.astype(np.int32), 0, H - 2)
            yi = np.clip(py.astype(np.int32), 0, W - 2)
            zi = np.clip(pz.astype(np.int32), 0, D - 2)
            al = alpha[xi, yi, zi] * ok
            if al.max() <= 1e-6:
                continue
            col = rgb[xi, yi, zi]
            em = emis[xi, yi, zi]
            nn = g[xi, yi, zi]
            shade = (0.22 + 0.9 * np.clip((nn * L).sum(-1), 0, 1)
                     + 0.2 * np.clip((nn * L2).sum(-1), 0, 1))
            src = col * shade[..., None] + col * em[..., None]
            a_s = np.clip(1 - np.exp(-al * step * gain), 0, 1)[..., None]
            C += T[..., None] * a_s * src
            T *= (1 - a_s[..., 0])
            if T.max() < 0.004:
                break
        img = 1 - np.exp(-np.clip(C, 0, None) * 2.0)
        if bg != (0, 0, 0):
            img = img + np.asarray(bg, np.float32) * (1 - img)
        return np.clip(img, 0, 1)

    views = [("hero", 38.0, 16.0), ("front", 0.0, 6.0),
             ("side", 90.0, 6.0), ("top", 0.0, 88.0)]
    for name, az, el in views:
        im = render(a.size, az, el)
        p = os.path.join(a.outdir, f"serial_{name}.png")
        Image.fromarray((im * 255).astype(np.uint8)).save(p)
        arr = np.asarray(Image.open(p).convert("RGB"))
        print(f"  {name:6s} az {az:5.1f} el {el:4.1f}  mean {arr.mean():6.1f}  "
              f"max {arr.max():3d}  lit {100*(arr.max(axis=2)>10).mean():5.1f}%")

    # turntable
    if not a.no_turntable:
        frames = []
        for i in range(48):
            im = render(760, i * 360.0 / 48, 15.0)
            frames.append(Image.fromarray((im * 255).astype(np.uint8)))
        frames[0].save(os.path.join(a.outdir, "serial_turntable.gif"),
                       save_all=True, append_images=frames[1:], duration=70,
                       loop=0)
        print("  turntable 48 frames")
    else:
        print("  turntable skipped")

    json.dump({
        "source": "Pentimalli et al., Cell Systems 2025, Zenodo 15240431",
        "note": "six 5 um sections of one NSCLC block, CosMx, 30 um spacing, "
                "STIM-registered; this is a real z stack, not a model",
        "px_per_um": PX_PER_UM,
        "sections": SECTIONS,
        "spacing_um": SPACING_UM,
        "n_cells": {str(s): int(len(ali[s])) for s in SECTIONS},
        "total_cells": int(sum(len(ali[s]) for s in SECTIONS)),
        "footprint_mm": [float(span[0] / 1000), float(span[1] / 1000)],
        "depth_um": float((len(SECTIONS) - 1) * SPACING_UM),
        "voxel_um": [a.vox, a.vox, SPACING_UM],
        "grid": [int(n), int(n), int(nz)],
        "adjacent_correlation_aligned": [float(c) for c in corrs],
        "adjacent_correlation_unaligned": [float(c) for c in corrs_raw],
        "empty_voxel_fraction": float(nog),
    }, open(os.path.join(a.outdir, "serial_volume.json"), "w"), indent=2)
    print(f"\nwrote {a.outdir}/serial_volume.json")


if __name__ == "__main__":
    main()
