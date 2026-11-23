"""Emission-absorption volume render of the inferred 3D cell model.

Why volume rendering and not the Blender surfaces: the target is the look of a
translucent body with structure glowing through it, and that is photometry, not
geometry.  Surface meshes and metaballs cap out around 3-5/10 no matter how the
material is tuned -- ten Blender iterations here confirmed it.  This file follows
the scientific-volume-rendering recipe: front-to-back ray march, per-sample
opacity, gradient-based shading, filmic shoulder.

The arithmetic first, because it constrains everything else.  The 3D model is
inferred from a *single* section, so z spans only +-6 um while x-y spans ~134 um.
That is a 12 : 134 wafer.  Rendering the whole thing would produce a pancake at
any camera angle, so the field is cropped in x-y to roughly twice the slab
thickness, giving a 40 x 40 x 14 um block -- a credible small piece of tissue.

Per-sample opacity matters more than anything else here.  A ray through a 24
voxel slab at step 0.6 has ~40 samples, so a per-sample alpha of 0.02 integrates
to 1-(1-0.02)^40 = 0.55 before any overlap and buries the interior.  The values
below are chosen so the faint tissue integrates to ~0.10 and the densest class to
~0.6.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

DOCS = r"D:\coda_matlab\cellhippo\docs"
OUT = r"D:\coda_matlab\cellhippo\figures\volume"

# phenotype -> (rgb, per-sample opacity, emission)
#   Per-sample opacity integrates to 1-(1-a)^N over N = D/step samples, which is
#   ~62 here.  The first values (a ~0.004) integrated to 0.23 and the render
#   measured mean 12.8 -- nearly empty.  These give ~0.5 total on a cell body,
#   which is the "solid but still translucent" target.
#
#   Classes 6..10 are the nuclei of phenotypes 0..4: denser, darker and less
#   emissive than their membrane, so a core is visible inside each body.
XFER = {
    1: ((0.42, 0.32, 0.80), 0.0090, 0.90),   # P0 membrane
    2: ((0.68, 0.38, 0.80), 0.0060, 0.42),   # P1 membrane
    3: ((0.48, 0.46, 0.88), 0.0060, 0.38),   # P2 membrane
    4: ((0.80, 0.30, 0.66), 0.0085, 0.78),   # P3 membrane
    5: ((0.32, 0.52, 0.94), 0.0075, 0.68),   # P4 membrane
    6: ((0.22, 0.12, 0.46), 0.0200, 0.30),   # P0 nucleus
    7: ((0.38, 0.16, 0.52), 0.0160, 0.18),   # P1 nucleus
    8: ((0.24, 0.22, 0.56), 0.0160, 0.16),   # P2 nucleus
    9: ((0.48, 0.12, 0.38), 0.0190, 0.26),   # P3 nucleus
    10: ((0.14, 0.26, 0.62), 0.0180, 0.24),  # P4 nucleus
}


def build_grid(model, field_um, vox_um, nucleus_frac=0.62):
    """Rasterise the cell spheres into a label grid, with a nucleus inside each.

    Cells are drawn as solid spheres of their inferred radius, so the grid is a
    faithful occupancy of the model rather than a scatter of points.  An inner
    core of `nucleus_frac` of the radius is labelled separately: uniform spheres
    have no internal structure by construction, and without a core the render
    reads as a cluster of flat discs no matter how the transfer function is set.
    The core gets its own opacity so the interior is visible through the
    membrane, which is the look of the reference's zoomed panels.
    """
    x = np.asarray(model["x"]); y = np.asarray(model["y"])
    z = np.asarray(model["z"]); r = np.asarray(model["r_true"])
    p = np.asarray(model["phenotype"])

    step = max(1.0, field_um / 6)
    best, bc = -1, (float(x.mean()), float(y.mean()))
    for cx in np.arange(x.min() + field_um / 2, x.max() - field_um / 2 + 1, step):
        for cy in np.arange(y.min() + field_um / 2, y.max() - field_um / 2 + 1,
                            step):
            m = ((x - cx) ** 2 + (y - cy) ** 2) < (field_um / 2) ** 2
            n = int(m.sum())
            if n < 8:
                continue
            n_ph = int(len(np.unique(p[m])))
            # density has to stay competitive: weighting diversity too hard
            # picked a 12-cell corner where three phenotypes happened to meet,
            # and the render was nearly empty
            score = n_ph * 1200 + n
            if score > best:
                best, bc = score, (float(cx), float(cy))
    keep = ((x - bc[0]) ** 2 + (y - bc[1]) ** 2) < (field_um / 2) ** 2
    print(f"  field {field_um:.0f} um at ({bc[0]:.0f},{bc[1]:.0f}): "
          f"{int(keep.sum())} cells, phenotypes "
          f"{sorted(set(p[keep].tolist()))}")

    half = field_um / 2.0
    zpad = half * 0.42
    nx = int(round(field_um / vox_um))
    nz = max(int(round((2 * zpad) / vox_um)), 6)
    lab = np.zeros((nx, nx, nz), dtype=np.int16)
    print(f"  grid {nx} x {nx} x {nz} voxels at {vox_um} um "
          f"= {field_um:.0f} x {field_um:.0f} x {nz*vox_um:.0f} um")

    for i in np.where(keep)[0]:
        ci = (x[i] - bc[0] + half) / vox_um
        cj = (y[i] - bc[1] + half) / vox_um
        ck = (z[i] + zpad) / vox_um
        rr = r[i] / vox_um
        lo = np.maximum([ci - rr, cj - rr, ck - rr], 0).astype(int)
        hi = np.minimum([ci + rr, cj + rr, ck + rr],
                        [nx - 1, nx - 1, nz - 1]).astype(int) + 1
        if (hi <= lo).any():
            continue
        gi, gj, gk = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = (gi - ci) ** 2 + (gj - cj) ** 2 + (gk - ck) ** 2
        sub = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[d2 <= rr * rr] = int(p[i]) + 1              # membrane class
        sub[d2 <= (rr * nucleus_frac) ** 2] = int(p[i]) + 6   # nucleus class
    return lab


def build_rgba(lab, soft=(0.40, 0.40, 0.24)):
    H, W, D = lab.shape
    rgb = np.zeros((H, W, D, 3), np.float32)
    a = np.zeros((H, W, D), np.float32)
    e = np.zeros((H, W, D), np.float32)
    for cls, (col, op, em) in XFER.items():
        m = (lab == cls)
        if not m.any():
            continue
        rgb[m] = np.array(col, np.float32)
        a[m] = op
        e[m] = em
    # soften just enough to remove voxel stair-stepping.  The first pass used
    # sigma 1.4 and dissolved every cell boundary into an amorphous cloud -- the
    # render measured 2/10 with no resolvable structure.  Colour is blurred less
    # than alpha so hue stays saturated where the body is thin.
    a = gaussian_filter(a, sigma=soft).astype(np.float32)
    for i in range(3):
        rgb[..., i] = gaussian_filter(rgb[..., i],
                                      sigma=tuple(s * 0.5 for s in soft))
    e = gaussian_filter(e, sigma=tuple(s * 0.5 for s in soft)).astype(np.float32)
    return rgb, a, e


def render(lab, out, size=1200, az=38.0, el=18.0, zoom=1.0, step=0.6,
           gain=2.2, bg=(0.0, 0.0, 0.0)):
    rgb, alpha, emis = build_rgba(lab)
    H, W, D = alpha.shape

    dens = alpha.copy()
    g = np.stack([sobel(dens, axis=i) for i in range(3)], axis=-1)
    gmag = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-6
    g = (g / gmag).astype(np.float32)

    a_ = np.deg2rad(az); e_ = np.deg2rad(el)
    fwd = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_),
                    np.sin(e_)], np.float32)
    up0 = np.array([0, 0, 1], np.float32)
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd); up /= np.linalg.norm(up)

    # orthographic framing from the projected bounding box; a fixed multiple of
    # the object size is what puts the camera inside the model
    ctr = np.array([H / 2, W / 2, D / 2], np.float32)
    pts = np.array([[i, j, k] for i in (0, H - 1) for j in (0, W - 1)
                    for k in (0, D - 1)], np.float32)
    rel = pts - ctr
    pr = rel @ right; pu = rel @ up; pf = rel @ fwd
    half = max(np.abs(pr).max(), np.abs(pu).max()) * 0.92 / zoom
    scale = size / (2 * half)
    print(f"  framing: half-extent {half:.1f} vox -> {scale:.3f} px/vox")

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xr = (xx - size / 2) / scale
    yu = -(yy - size / 2) / scale
    origin = ctr - fwd * (pf.max() + 2)
    nsteps = int((pf.max() - pf.min() + 6) / step)
    print(f"  {nsteps} march steps")

    C = np.zeros((size, size, 3), np.float32)
    T = np.ones((size, size), np.float32)
    L = np.array([-0.45, -0.68, 0.58], np.float32); L /= np.linalg.norm(L)
    L2 = np.array([0.55, 0.30, -0.35], np.float32); L2 /= np.linalg.norm(L2)

    for s in range(nsteps):
        t = s * step
        px = origin[0] + xr * right[0] + yu * up[0] + t * fwd[0]
        py = origin[1] + xr * right[1] + yu * up[1] + t * fwd[1]
        pz = origin[2] + xr * right[2] + yu * up[2] + t * fwd[2]
        valid = ((px >= 0) & (px < H - 1) & (py >= 0) & (py < W - 1) &
                 (pz >= 0) & (pz < D - 1))
        if not valid.any():
            continue
        xi = np.clip(px.astype(np.int32), 0, H - 2)
        yi = np.clip(py.astype(np.int32), 0, W - 2)
        zi = np.clip(pz.astype(np.int32), 0, D - 2)

        al = alpha[xi, yi, zi] * valid
        if al.max() <= 1e-6:
            continue
        col = rgb[xi, yi, zi]
        em = emis[xi, yi, zi]
        nn = g[xi, yi, zi]

        lam = np.clip((nn * L).sum(-1), 0, 1)
        lam2 = np.clip((nn * L2).sum(-1), 0, 1)
        # a rim term keeps the round bodies from flattening into discs
        shade = 0.26 + 0.88 * lam + 0.18 * lam2
        src = col * shade[..., None] + col * em[..., None]

        a_s = np.clip(1.0 - np.exp(-al * step * gain), 0, 1)[..., None]
        C += T[..., None] * a_s * src
        T *= (1.0 - a_s[..., 0])
        if T.max() < 0.004:
            break

    # filmic shoulder rather than a hard clip, so overlapping bodies do not blow
    # out to white holes
    img = 1.0 - np.exp(-np.clip(C, 0, None) * 2.2)
    img = np.clip(img, 0, 1)
    if bg != (0.0, 0.0, 0.0):
        img = img + np.asarray(bg, np.float32) * (1 - img)
    path = out
    Image.fromarray((img * 255).astype(np.uint8)).save(path)

    # pixel statistics -- a render is not judged by the log saying "saved"
    arr = np.asarray(Image.open(path).convert("RGB"))
    lit = int((arr.max(axis=2) > 10).sum())
    print(f"  wrote {path}")
    print(f"  mean {arr.mean():.1f}  max {arr.max()}  "
          f"corner {arr[5,5].tolist()}  lit {lit/(arr.shape[0]*arr.shape[1])*100:.1f}%")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(DOCS, "model3d_5k.json"))
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--tag", default="5k")
    ap.add_argument("--field", type=float, default=40.0, help="x-y field, microns")
    ap.add_argument("--vox", type=float, default=0.55, help="voxel size, microns")
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--az", type=float, default=38.0)
    ap.add_argument("--el", type=float, default=18.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    model = json.load(open(a.model))
    print(f"=== volume render {a.tag} ===")
    print(f"  model: {len(model['x'])} cells, z "
          f"{min(model['z']):+.1f}..{max(model['z']):+.1f} um")
    lab = build_grid(model, a.field, a.vox)
    n_cls = len(np.unique(lab)) - 1
    print(f"  classes present: {n_cls}")
    render(lab, os.path.join(a.outdir, f"{a.tag}_volume.png"),
           a.size, a.az, a.el, a.zoom)


if __name__ == "__main__":
    main()
