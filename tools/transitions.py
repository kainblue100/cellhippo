"""Where the patterns hand over to each other, drawn on the slide.

A pattern map says which pattern sits where.  It does not say which patterns touch.
That is a different and directly countable object: for every pair of adjacent
windows assigned to different patterns, there is one transition.  Counting them
gives an adjacency matrix of the tissue itself — no embedding, no pseudotime, no
model — and drawing the boundaries on the registered H&E shows whether a pattern
changes gradually along a front or interleaves in a mosaic.

This is the spatial counterpart to the embedding plots: UMAP and t-SNE say which
patterns are close in feature space, this says which are close on the section.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import tifffile
from PIL import Image, ImageDraw

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = os.path.join(FIGS, "transitions")

MAP = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
       "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
       "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE")}

PATTERN_RGB = {0: (194, 65, 12), 1: (29, 78, 216), 2: (15, 118, 110),
               3: (124, 58, 237), 4: (219, 39, 119), 5: (180, 83, 9),
               6: (2, 132, 199), 7: (22, 163, 74)}


def build_grid(tag, name):
    res = json.load(open(os.path.join(WORK, f"{name}.json")))
    pheno = np.array(res["tile_pheno"])
    meta = np.array(res["tile_meta"])
    slides = np.array(res["slides"])
    off = 0
    for s in res["tags"]:
        n_s = int((slides == s).sum())
        if s == tag:
            break
        off += n_s
    ph = pheno[off:off + int((slides == tag).sum())]
    mt = meta[slides == tag]
    return {(int(c), int(r)): int(p) for (c, r, n, k), p in zip(mt, ph)}


def transition_matrix(grid):
    """Adjacent-window pairs with different patterns, counted both ways."""
    pairs = Counter()
    edges = []
    for (ix, iy), p in grid.items():
        for dx, dy in ((1, 0), (0, 1)):
            q = grid.get((ix + dx, iy + dy))
            if q is None or q == p:
                continue
            pairs[(min(p, q), max(p, q))] += 1
            edges.append((ix, iy, dx, dy, p, q))
    return pairs, edges


def draw_slide(tag, grid, pairs, edges, outdir, max_px=1800):
    ver, nm = MAP[tag]
    reg = json.load(open(os.path.join(FIGS, "registration_final.json")))[tag]
    lvl = json.load(open(os.path.join(WORK, f"{tag}_tiles_meta.json")))
    level = lvl["level"]
    tp = lvl["tile_px"]

    he = os.path.join(TENX, f"{tag}__{ver}__{nm}", f"{nm}_he_image.ome.tif")
    with tifffile.TiffFile(he) as tf:
        arr = tf.series[0].levels[level].asarray()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3].astype(np.uint8)
    H, W = arr.shape[:2]

    # Downsample FIRST, then draw.  Drawing 3 px lines on the 6680x10818 level
    # and resizing by 0.166 turned them into half-pixel ghosts -- the check
    # measured 0.00% of pixels darker than 90, i.e. no lines at all.
    scale = min(1.0, max_px / max(W, H))
    ow, oh = int(W * scale), int(H * scale)
    im = Image.fromarray(arr).resize((ow, oh), Image.LANCZOS)
    base = np.asarray(im).astype(np.float32)

    S = lambda v: int(round(v * scale))
    tps = max(2, S(tp))

    # tint window by window, in output coordinates
    overlay = np.zeros_like(base)
    mask = np.zeros((oh, ow), dtype=bool)
    for (ix, iy), p in grid.items():
        x0, y0 = S(ix * tp), S(iy * tp)
        x1, y1 = min(ow, x0 + tps), min(oh, y0 + tps)
        if x1 <= x0 or y1 <= y0:
            continue
        overlay[y0:y1, x0:x1] = PATTERN_RGB.get(p, (128, 128, 128))
        mask[y0:y1, x0:x1] = True
    mixed = np.where(mask[..., None],
                     base * 0.62 + overlay * 0.38, base).astype(np.uint8)
    im = Image.fromarray(mixed).convert("RGB")
    dr = ImageDraw.Draw(im)

    # handover bars, now several output pixels wide
    lw = max(2, int(round(tps * 0.16)))
    for ix, iy, dx, dy, p, q in edges:
        x0, y0 = S(ix * tp), S(iy * tp)
        x1, y1 = x0 + tps, y0 + tps
        if dx == 1:
            sx = min(ow - 1, x1)
            dr.line([(sx, y0), (sx, min(oh - 1, y1))], fill=(16, 16, 20),
                    width=lw)
        else:
            sy = min(oh - 1, y1)
            dr.line([(x0, sy), (min(ow - 1, x1), sy)], fill=(16, 16, 20),
                    width=lw)

    p_out = os.path.join(outdir, f"{tag}_transitions.png")
    im.save(p_out)

    # measure rather than assume: the lines must be present this time
    a = np.asarray(im)
    dark = float((a.max(axis=2) < 90).mean())
    print(f"  {tag}: {len(grid)} windows, {len(edges)} handovers, "
          f"{ow}x{oh}, line width {lw}px, dark pixels {dark*100:.2f}%")
    print(f"       -> {p_out}")
    return p_out, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["5k", "lc"])
    ap.add_argument("--name", default="fused")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    payload = {"method": {
        "unit": "adjacent-window pairs assigned to different patterns",
        "edges": "right and down neighbours only, so each pair is counted once",
    }, "slides": {}}

    for tag in a.tags:
        grid = build_grid(tag, a.name)
        pairs, edges = transition_matrix(grid)
        ks = sorted(set(grid.values()))
        ng = max(ks) + 1 if ks else 0
        M = np.zeros((ng, ng), dtype=int)
        for (p, q), n in pairs.items():
            if p < ng and q < ng:
                M[p, q] = M[q, p] = n
        print(f"\n=== {tag} ===")
        print(f"  transition matrix (adjacent windows, different patterns):")
        print("      " + "".join(f"P{j:>7}" for j in range(ng)))
        for i in range(ng):
            print(f"   P{i} " + "".join(f"{M[i, j]:7d}" for j in range(ng)))
        top = pairs.most_common(6)
        print("  most common handovers: " + ", ".join(
            f"P{p}-P{q} {n}" for (p, q), n in top))

        p_out, shape = draw_slide(tag, grid, pairs, edges, a.outdir)
        payload["slides"][tag] = {
            "windows": len(grid),
            "handovers": len(edges),
            "matrix": M.tolist(),
            "pairs": {f"{p}-{q}": int(n) for (p, q), n in pairs.most_common()},
            "image_shape": list(shape),
            "figure": os.path.basename(p_out),
        }

    json.dump(payload, open(os.path.join(a.outdir, "transitions.json"), "w"),
              indent=2)
    print(f"\nwrote {a.outdir}/transitions.json")


if __name__ == "__main__":
    main()
