"""Directed pattern migration, per slide.

Annotile's version of this has three parts, and all three are computable from
what the pipeline already produced:

  1. a directed graph per slide: nodes are patterns, an edge is allowed only where
     the two patterns actually share >= 3 adjacent windows on the section, and the
     direction is up the pseudotime axis;
  2. each pattern attaches to its *upstream* neighbour at the shortest feature
     distance -- the adjacency says what is in contact, the feature distance says
     which way the contact goes. A pattern with no upstream neighbour is a root;
  3. arrows drawn on the registered H&E from the upstream pattern's centre of mass
     to the downstream, with the root-to-farthest path drawn heavier because that
     is the branch the animation follows.

What this is not: a lineage. Every slide is a different donor, the pseudotime is
fitted to one section, and cells are never followed from one window to the next.
It is a directed description of which pattern sits upstream of which *on this
section*, and it should be read that way.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OUT = os.path.join(FIGS, "migration")

MAP = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
       "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
       "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE")}

PALETTE = ["#c2410c", "#1d4ed8", "#0f766e", "#7c3aed", "#db2777", "#a16207",
           "#0369a1", "#15803d"]

MIN_SHARED = 3          # Annotile's rule: >= 3 adjacent windows to allow an edge


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def font(size=20):
    for n in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def zscore(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd


def slide_block(tag, name):
    """Per-slide features, phenotype labels, and the tile grid."""
    res = json.load(open(os.path.join(WORK, f"{name}.json"), encoding="utf-8"))
    pheno = np.array(res["tile_pheno"])
    meta = np.array(res["tile_meta"])
    slides = np.array(res["slides"])
    off = 0
    for s in res["tags"]:
        if s == tag:
            break
        off += int((slides == s).sum())
    n = int((slides == tag).sum())
    ph = pheno[off:off + n]
    mt = meta[slides == tag]

    z = np.load(os.path.join(WORK, f"{tag}_tiles.npz"))
    vocab = [str(v) for v in res["annotations"]]
    idx = {v: i for i, v in enumerate(vocab)}
    A = np.zeros((len(z["ann"]), len(vocab)), dtype=np.float64)
    for j, lab in enumerate(z["annotations"]):
        A[:, idx[str(lab)]] = z["ann"][:, j]
    blocks = [zscore(A), zscore(z["expr"].astype(np.float64))]
    emb = os.path.join(WORK, f"{tag}_cnn.npz")
    if os.path.exists(emb):
        blocks.append(zscore(np.load(emb)["emb"].astype(np.float64)))
    X = np.hstack(blocks)
    return X, ph, mt


def grid_of(ph, mt):
    return {(int(c), int(r)): int(p) for (c, r, _n, _u), p in zip(mt, ph)}


def adjacency(grid):
    """Adjacent-window pairs with different patterns (right and down only)."""
    pairs = Counter()
    for (ix, iy), p in grid.items():
        for dx, dy in ((1, 0), (0, 1)):
            q = grid.get((ix + dx, iy + dy))
            if q is None or q == p:
                continue
            pairs[(min(p, q), max(p, q))] += 1
    return pairs


def build_tree(ks, med, pairs, cen):
    """Orient the adjacency graph by pseudotime and hang each pattern on its
    closest upstream neighbour.

    `pairs` gives the contacts that actually exist on the section, so no edge is
    invented between patterns that never touch.  Among a pattern's upstream
    contacts, the one with the shortest feature distance is its parent -- the
    same choice Annotile makes.
    """
    adj = {k: [] for k in ks}
    for (a, b), n in pairs.items():
        if n < MIN_SHARED:
            continue
        adj[a].append(b)
        adj[b].append(a)

    root = min(ks, key=lambda k: med[k])
    parent = {root: None}
    order = []
    # process in pseudotime order so a parent always exists already
    for k in sorted(ks, key=lambda k: med[k]):
        if k == root:
            order.append(k)
            continue
        ups = [j for j in adj[k] if med[j] < med[k] and (j in parent or j == root)]
        if ups:
            p = min(ups, key=lambda j: float(np.linalg.norm(cen[k] - cen[j])))
        else:
            # no upstream contact: fall back to the closest pattern already placed
            placed = [j for j in parent]
            p = min(placed, key=lambda j: float(np.linalg.norm(cen[k] - cen[j])))
        parent[k] = p
        order.append(k)

    children = {k: [] for k in ks}
    for k, p in parent.items():
        if p is not None:
            children[p].append(k)

    # the branch the animation follows: root -> the pattern with the largest
    # pseudotime, walking down the tree
    far = max(ks, key=lambda k: med[k])
    branch, cur = [], far
    while cur is not None:
        branch.append(cur)
        cur = parent[cur]
    branch = branch[::-1]
    return parent, children, root, branch


def draw_arrows(im, grid, cen, parent, branch, tile_px, legend=True):
    """White arrows from each upstream pattern's centre of mass to its children."""
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(max(14, int(im.width * 0.016)))
    for k, p in parent.items():
        if p is None:
            continue
        heavy = (k in branch) and (p in branch)
        x0, y0 = cen[p]
        x1, y1 = cen[k]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1:
            continue
        ux, uy = dx / L, dy / L
        w = max(2, int(im.width * (0.006 if heavy else 0.003)))
        sh = max(8, int(im.width * (0.022 if heavy else 0.015)))
        # stop short of both nodes so the arrowhead does not sit under a dot
        sx, sy = x0 + ux * sh * 0.9, y0 + uy * sh * 0.9
        ex, ey = x1 - ux * sh * 1.05, y1 - uy * sh * 1.05
        if math.hypot(ex - sx, ey - sy) < 4:
            continue
        col = (255, 255, 255, 255) if heavy else (255, 255, 255, 205)
        dark = (30, 26, 34, 210)
        dr.line([(sx, sy), (ex, ey)], fill=dark, width=w + 4)
        dr.line([(sx, sy), (ex, ey)], fill=col, width=w)
        # head
        hx, hy = ex, ey
        px, py = -uy, ux
        hs = sh * 1.5
        dr.polygon([(hx, hy),
                    (hx - ux * hs + px * hs * 0.5, hy - uy * hs + py * hs * 0.5),
                    (hx - ux * hs - px * hs * 0.5, hy - uy * hs - py * hs * 0.5)],
                   fill=dark)
        dr.polygon([(hx, hy),
                    (hx - ux * hs * 0.82 + px * hs * 0.42,
                     hy - uy * hs * 0.82 + py * hs * 0.42),
                    (hx - ux * hs * 0.82 - px * hs * 0.42,
                     hy - uy * hs * 0.82 - py * hs * 0.42)],
                   fill=col)

    if legend:
        pad = int(im.width * 0.012)
        lh = int(f.size * 1.7)
        sw = int(im.width * 0.011)
        tw = int(im.width * 0.060)
        dr.rectangle([pad, pad, pad + tw, pad + lh], fill=(255, 255, 255, 225))
        lx, ly = pad + int(tw * 0.18), pad + int(lh * 0.5)
        dr.line([(lx, ly), (lx + sw * 2, ly)], fill=(30, 26, 34, 230), width=6)
        dr.line([(lx, ly), (lx + sw * 2, ly)], fill=(255, 255, 255, 255),
                width=max(2, int(im.width * 0.003)))
        dr.text((lx + sw * 3, ly - f.size * 0.62), "演化方向（上游 → 下游）",
                fill=(28, 24, 32), font=f)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["5k", "lc", "luad"])
    ap.add_argument("--name", default="fused")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--max-px", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    traj = json.load(open(os.path.join(WORK, "trajectory", "trajectory.json"),
                          encoding="utf-8"))["slides"]
    payload = {"method": {
        "edge": f"two patterns must share >= {MIN_SHARED} adjacent windows",
        "direction": "up the diffusion-pseudotime axis",
        "parent": "among the upstream contacts, the shortest feature distance",
        "root": "lowest median pseudotime on that slide",
        "branch": "root -> the pattern with the largest pseudotime",
        "not_a_lineage": "single sections, different donors, cells never followed",
    }, "slides": {}}

    for tag in a.tags:
        if tag not in traj:
            print(f"  {tag}: no trajectory, skipped")
            continue
        X, ph, mt = slide_block(tag, a.name)
        grid = grid_of(ph, mt)
        ks = sorted(set(int(v) for v in ph))
        med = {int(k): float(v) for k, v in traj[tag]["pseudotime_median"].items()}
        pairs = adjacency(grid)

        cen_feat = {k: X[ph == k].mean(axis=0) for k in ks}
        # centre of mass on the section, in tile units
        cen = {k: (float(mt[ph == k, 0].mean()), float(mt[ph == k, 1].mean()))
               for k in ks}
        parent, children, root, branch = build_tree(ks, med, pairs, cen_feat)

        opt = {k: [j for j in children[k]] for k in ks}
        print(f"\n=== {tag} ===  {len(ks)} patterns, root P{root}")
        print("  median pseudotime: " +
              ", ".join(f"P{k}={med[k]:.3f}" for k in ks))
        print("  contacts (>= "
              f"{MIN_SHARED} adjacent windows): " +
              ", ".join(f"P{x}-P{y} {n}" for (x, y), n in
                        sorted(pairs.items(), key=lambda kv: -kv[1])
                        if n >= MIN_SHARED))
        for k in ks:
            p = parent[k]
            if p is None:
                print(f"    P{k}  root")
            else:
                d = float(np.linalg.norm(cen_feat[k] - cen_feat[p]))
                print(f"    P{k}  <- P{p}   feature distance {d:.2f}"
                      + ("   [video branch]" if (k in branch and p in branch)
                         else ""))
        print("  video branch: " + " → ".join(f"P{k}" for k in branch))

        # ---- slide map with the arrows, rotated to landscape if the slide is tall
        ver, nm = MAP[tag]
        reg = json.load(open(os.path.join(FIGS, "registration_final.json"),
                             encoding="utf-8"))[tag]
        lvl = json.load(open(os.path.join(WORK, f"{tag}_tiles_meta.json"),
                             encoding="utf-8"))
        level, tp = lvl["level"], lvl["tile_px"]
        he = os.path.join(TENX, f"{tag}__{ver}__{nm}", f"{nm}_he_image.ome.tif")
        with tifffile.TiffFile(he) as tf:
            arr = tf.series[0].levels[level].asarray()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3].astype(np.uint8)

        scale = min(1.0, a.max_px / max(arr.shape[:2]))
        ow, oh = int(arr.shape[1] * scale), int(arr.shape[0] * scale)
        im = Image.fromarray(arr).resize((ow, oh), Image.LANCZOS).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        step = max(3, int(tp * scale))
        for (ix, iy), p in grid.items():
            x0, y0 = int(ix * tp * scale), int(iy * tp * scale)
            col = rgb(PALETTE[p % len(PALETTE)])
            dr.rectangle([x0, y0, min(ow, x0 + step), min(oh, y0 + step)],
                         fill=col + (120,), outline=col + (200,))
        cen_px = {k: (cen[k][0] * tp * scale, cen[k][1] * tp * scale)
                  for k in ks}
        draw_arrows(im, grid, cen_px, parent, branch, tp)
        if im.height > im.width:
            im = im.rotate(90, expand=True)
        p_out = os.path.join(a.outdir, f"{tag}_migration.png")
        im.save(p_out)
        print(f"  wrote {p_out}  {im.size}")

        payload["slides"][tag] = {
            "patterns": ks,
            "median_pseudotime": {str(k): med[k] for k in ks},
            "parent": {str(k): (None if parent[k] is None else parent[k])
                       for k in ks},
            "root": root,
            "video_branch": branch,
            "contacts": {f"{x}-{y}": int(n) for (x, y), n in pairs.items()},
            "arrows": [{"from": parent[k], "to": k,
                        "heavy": (k in branch and parent[k] in branch),
                        "distance": float(np.linalg.norm(cen_feat[k] -
                                                         cen_feat[parent[k]]))}
                       for k in ks if parent[k] is not None],
            "image": os.path.basename(p_out),
            "image_wh": list(im.size),
        }

    json.dump(payload, open(os.path.join(a.outdir, "migration.json"), "w"),
              indent=2)
    print(f"\nwrote {a.outdir}/migration.json")


if __name__ == "__main__":
    main()
