"""Rebuild HIPPO's figures from the fused tile clustering.

Three panels, each mapped to the figure in the original it stands in for:

  gallery   HIPPO Fig 2C  representative tiles per phenotype, in rows
  spatial   HIPPO Fig 5B  every tile drawn at its place on the slide, coloured by
                          phenotype, so the units can be read against the tissue
  heatmap   HIPPO Fig 3A  phenotype composition per slide, and phenotype vs the
                          annotated cell types present in the tile

The tiles in the gallery are the actual pixels the clustering saw (Reinhard
normalised), not reconstructions, so a reader can judge the phenotype by eye.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

MAP = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
       "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
       "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE")}

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
           "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#B07AA1", "#FF9DA7",
           "#9C755F", "#BAB0AC", "#EDC948", "#76B7B2", "#59A14F", "#F28E2B",
           "#E15759", "#4E79A7", "#A0CBE8", "#FFBE7D", "#8CD17D", "#F1CE63"]


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def gallery(img, pheno, path, n_per_row=8, thumb=112):
    """HIPPO Fig 2C: for each phenotype, the tiles nearest its centroid."""
    from PIL import Image, ImageDraw

    g = int(pheno.max()) + 1
    th, tw = img.shape[1], img.shape[2]
    sc = max(1, thumb // max(th, tw))
    sh, sw = th * sc, tw * sc
    pad, lab = 6, 46
    W = n_per_row * (sw + pad) + pad + lab
    H = g * (sh + pad) + pad + 30
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((lab, 8), f"representative H&E tiles per phenotype "
                      f"(8 nearest the phenotype mean, {th}x{tw} px, "
                      f"Reinhard-normalised)", fill=(40, 40, 40))
    for r in range(g):
        idx = np.where(pheno == r)[0]
        if len(idx) == 0:
            continue
        # towered pixel mean of the phenotype, to pick typical tiles
        mean = img[idx].astype(np.float32).mean(axis=0)
        d = ((img[idx].astype(np.float32) - mean) ** 2).mean(axis=(1, 2, 3))
        order = idx[np.argsort(d)][:n_per_row]
        y = pad + 30 + r * (sh + pad)
        dr.text((6, y + sh // 3), f"P{r}", fill=(25, 25, 25))
        dr.text((6, y + sh // 3 + 15), f"n={len(idx)}", fill=(120, 120, 120))
        for c, i in enumerate(order):
            x = lab + c * (sw + pad)
            canvas.paste(Image.fromarray(img[i]), (x, y))
            dr.rectangle([x, y, x + sw - 1, y + sh - 1], outline=(200, 200, 200))
    canvas.save(path, quality=93)
    return g


def spatial(tile_img_shape, pheno, meta, tile_px, path, bg=None):
    """HIPPO Fig 5B: every tile drawn where it sits on the slide."""
    from PIL import Image, ImageDraw

    if bg is None:
        H = (meta[:, 1].max() + 1) * tile_px
        W = (meta[:, 0].max() + 1) * tile_px
        base = Image.new("RGB", (W, H), (250, 250, 250))
    else:
        base = Image.fromarray(bg).convert("RGB")
        W, H = base.size
    scale = 1400 / max(W, H)
    ow, oh = int(W * scale), int(H * scale)
    base = base.resize((ow, oh), Image.LANCZOS)
    ov = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    step = max(1, int(tile_px * scale))
    for j in range(len(pheno)):
        c, r = int(meta[j, 0]), int(meta[j, 1])
        x0, y0 = int(c * tile_px * scale), int(r * tile_px * scale)
        col = rgb(PALETTE[int(pheno[j]) % len(PALETTE)])
        dr.rectangle([x0, y0, x0 + step, y0 + step],
                     fill=col + (140,), outline=col + (220,))
    out = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    out.save(path, quality=93)
    return out.size, ow, oh


def heatmap(profiles, slides, G, ann, ann_frac, path):
    from PIL import Image, ImageDraw

    A = len(ann)
    cw, ch = 44, 24
    W = 200 + max(len(slides), A) * cw + 30
    H = 90 + G * ch + 40 + G * ch + 60
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)

    def cell(x, y, v):
        col = (255, 255, 255) if v < 0.005 else (
            int(255 - 190 * v), int(235 - 120 * v), int(255 - 70 * v))
        dr.rectangle([x, y, x + cw - 2, y + ch - 2], fill=col,
                     outline=(228, 228, 228))
        if v >= 0.10:
            dr.text((x + 4, y + 6), f"{v*100:.0f}", fill=(20, 20, 20))

    dr.text((10, 8), "A. phenotype composition per slide (fraction of tiles)",
            fill=(35, 35, 35))
    for j, s in enumerate(slides):
        dr.text((200 + j * cw + 4, 32), str(s), fill=(60, 60, 60))
    for i in range(G):
        dr.text((10, 46 + i * ch + 5), f"P{i}", fill=(35, 35, 35))
        for j, s in enumerate(slides):
            v = profiles[s][i] if i < len(profiles[s]) else 0.0
            cell(200 + j * cw, 42 + i * ch, v)

    y2 = 42 + G * ch + 60
    dr.text((10, y2 - 24), "B. phenotype x annotated cell type "
                           "(mean fraction of the tile's cells)", fill=(35, 35, 35))
    for j, x in enumerate(ann):
        dr.text((200 + j * cw + 2, y2), str(x)[:10], fill=(60, 60, 60))
    for i in range(G):
        for j in range(A):
            cell(200 + j * cw, y2 + 14 + i * ch, float(ann_frac[i, j]))
    canvas.save(path, quality=93)
    return canvas.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    res = json.load(open(os.path.join(WORK, f"{a.name}.json")))
    pheno_all = np.array(res["tile_pheno"])
    meta_all = np.array(res["tile_meta"])
    slides_all = np.array(res["slides"])
    G = int(pheno_all.max()) + 1
    voc = list(res["annotations"])
    made = {}

    off = 0
    for t in a.tags:
        z = np.load(os.path.join(WORK, f"{t}_tiles.npz"))
        img = z["img"]
        tile_px = int(z["tile_px"][0])
        n = len(img)
        ph = pheno_all[off:off + n]
        mt = meta_all[off:off + n]
        off += n

        p = os.path.join(OUT, f"{a.name}_{t}_gallery.jpg")
        nph = gallery(img, ph, p)
        made[f"{t}_gallery"] = os.path.basename(p)
        print(f"  gallery {t}: {nph} phenotypes -> {p}")

        # spatial panel over the H&E at the tiling level
        import tifffile
        ver, name = MAP[t]
        he = os.path.join(TENX, f"{t}__{ver}__{name}", f"{name}_he_image.ome.tif")
        lvl_meta = json.load(open(os.path.join(WORK, f"{t}_tiles_meta.json")))
        bg = None
        if os.path.exists(he):
            with tifffile.TiffFile(he) as tf:
                arr = tf.series[0].levels[lvl_meta["level"]].asarray()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.moveaxis(arr, 0, -1)
            bg = arr[..., :3].astype(np.uint8)
        p2 = os.path.join(OUT, f"{a.name}_{t}_spatial.jpg")
        size, ow, oh = spatial(img.shape, ph, mt, tile_px, p2, bg)
        made[f"{t}_spatial"] = os.path.basename(p2)
        print(f"  spatial {t}: {size} -> {p2}")

        # phenotype x annotation, over the shared vocabulary: each slide's ann
        # array is laid out on its own label list (17 on 5k, 14 on lc), so its
        # columns must be scattered into the union before they can be pooled
        A = len(voc)
        idx = {v: i for i, v in enumerate(voc)}
        M = np.zeros((G, A))
        cnt = np.zeros(G)
        A_tag = np.zeros((n, A), dtype=np.float32)
        for j, lab in enumerate(z["annotations"]):
            A_tag[:, idx[str(lab)]] = z["ann"][:, j]
        for j in range(n):
            M[int(ph[j])] += A_tag[j]
            cnt[int(ph[j])] += 1
        M = M / np.maximum(cnt[:, None], 1)
        p3 = os.path.join(OUT, f"{a.name}_{t}_heatmap.jpg")
        size = heatmap(res["slide_profiles"], sorted(res["slide_profiles"]),
                       G, voc, M, p3)
        made[f"{t}_heatmap"] = os.path.basename(p3)
        print(f"  heatmap {t}: {size} -> {p3}")

    json.dump(made, open(os.path.join(OUT, f"figures_{a.name}.json"), "w"),
              indent=2)
    print(f"\nwrote {len(made)} figures to {OUT}")


if __name__ == "__main__":
    main()
