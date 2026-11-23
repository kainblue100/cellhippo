"""Assemble the pattern renders into a movie along the diffusion pseudotime axis.

This is the "morphological movie" that Vig, Wu & Lotter (arXiv:2601.21334)
propose: sample along a pseudotime axis fitted to static snapshots and read the
resulting sequence as a progression.  It is a movie of an ordering, not of
observed time — the axis is a diffusion distance, and its direction is fixed by an
assumed root.

Layout per frame
  left    the 3D cell model of whichever pattern the axis is passing through,
          cross-dissolved at the handover
  right   the axis itself, with the five pattern bands and a travelling marker
  bottom  cell-type composition of the current pattern

Encoded to both MP4 (H.264 via imageio-ffmpeg) and GIF, because one is small and
the other plays everywhere without a plugin.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
VOL = "/mnt/storage/bzhang/AIVC/cellhippo/figures/volume"

PATTERN_COLOR = {0: "#c2410c", 1: "#1d4ed8", 2: "#0f766e", 3: "#7c3aed",
                 4: "#db2777"}


def font(size, bold=False):
    for name in ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                 "arialbd.ttf" if bold else "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit(im, w, h):
    im = im.copy()
    im.thumbnail((w, h), Image.LANCZOS)
    return im


def build_frames(tag, traj, renders, comps, outdir, size=1100, fps=12,
                 frames_per_seg=14, meta=None, lung=None):
    meta = meta or {}
    slide = traj["slides"][tag]
    order = [int(p) for p in slide["order"]]
    med = {int(k): v for k, v in slide["pseudotime_median"].items()}
    band = {int(k): v for k, v in slide["bands"].items()}
    stab = slide["order_stability"]

    W = H = size
    left_w = int(W * 0.60)
    panel_x = left_w + 24
    panel_w = W - panel_x - 24

    frames = []
    n_seg = len(order) - 1
    for seg in range(n_seg):
        a, b = order[seg], order[seg + 1]
        for f in range(frames_per_seg):
            t = f / frames_per_seg
            # ease so the marker dwells on a pattern rather than sliding past it
            te = 0.5 - 0.5 * np.cos(np.pi * t)
            canvas = Image.new("RGB", (W, H), "white")
            dr = ImageDraw.Draw(canvas)

            ra, rb = renders.get((tag, a)), renders.get((tag, b))
            if ra is None and rb is None:
                continue
            if ra is None:
                ra = rb
            if rb is None:
                rb = ra
            ia = fit(ra, left_w, left_w)
            ib = fit(rb, left_w, left_w)
            ia = ia.resize((left_w, left_w), Image.LANCZOS)
            ib = ib.resize((left_w, left_w), Image.LANCZOS)
            mixed = Image.blend(ia, ib, float(te))

            panel = Image.new("RGB", (left_w, left_w), "white")
            if lung is not None:
                # The lung silhouette covers only ~30% of its frame and the
                # inset covers 60%^2 = 36%, so a 60% inset hid the lung almost
                # entirely -- the first check found pure white in every corner.
                # The inset is 50% now and the lung is composited through its
                # alpha rather than by convert("RGB"), which discards it.
                # the crop is not square, so fit by its long side and centre --
                # stretching it to the panel would distort the lung
                lg = lung.copy()
                lg.thumbnail((left_w, left_w), Image.LANCZOS)
                white = Image.new("RGBA", (left_w, left_w), (255, 255, 255, 255))
                white.paste(lg, ((left_w - lg.width) // 2,
                                 (left_w - lg.height) // 2), lg)
                panel = Image.blend(Image.new("RGB", (left_w, left_w), "white"),
                                    white.convert("RGB"), 0.62)
                inner = int(left_w * 0.42)
                small = mixed.resize((inner, inner), Image.LANCZOS)
                off = (left_w - inner) // 2
                panel.paste(small, (off, off))
            else:
                panel = mixed
            canvas.paste(panel, (0, (H - left_w) // 2))

            cur = a if te < 0.5 else b
            dr.text((panel_x, 24), f"{tag}   P{cur}", fill=(26, 26, 30),
                    font=font(34, True))
            dr.text((panel_x, 70), "diffusion pseudotime over window",
                    fill=(112, 112, 120), font=font(15))
            dr.text((panel_x, 89), "embeddings  (Haghverdi 2016)",
                    fill=(112, 112, 120), font=font(15))

            # the axis
            ax_y = 178
            lo, hi = min(med.values()), max(med.values())
            span = max(1e-6, hi - lo)
            dr.line([(panel_x, ax_y), (panel_x + panel_w, ax_y)],
                    fill=(70, 70, 78), width=3)
            # labels alternate above and below so neighbours cannot collide
            for j, k in enumerate(order):
                p10 = band[k]["p10"]
                p90 = band[k]["p90"]
                x0 = panel_x + (p10 - lo) / span * panel_w
                x1 = panel_x + (p90 - lo) / span * panel_w
                x0, x1 = max(panel_x, x0), min(panel_x + panel_w, x1)
                col = PATTERN_COLOR.get(k, "#888888")
                dr.rectangle([x0, ax_y - 14, max(x1, x0 + 3), ax_y + 14],
                             fill=col)
                xm = panel_x + (med[k] - lo) / span * panel_w
                up = (j % 2 == 0)
                ty = ax_y - 30 if up else ax_y + 20
                dr.text((min(max(xm - 9, panel_x), panel_x + panel_w - 24), ty),
                        f"P{k}", fill=(46, 46, 54), font=font(15, True))
            # travelling marker
            cur_pt = med[a] + (med[b] - med[a]) * te
            xm = panel_x + (cur_pt - lo) / span * panel_w
            dr.ellipse([xm - 10, ax_y - 10, xm + 10, ax_y + 10],
                       outline=(18, 18, 22), width=3)
            dr.text((panel_x, ax_y + 48),
                    f"pseudotime {cur_pt:.3f}   root P{slide['root']}"
                    f"   {int(np.sum(np.asarray(slide['window_pattern']) == cur))}"
                    f" of {slide['n_windows']} windows",
                    fill=(112, 112, 120), font=font(14))

            # composition of the current pattern
            comp = comps.get((tag, cur))
            bar_y = ax_y + 108
            if comp:
                dr.text((panel_x, bar_y - 28), "cell types in this block",
                        fill=(112, 112, 120), font=font(14))
                shown = sorted(comp.items(), key=lambda kv: -kv[1])[:7]
                mx = max(v for _, v in shown) or 1.0
                bar_room = panel_w - 210
                for i, (name, v) in enumerate(shown):
                    y = bar_y + i * 32
                    w = (v / mx) * bar_room
                    dr.rectangle([panel_x + 150, y, panel_x + 150 + w, y + 18],
                                 fill=(198, 190, 206))
                    dr.text((panel_x, y - 1), name[:17], fill=(58, 58, 66),
                            font=font(13))
                    # right-aligned counts so the numbers form a column
                    dr.text((panel_x + 150 + bar_room + 14, y - 1),
                            f"{v*100:.0f}%", fill=(106, 106, 114),
                            font=font(13))
                info_y = bar_y + len(shown) * 32 + 26
                dr.text((panel_x, info_y),
                        f"{meta.get((tag, cur), {}).get('cells', '?')} cells in "
                        f"{meta.get((tag, cur), {}).get('patch', '?')} um block",
                        fill=(140, 138, 146), font=font(13))

            # provenance under the panel: the silhouette is an atlas, the inset
            # is the measurement, and the label says which is which
            if lung is not None:
                dr.text((14, H - 30),
                        "pale lung = reference anatomy (BodyParts3D), not this "
                        "specimen;  inset = measured block",
                        fill=(150, 140, 148), font=font(12))

            # provenance, because the ordering is a model and should say so
            dr.text((panel_x, H - 74),
                    "ordering only — no velocity: Xenium has mature transcripts",
                    fill=(150, 140, 148), font=font(13))
            stable = [f"P{k2.split('-')[0]}-P{k2.split('-')[1]} {v:.0%}"
                      for k2, v in sorted(stab.items(), key=lambda kv: -kv[1])][:2]
            dr.text((panel_x, H - 54), "most stable pairs: " + ", ".join(stable),
                    fill=(150, 140, 148), font=font(13))
            frames.append(canvas)

    os.makedirs(outdir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(outdir, f"{tag}_{i:04d}.png"))

    mp4 = os.path.join(outdir, f"{tag}_evolution.mp4")
    try:
        import imageio.v2 as iio
        iio.mimwrite(mp4, [np.asarray(f) for f in frames],
                     fps=fps, codec="libx264", quality=9,
                     macro_block_size=1)
        print(f"  wrote {mp4}")
    except Exception as e:
        print(f"  mp4 failed: {type(e).__name__}: {e}")

    gif = os.path.join(outdir, f"{tag}_evolution.gif")
    small = [f.resize((size // 2, size // 2), Image.LANCZOS) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  wrote {gif}  ({len(frames)} frames)")
    return mp4, gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(WORK, "trajectory",
                                                   "trajectory.json"))
    ap.add_argument("--volume", default=os.path.join(WORK, "volume", "scene.json"))
    ap.add_argument("--renderdir", default=os.path.join(VOL, "blender"))
    ap.add_argument("--outdir", default=os.path.join(VOL, "movie"))
    ap.add_argument("--size", type=int, default=1100)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--lung", default=os.path.join(VOL, "lung_silhouette.png"),
                    help="reference-anatomy plate; omit for no backdrop")
    a = ap.parse_args()

    traj = json.load(open(a.traj))
    scene = json.load(open(a.volume))
    renders, comps, meta = {}, {}, {}
    for m in scene["patterns"]:
        tag, pat = m["slice"], int(m["pattern"].lstrip("P"))
        p = os.path.join(a.renderdir, f"P{pat}_{tag}_cube.png")
        if os.path.exists(p):
            renders[(tag, pat)] = Image.open(p).convert("RGB")
        agg = {}
        for c in m["cells"]:
            agg[c["type"]] = agg.get(c["type"], 0) + 1
        tot = sum(agg.values()) or 1
        comps[(tag, pat)] = {k: v / tot for k, v in agg.items()}
        meta[(tag, pat)] = {"cells": m["cells_in_patch"],
                            "patch": int(m["patch_um"])}

    lung = None
    if a.lung and os.path.exists(a.lung):
        raw = Image.open(a.lung).convert("RGBA")
        # crop to the alpha bounding box first: the render carries a wide empty
        # margin, so resizing the full frame left the lung small and pale inside
        # the panel and the check found only a 5-level tint
        bbox = raw.getchannel("A").getbbox()
        lung = raw.crop(bbox) if bbox else raw
        print(f"lung backdrop: {raw.size} -> cropped {lung.size}")

    for tag in traj["slides"]:
        print(f"=== {tag} ===")
        build_frames(tag, traj, renders, comps, a.outdir, a.size, a.fps,
                     meta=meta, lung=lung)


if __name__ == "__main__":
    main()
