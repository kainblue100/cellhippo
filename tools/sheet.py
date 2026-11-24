"""Assemble the per-pattern renders into the 2xN plate.

Renders come out one file per pattern and view; the plate puts every pattern's
glass block on the top row and the close-up on the bottom, matching how the
reference figure is laid out.
"""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def label_font(size=26):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=r"D:\coda_matlab\cellhippo\figures\volume\scene.json")
    ap.add_argument("--dir", default=r"D:\coda_matlab\cellhippo\figures\volume\blender")
    ap.add_argument("--out", default=r"D:\coda_matlab\cellhippo\figures\volume\sheet_cube.png")
    ap.add_argument("--view", default="cube", choices=["cube", "close"])
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=420)
    a = ap.parse_args()

    scene = json.load(open(a.scene))
    models = scene["patterns"]
    tiles = []
    for m in models:
        stem = f"{m['pattern']}_{m['slice']}"
        p = os.path.join(a.dir, f"{stem}_{a.view}.png")
        if os.path.exists(p):
            tiles.append((stem, Image.open(p).convert("RGB")))
        else:
            print(f"  missing {p}")

    if not tiles:
        raise SystemExit("nothing to assemble")

    cols = min(a.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    cell = a.cell
    pad = 8
    sheet = Image.new("RGB", (cols * (cell + pad) + pad,
                              rows * (cell + pad) + pad), "white")
    dr = ImageDraw.Draw(sheet)
    font = label_font(24)
    for i, (stem, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        im = im.resize((cell, cell), Image.LANCZOS)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + pad)
        sheet.paste(im, (x, y))
        dr.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(228, 224, 232))
        dr.text((x + 10, y + 8), stem, fill=(40, 34, 44), font=font)
    sheet.save(a.out)
    print(f"wrote {a.out}  {len(tiles)} tiles, {cols}x{rows}")


if __name__ == "__main__":
    main()
