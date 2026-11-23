"""Crop the H&E behind each pattern's cell block and save it as a backdrop plate.

The renderer puts this plate on a plane behind the glass box, so the 3D block
reads as sitting on its own piece of tissue rather than floating in grey.  Same
idea as Annotile's export_backdrops.py, adapted to this project's registration:
the patch is defined in micron slide coordinates and mapped to H&E pixels through
the transform already verified in register_final.py.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
from PIL import Image, ImageEnhance, ImageFilter

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

MAP = {"5k": ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
       "lc": ("2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
       "luad": ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(WORK, "volume", "scene.json"))
    ap.add_argument("--margin", type=float, default=140.0,
                    help="micron margin around the patch; the first run used 60 "
                         "and the resulting plate was so magnified that the "
                         "tissue structure was unreadable")
    ap.add_argument("--level", type=int, default=1,
                    help="pyramid level to crop from; level 1 is 2x sharper than "
                         "the tiling level and still fits in memory (level 0 is "
                         "3.5 GB for the 5k slide)")
    ap.add_argument("--size", type=int, default=1100)
    ap.add_argument("--blur", type=float, default=9.0,
                    help="Gaussian radius on the plate, in output pixels.  The "
                         "plate is meant to read as out-of-focus tissue; relying "
                         "on Blender's thin-lens maths for that was unpredictable "
                         "at this scene scale, so the defocus is baked in.")
    ap.add_argument("--desat", type=float, default=0.18,
                    help="pull saturation down slightly so the plate sits behind "
                         "the subject instead of competing with it")
    a = ap.parse_args()

    scene = json.load(open(a.scene))
    outdir = os.path.join(os.path.dirname(a.scene), "backdrops")
    os.makedirs(outdir, exist_ok=True)
    reg = json.load(open(os.path.join(FIGS, "registration_final.json")))

    cache = {}
    for model in scene["patterns"]:
        tag = model["slice"]
        ver, name = MAP[tag]
        r = reg[tag]
        ref_level = r["level"]
        level = a.level
        # scale and translation both live in the pixels of the level the
        # registration was solved at.  Moving to a finer level doubles them, and
        # forgetting the translation is what sent two 5k crops off the image.
        zoom = 2.0 ** (ref_level - level)
        s_lvl = (r["scale_L0"] / (2 ** ref_level)) * zoom
        R = np.array(r["R"])
        t = np.array(r["t"]) * zoom

        if tag not in cache:
            he = os.path.join(TENX, f"{tag}__{ver}__{name}",
                              f"{name}_he_image.ome.tif")
            with tifffile.TiffFile(he) as tf:
                arr = tf.series[0].levels[level].asarray()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.moveaxis(arr, 0, -1)
            cache[tag] = arr[..., :3].astype(np.uint8)
            cache[tag + "_wh"] = (arr.shape[1], arr.shape[0])
            print(f"  loaded {tag} level {level} {cache[tag + '_wh']}")
        img = cache[tag]
        W, H = cache[tag + "_wh"]

        # patch corners in microns -> H&E pixels.  R is the 2x2 rotation from
        # register_final.json: apply it to the (x, y) pair, then scale and add
        # the translation, rather than treating the points as homogeneous.
        x0 = model["x0_um"] - a.margin
        y0 = model["y0_um"] - a.margin
        x1 = model["x0_um"] + model["patch_um"] + a.margin
        y1 = model["y0_um"] + model["patch_um"] + a.margin
        corners = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)],
                           dtype=float)
        px = (R @ corners.T).T * s_lvl + t
        cx0, cx1 = int(np.floor(px[:, 0].min())), int(np.ceil(px[:, 0].max()))
        cy0, cy1 = int(np.floor(px[:, 1].min())), int(np.ceil(px[:, 1].max()))
        cx0, cx1 = max(0, cx0), min(W, cx1)
        cy0, cy1 = max(0, cy0), min(H, cy1)
        if cx1 <= cx0 or cy1 <= cy0:
            print(f"  {model['pattern']}: empty crop, skipped")
            continue
        crop = img[cy0:cy1, cx0:cx1]
        im = Image.fromarray(crop).resize((a.size, a.size),
                                          Image.Resampling.LANCZOS)
        if a.blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(a.blur))
        if a.desat > 0:
            im = ImageEnhance.Color(im).enhance(1.0 - a.desat)
        path = os.path.join(outdir, os.path.basename(model["backdrop"]))
        im.save(path)
        model["backdrop"] = os.path.relpath(path, os.path.dirname(a.scene))
        print(f"  {model['pattern']} {tag}: crop {crop.shape[:2]} -> {path}")

    json.dump(scene, open(a.scene, "w"), indent=1)
    print(f"updated {a.scene}")


if __name__ == "__main__":
    main()
