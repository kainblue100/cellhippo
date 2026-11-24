"""Can the Xenium_HE tiffs be registered to the cell centroids?

There is no he_imagealignment.csv for these samples -- they are not 10x-formatted
downloads.  So the question is whether an affine fit from the tissue outline can
place the cells on the H&E, or whether the only aligned image available is the
morphology MIP (which shares the cell coordinate frame by construction).

This checks the geometry of both.
"""
import json
import os

import numpy as np
import pandas as pd
import tifffile

RAW = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw/Xenium_LUAD_No2"
HE = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_HE/XeniumHE_LUAD_No2.tif"


def cell_extent():
    cells = pd.read_parquet(os.path.join(RAW, "cells.parquet"))
    print("  cells.parquet columns:", list(cells.columns)[:12])
    print("  n cells:", len(cells))
    for c in ("x_centroid", "y_centroid"):
        v = cells[c].to_numpy()
        print(f"    {c}: {v.min():.1f} .. {v.max():.1f}  span {v.max()-v.min():.1f}")
    return cells


def he_geometry():
    with tifffile.TiffFile(HE) as tf:
        s = tf.series[0]
        print(f"  HE series: {len(s.levels)} levels")
        for i, lv in enumerate(s.levels[:5]):
            print(f"    level {i}: {lv.shape} {lv.dtype}")
        ome = tf.ome_metadata
        print("  OME present:", ome is not None)
        if ome:
            import re
            m = re.findall(r'PhysicalSize[XY]="([^"]+)"', ome)
            print("  PhysicalSizeXY:", m)
            unit = re.findall(r'PhysicalSize[XY]Unit="([^"]+)"', ome)
            print("  units:", set(unit))


print("=== cell centroids ===")
cells = cell_extent()
print("\n=== Xenium HE ===")
he_geometry()
print("\n=== morphology_mip ===")
with tifffile.TiffFile(os.path.join(RAW, "morphology_mip.ome.tif")) as tf:
    s = tf.series[0]
    print(f"  levels: {len(s.levels)}")
    for i, lv in enumerate(s.levels[:5]):
        print(f"    level {i}: {lv.shape} {lv.dtype}")
