"""Inspect the 10x h5 layout before assuming an orientation."""
import h5py
import numpy as np

P = ("/mnt/storage/bzhang/AIVC/cellhippo/tenx/"
     "5k__3.0.0__Xenium_Prime_Human_Lung_Cancer_FFPE/"
     "Xenium_Prime_Human_Lung_Cancer_FFPE_cell_feature_matrix.h5")

with h5py.File(P, "r") as h:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:44s} shape {obj.shape} dtype {obj.dtype}")
        else:
            print(f"  {name:44s} GROUP {dict(obj.attrs)}")
    h.visititems(walk)
    print("\n  attrs root:", dict(h.attrs))
    m = h["matrix"]
    print("  matrix attrs:", dict(m.attrs))
    print("  shape:", m["shape"][:])
    print("  indptr len:", m["indptr"].shape, "data len:", m["data"].shape)
    print("  barcodes:", m["barcodes"].shape, m["barcodes"][:2])
    print("  features keys:", list(m["features"].keys()))
    print("  features/name:", m["features"]["name"].shape, m["features"]["name"][:2])
