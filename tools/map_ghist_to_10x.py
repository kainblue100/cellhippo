"""Which 10x dataset is which GHIST set? Barcodes are the proof.

Same underlying run => the cell_id sets match essentially 1:1. Different runs
share nothing.
"""
import csv
import os

import h5py

GH = ("/mnt/storage/bzhang/AIVC/ghist/"
      "D10ny3u3-GHIST_input-4bf5fa04284b8c794887ef69456cf9588cb9290b")
TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

SETS = {
    "5k": "data_input/5k/xenium_5kt_by_annotation.csv",
    "lc": "data_input/lc/xenium_lc_by_annotation.csv",
    "luad": "data_input/luad/01_seurat_process/xenium_luad_by_annotation.csv",
}

gh = {}
for name, rel in SETS.items():
    with open(os.path.join(GH, rel), newline="") as f:
        r = csv.reader(f)
        next(r)
        ids = [row[0] for row in r if row]
    gh[name] = set(ids)
    print(f"  GHIST {name:5s} {len(ids):8,} cells")

print("\n=== overlap matrix (fraction of GHIST ids found in the 10x run) ===")
hdr = f"  {'10x dataset':54s}" + "".join(f"{k:>10s}" for k in SETS)
print(hdr)
for d in sorted(os.listdir(TENX)):
    p = os.path.join(TENX, d)
    if not os.path.isdir(p):
        continue
    h5s = [x for x in os.listdir(p) if x.endswith("cell_feature_matrix.h5")]
    if not h5s:
        print(f"  {d:54s}" + "".join(f"{'-':>10s}" for _ in SETS))
        continue
    with h5py.File(os.path.join(p, h5s[0]), "r") as h:
        bc = set(x.decode() if isinstance(x, bytes) else str(x)
                 for x in h["matrix"]["barcodes"][:])
    cells = "".join(f"{len(bc & gh[k])/len(gh[k])*100:9.2f}%" for k in SETS)
    print(f"  {d:54s}{cells}   (n={len(bc):,})")
