"""Download matrix + coordinates for all three GHIST source datasets.

Mapping established by cell counts and confirmed by barcode overlap:
  5k   3.0.0 Xenium_Prime_Human_Lung_Cancer_FFPE     278,328 cells
  lc   2.0.0 Xenium_V1_Human_Lung_Cancer_Addon_FFPE  161,000 cells
  luad 2.0.0 Xenium_V1_humanLung_Cancer_FFPE         162,254 cells
"""
import json
import os
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

DATASETS = [
    ("5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    ("lc", "2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    ("luad", "2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
]
FILES = ["cell_feature_matrix.h5", "cells.parquet", "experiment.xenium",
         "gene_panel.json"]


def fetch(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"    have {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return True
    tmp = dest + ".part"
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                n = 0
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    f.write(c)
                    n += len(c)
            os.replace(tmp, dest)
            print(f"    got  {os.path.basename(dest)} ({n/1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"    try {t+1}/{tries} failed: {type(e).__name__}")
            if os.path.exists(tmp):
                os.remove(tmp)
    return False


manifest_path = os.path.join(OUT, "manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

for tag, ver, name in DATASETS:
    d = os.path.join(OUT, f"{tag}__{ver}__{name}")
    os.makedirs(d, exist_ok=True)
    print(f"\n[{tag}] {ver}/{name}")
    got = {}
    for suf in FILES:
        url = f"{BASE}/{ver}/{name}/{name}_{suf}"
        if fetch(url, os.path.join(d, f"{name}_{suf}")):
            got[suf] = os.path.getsize(os.path.join(d, f"{name}_{suf}"))
    manifest[tag] = {"ver": ver, "name": name, "files": got}
    json.dump(manifest, open(manifest_path, "w"), indent=2)

print("\n=== summary ===")
for tag, info in manifest.items():
    print(f"  {tag:5s} {info['ver']} {info['name']:44s} "
          f"{ {k: f'{v/1e6:.1f}MB' for k, v in info['files'].items()} }")
