"""Get the small registration files, read the affine transform, and check which
tiff reader the environment has.

Cell centroids in cells.parquet are in micron slide coordinates; the H&E image is
in pixels.  he_imagealignment.csv is what bridges them, so it has to be read
before any tile can be cut.
"""
import json
import os
import subprocess
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

DATASETS = [
    ("5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    ("lc", "2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    ("luad", "2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
]
SMALL = ["he_imagealignment.csv", "cell_boundaries.parquet",
         "nucleus_boundaries.parquet"]


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest + ".p", "wb") as f:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                f.write(c)
        os.replace(dest + ".p", dest)
        return True
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        return False


for tag, ver, name in DATASETS:
    d = os.path.join(OUT, f"{tag}__{ver}__{name}")
    os.makedirs(d, exist_ok=True)
    print(f"\n=== {tag} ===")
    for suf in SMALL:
        ok = fetch(f"{BASE}/{ver}/{name}/{name}_{suf}",
                   os.path.join(d, f"{name}_{suf}"))
        if ok:
            print(f"  ok {suf} ({os.path.getsize(os.path.join(d, f'{name}_{suf}'))/1e6:.2f} MB)")

print("\n=== alignment contents (5k) ===")
p = os.path.join(OUT, "5k__3.0.0__Xenium_Prime_Human_Lung_Cancer_FFPE",
                 "Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv")
if os.path.exists(p):
    txt = open(p, encoding="utf-8", errors="replace").read()
    print(txt[:1200])

print("\n=== image libraries available ===")
for mod in ("tifffile", "pyvips", "openslide", "PIL", "zarr", "imageio",
            "skimage", "cv2"):
    try:
        m = subprocess.run(
            ["/mnt/storage/bzhang/AIVC/vcellx/.venv/bin/python", "-c",
             f"import {mod}; print(getattr({mod},'__version__','?'))"],
            capture_output=True, text=True, timeout=60)
        v = m.stdout.strip() if m.returncode == 0 else "MISSING"
    except Exception:
        v = "ERR"
    print(f"  {mod:12s} {v}")

print("\n=== cells.parquet columns ===")
import pyarrow.parquet as pq
pqf = os.path.join(OUT, "5k__3.0.0__Xenium_Prime_Human_Lung_Cancer_FFPE",
                   "Xenium_Prime_Human_Lung_Cancer_FFPE_cells.parquet")
print(pq.read_schema(pqf))
