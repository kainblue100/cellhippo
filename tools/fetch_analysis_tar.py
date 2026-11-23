"""Grab analysis.tar.gz -- the small Xenium Explorer subset.

It is 28-59 MB rather than the 8-22 GB outs.zip, and it contains
morphology_focus/, the segmentation stain image.  That image shares the
coordinate frame of cells.parquet exactly (experiment.xenium gives its pixel
size), so registering it to the H&E gives the cell->H&E transform without any
guesswork about what he_imagealignment.csv means.
"""
import os
import shutil
import tarfile
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

DATASETS = [
    ("5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    ("lc", "2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    ("luad", "2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
]

for tag, ver, name in DATASETS:
    d = os.path.join(OUT, f"{tag}__{ver}__{name}")
    os.makedirs(d, exist_ok=True)
    tar = os.path.join(d, f"{name}_analysis.tar.gz")
    if not (os.path.exists(tar) and os.path.getsize(tar) > 1000):
        url = f"{BASE}/{ver}/{name}/{name}_analysis.tar.gz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, \
                    open(tar + ".part", "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    f.write(c)
            os.replace(tar + ".part", tar)
        except Exception as e:
            print(f"{tag}: FAIL {type(e).__name__} {e}")
            continue
    print(f"\n=== {tag}: {os.path.getsize(tar)/1e6:.1f} MB ===")
    dst = os.path.join(d, "analysis")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with tarfile.open(tar) as t:
        names = t.getnames()
        for n in names[:15]:
            print("   ", n)
        if len(names) > 15:
            print(f"    ... {len(names)-15} more")
        t.extractall(dst)
    print(f"  extracted {len(names)} entries")
