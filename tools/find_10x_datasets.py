"""Find the 10x dataset folder names for all three GHIST sets, and check that a
cell_feature_matrix.h5 AND a cells.parquet are both available.

GHIST sources:
  5k    .../xenium-human-lung-cancer-post-xenium-technote        (experiment 2, 5K panel)
  lc    .../ffpe-human-lung-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard
  luad  .../preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard
"""
import json
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"

CANDIDATES = [
    # already confirmed
    ("3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE", "5k"),
    ("3.0.0", "Xenium_V1_Human_Lung_Cancer_FFPE", "exp1"),
    # lc: FFPE lung cancer with human immuno-oncology panel + custom add-on
    ("3.0.0", "Xenium_V1_FFPE_Human_Lung_Cancer_IO", "lc?"),
    ("3.0.0", "Xenium_V1_Human_Lung_Cancer_IO_FFPE", "lc?"),
    ("3.0.0", "Xenium_V1_FFPE_Human_Lung_Cancer_IO_Addon", "lc?"),
    ("3.0.0", "Xenium_V1_Human_Lung_Cancer_FFPE_IO", "lc?"),
    ("2.0.0", "Xenium_V1_FFPE_Human_Lung_Cancer_IO", "lc?"),
    # luad: preview, multimodal cell segmentation
    ("3.0.0", "Xenium_V1_FFPE_Human_Lung_Cancer_preview", "luad?"),
    ("3.0.0", "Xenium_V1_Human_Lung_Cancer_preview_FFPE", "luad?"),
    ("2.0.0", "Xenium_V1_FFPE_Human_Lung_Cancer_preview", "luad?"),
    ("3.0.0", "Xenium_V1_humanLung_Cancer_preview_FFPE", "luad?"),
    ("2.0.0", "Xenium_V1_humanLung_Cancer_preview_FFPE", "luad?"),
    # generic preview naming seen in older releases
    ("2.0.0", "Xenium_V1_humanLung_Cancer_FFPE", "luad?"),
]
SUFFIXES = ["cell_feature_matrix.h5", "cells.parquet", "experiment.xenium"]


def head(url, timeout=25):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, int(r.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception as e:
        return type(e).__name__, 0


rows = []
for ver, name, tag in CANDIDATES:
    base = f"{BASE}/{ver}/{name}/{name}"
    hits = {}
    for suf in SUFFIXES:
        code, size = head(f"{base}_{suf}")
        if code == 200:
            hits[suf] = size
    if hits:
        has_both = ("cell_feature_matrix.h5" in hits and "cells.parquet" in hits)
        print(f"  OK   {ver} {name:48s} [{tag}]  "
              f"{ {k: f'{v/1e6:.1f}MB' for k, v in hits.items()} }  "
              f"both={has_both}")
        rows.append({"ver": ver, "name": name, "tag": tag, "files": hits,
                     "has_matrix_and_coords": has_both})

json.dump(rows, open("/mnt/storage/bzhang/AIVC/cellhippo/tenx_found.json", "w"),
          indent=2)
print(f"\n  datasets with matrix + coordinates: "
      f"{sum(1 for r in rows if r['has_matrix_and_coords'])}")
