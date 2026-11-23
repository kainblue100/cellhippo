"""What image files does the 10x CDN expose for these three datasets?

cellhippo so far used expression + annotation only.  To reproduce HIPPO's figure
style we need the H&E image itself -- morphology tiles are the whole point of the
original method, and the spatial maps need coordinates registered to an image.

Candidates, in the order 10x documentation names them:
  he_image.ome.tif            post-Xenium H&E (what Xenium Explorer shows)
  he_imagealignment.csv       affine from image pixels to slide coordinates
  morphology_focus_000N.ome.tif   per-cycle morphology stack
  morphology_mip.ome.tif      max-intensity projection
  outs.zip                    everything (large)
"""
import json
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"

DATASETS = [
    ("5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    ("lc", "2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    ("luad", "2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
]

SUFFIXES = [
    "he_image.ome.tif",
    "he_imagealignment.csv",
    "morphology_mip.ome.tif",
    "morphology_focus.ome.tif",
    "morphology_focus_0000.ome.tif",
    "morphology_focus_0001.ome.tif",
    "morphology_focus_0002.ome.tif",
    "morphology_focus_0003.ome.tif",
    "transcripts.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "analysis.tar.gz",
    "outs.zip",
]


def head(url, timeout=25):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, int(r.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception:
        return 0, 0


found = {}
for tag, ver, name in DATASETS:
    base = f"{BASE}/{ver}/{name}/{name}"
    print(f"\n=== {tag}  {ver}/{name} ===")
    hits = {}
    for suf in SUFFIXES:
        code, size = head(f"{base}_{suf}")
        if code == 200:
            hits[suf] = size
            print(f"  OK   {suf:34s} {size/1e6:9.2f} MB")

    # morphology_focus often lives in a subdirectory rather than as a suffix
    for i in range(6):
        u = (f"{BASE}/{ver}/{name}/morphology_focus/"
             f"{name}_morphology_focus_{i:04d}.ome.tif")
        c, s = head(u)
        if c == 200:
            hits[f"morphology_focus/{i:04d}"] = s
            print(f"  OK   morphology_focus/{i:04d}          {s/1e6:9.2f} MB")

    if not hits:
        print("  (nothing found)")
    found[tag] = hits

json.dump(found, open("/mnt/storage/bzhang/AIVC/cellhippo/tenx_images.json", "w"),
          indent=2)
print("\n=== summary ===")
for tag, h in found.items():
    has_he = any("he_image" in k for k in h)
    print(f"  {tag:5s} {len(h)} file(s)  H&E present: {has_he}")
