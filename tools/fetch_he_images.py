"""Download the post-Xenium H&E images (OME-TIFF).

5k 2.9 GB, lc 1.5 GB, luad 1.2 GB.  Resumable: a partial file is kept as .part
and re-requested from 0, because the 10x CDN does not honour Range requests on
these objects (same behaviour seen on the GDC endpoint earlier).
"""
import json
import os
import sys
import time
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BASE = "https://cf.10xgenomics.com/samples/xenium"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

DATASETS = [
    ("5k", "3.0.0", "Xenium_Prime_Human_Lung_Cancer_FFPE"),
    ("lc", "2.0.0", "Xenium_V1_Human_Lung_Cancer_Addon_FFPE"),
    ("luad", "2.0.0", "Xenium_V1_humanLung_Cancer_FFPE"),
]


def fetch(url, dest, expect):
    if os.path.exists(dest) and os.path.getsize(dest) == expect:
        print(f"  have {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)",
              flush=True)
        return True
    tmp = dest + ".part"
    for attempt in range(6):
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                t0 = time.time()
                while True:
                    c = r.read(1 << 22)
                    if not c:
                        break
                    f.write(c)
                    got += len(c)
                    if got % (1 << 28) < (1 << 22):
                        mb = got / 1e6
                        el = time.time() - t0
                        print(f"    {mb:8.1f} / {expect/1e6:.1f} MB  "
                              f"({mb/el:.1f} MB/s)", flush=True)
            if os.path.getsize(tmp) == expect:
                os.replace(tmp, dest)
                print(f"  done {os.path.basename(dest)}", flush=True)
                return True
            print(f"  short read: {os.path.getsize(tmp)} != {expect}", flush=True)
        except Exception as e:
            print(f"  attempt {attempt+1}/6 failed: {type(e).__name__} "
                  f"(had {got/1e6:.1f} MB)", flush=True)
            time.sleep(5)
    return False


ok = {}
for tag, ver, name in DATASETS:
    d = os.path.join(OUT, f"{tag}__{ver}__{name}")
    os.makedirs(d, exist_ok=True)
    url = f"{BASE}/{ver}/{name}/{name}_he_image.ome.tif"
    dest = os.path.join(d, f"{name}_he_image.ome.tif")
    print(f"\n=== {tag}: {name}_he_image.ome.tif ===", flush=True)
    # size from the earlier HEAD probe
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        expect = int(r.headers.get("Content-Length"))
    print(f"  expecting {expect/1e6:.1f} MB", flush=True)
    ok[tag] = fetch(url, dest, expect)

json.dump(ok, open("/mnt/storage/bzhang/AIVC/cellhippo/he_download.json", "w"),
          indent=2)
print("\n=== result ===")
for k, v in ok.items():
    print(f"  {k:5s} {'OK' if v else 'FAILED'}")
