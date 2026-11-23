"""Pull the image-pipeline files from GHIST_plus (and retry HIPOS).

The point is to copy their tiling, normalisation and backbone conventions rather
than invent them.
"""
import base64
import json
import os
import subprocess

GH = r"D:\coda_lungs\tools\gh_extract\bin\gh.exe"
OUT = r"D:\coda_matlab\cellhippo\refs"

FILES = {
    "SydneyBioX/GHIST_plus": [
        "README.md",
        "dataio/dataset_input.py",
        "dataio/spatial.py",
        "model/backbone.py",
        "model/layers.py",
        "train.py",
        "configs/config_all_cancers.json",
        "configs/config_pancancer.json",
        "dataio/samplers.py",
        "utils/utils.py",
    ],
    "ZhoulabCPH/HIPOS": [
        "README.md",
    ],
}


def api(path, tries=3):
    last = ""
    for _ in range(tries):
        p = subprocess.run([GH, "api", path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode == 0:
            return json.loads(p.stdout)
        last = p.stderr[:200]
    raise RuntimeError(last)


os.makedirs(OUT, exist_ok=True)
for repo, files in FILES.items():
    tag = repo.split("/")[1]
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    print(f"\n=== {repo} ===")
    for f in files:
        try:
            r = api(f"repos/{repo}/contents/{f}")
            content = base64.b64decode(r["content"]).decode("utf-8", "replace")
            p = os.path.join(d, f.replace("/", "__"))
            open(p, "w", encoding="utf-8").write(content)
            print(f"  {len(content)/1024:8.1f} KB  {f}")
        except Exception as e:
            print(f"  FAILED {f}: {str(e)[:120]}")
