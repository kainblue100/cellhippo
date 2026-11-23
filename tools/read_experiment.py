"""experiment.xenium is the authoritative description of the image registration.
Read it before guessing at the matrix again."""
import json
import os

BASE = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"

for d in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, d, "experiment.xenium")
    if not os.path.exists(p):
        # the file was saved as <name>_experiment.xenium by the downloader
        cand = [x for x in os.listdir(os.path.join(BASE, d))
                if x.endswith("experiment.xenium")]
        if not cand:
            continue
        p = os.path.join(BASE, d, cand[0])
    print(f"\n=== {d} ===")
    print(open(p, encoding="utf-8", errors="replace").read())
