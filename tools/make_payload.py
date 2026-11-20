"""Distil the cellhippo JSON outputs into one compact payload for the webpage.

Also computes the two things the write-up needs but the tool does not emit:
  - how many cells sit in each phenotype per dataset
  - for the split annotations, which phenotypes they were cut into and how big
    each piece is (this is the evidence that a phenotype is a refinement, not a
    relabelling)
"""
import json
import os

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
OUT = "/mnt/storage/bzhang/AIVC/cellhippo/work/web_payload.json"
TAGS = ["5k", "lc", "luad", "pooled"]


def main():
    payload = {"datasets": {}}
    for t in TAGS:
        p = os.path.join(WORK, f"{t}_cellhippo.json")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = json.load(open(p))
        pc = d["phenotype_composition"]
        cells_by_ph = {r["phenotype"]: r["cells"] for r in pc}
        splits = [r for r in d["annotation_concordance"]["per_annotation"]
                  if r["split"]]
        kept = [r for r in d["annotation_concordance"]["per_annotation"]
                if not r["split"]]
        payload["datasets"][t] = {
            "cells": d["cells"],
            "state_shape": d["state_shape"],
            "fine_clusters": d["fine_clusters"],
            "n_phenotypes": d["n_phenotypes"],
            "bic_curve": d["bic_curve"],
            "ari": round(d["annotation_concordance"]["ari"], 4),
            "nmi": round(d["annotation_concordance"]["nmi"], 4),
            "n_annotations": len(d["annotation_concordance"]["per_annotation"]),
            "n_split": len(splits),
            "cells_by_phenotype": cells_by_ph,
            "phenotype_composition": pc,
            "split_examples": [
                {"annotation": r["annotation"], "cells": r["cells"],
                 "n_phenotypes": r["n_phenotypes"],
                 "dominant": r["dominant_phenotype"],
                 "dominant_fraction": round(r["dominant_fraction"], 4)}
                for r in splits[:8]],
            "kept_examples": [
                {"annotation": r["annotation"], "cells": r["cells"],
                 "dominant_fraction": round(r["dominant_fraction"], 4)}
                for r in kept[:8]],
        }
        if "slides" in d:
            payload["datasets"][t]["slides"] = d["slides"]
            payload["datasets"][t]["slide_profiles"] = d["slide_profiles"]
            payload["datasets"][t]["subtyping"] = d["subtyping"]

    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB)")

    print("\n=== headline table ===")
    print(f"  {'set':7s}{'cells':>10s}{'clusters':>10s}{'pheno':>7s}"
          f"{'ARI':>8s}{'NMI':>8s}{'split':>9s}")
    for t, v in payload["datasets"].items():
        print(f"  {t:7s}{v['cells']:10,}{v['fine_clusters']:10d}"
              f"{v['n_phenotypes']:7d}{v['ari']:8.3f}{v['nmi']:8.3f}"
              f"{v['n_split']:4d}/{v['n_annotations']:<4d}")

    print("\n=== the control contrast ===")
    a = payload["datasets"]
    print(f"  label-only   ARI {a['pooled']['ari']:.3f}  "
          f"splits {a['pooled']['n_split']}/{a['pooled']['n_annotations']}")
    for t in ("5k", "lc", "luad"):
        print(f"  {t:5s}+expr   ARI {a[t]['ari']:.3f}  "
              f"splits {a[t]['n_split']}/{a[t]['n_annotations']}")


if __name__ == "__main__":
    main()
