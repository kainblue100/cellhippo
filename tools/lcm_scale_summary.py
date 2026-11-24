"""Compare the pattern chain across window sizes.

The point of the exercise: a phenotype that only exists at one window size is a
property of the window, not of the tissue.  Three scales spanning 4x:

    112 px =  95 um     224 px = 190 um     448 px = 381 um

Reports, per scale: how many windows, how many phenotypes, how far apart the
phenotype cell-type signatures are, and whether the same handover pairs recur.
"""
import glob
import json
import os

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
SCALES = [("112", 95), ("224", 190), ("448", 381)]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def analyse(name):
    res = load(os.path.join(WORK, f"{name}.json"))
    if not res:
        return None
    d = {"tiles": res["n_tiles"], "patterns": int(np.array(res["tile_pheno"]).max()) + 1}
    an = load(os.path.join(WORK, f"{name}_analysis.json"))
    if an:
        d["separation"] = an["separation"]
        d["signature"] = an["signature"]
        d["handovers"] = an["handovers"]
        d["recurrence"] = an["handover_recurrence"]
        d["composition"] = an["composition"]
        d["samples"] = an["samples"]
    return d


def main():
    print(f"{'scale':>10s} {'px':>5s} {'um':>5s} {'tiles':>7s} {'patterns':>9s} "
          f"{'cos-min':>8s} {'cos-max':>8s}")
    rows = {}
    for tag, um in SCALES:
        name = "lcm_fused" if tag == "224" else f"lcm_fused_{tag}"
        d = analyse(name)
        if not d:
            print(f"{tag:>10s} {tag:>5s} {um:>5d}   (missing)")
            continue
        rows[tag] = d
        sep = d.get("separation") or {}
        print(f"{tag:>10s} {tag:>5s} {um:>5d} {d['tiles']:7d} "
              f"{d['patterns']:9d} {sep.get('min', float('nan')):8.3f} "
              f"{sep.get('max', float('nan')):8.3f}")

    print("\n=== does a phenotype survive a change of window size? ===")
    print("  (a phenotype exists at a scale if any sample assigns >2% of its "
          "tiles to it)")
    for tag, d in rows.items():
        comp = np.array(d.get("composition") or [])
        if not comp.size:
            continue
        strong = [k for k in range(comp.shape[0]) if comp[k].max() > 0.02]
        print(f"  {tag:>4s}  {d['patterns']} phenotypes, "
              f"{len(strong)} of them reach >2% in some sample: "
              f"{['P'+str(k) for k in strong]}")

    print("\n=== handover pairs, pooled over samples ===")
    for tag, d in rows.items():
        rec = d.get("recurrence") or {}
        if not rec:
            continue
        ns = len(d.get("samples") or [])
        pairs = sorted(rec.items(), key=lambda kv: -kv[1]["samples"])[:4]
        print(f"  {tag:>4s}  " + "  ".join(
            f"{k} in {v['samples']}/{ns} (n={sum(v['per_sample'].values())})"
            for k, v in pairs))

    print("\n=== signature of the largest phenotype at each scale ===")
    for tag, d in rows.items():
        sig = d.get("signature") or {}
        comp = np.array(d.get("composition") or [])
        if not comp.size:
            continue
        k = int(comp.mean(axis=1).argmax())
        top = list(sig.get(str(k), {}).items())[:4]
        print(f"  {tag:>4s}  P{k}: " +
              ", ".join(f"{a} {v*100:.0f}%" for a, v in top))

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "handovers"}
               for k, v in rows.items()},
              open(os.path.join(WORK, "lcm_scale.json"), "w"), indent=2)
    print("\nwrote work/lcm_scale.json")


if __name__ == "__main__":
    main()
