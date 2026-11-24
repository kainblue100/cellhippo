"""Compare the strict run against the relaxed-filter run.

The user asked for a larger sample.  With one more donor still downloading, the
available move was to admit more windows from the same three slides (3,040 ->
3,430).  The pattern COUNT is stable at 4, but that is the weakest possible form
of stability: what matters is whether the same tissue ends up in the same
pattern, and whether the conclusions drawn from it survive.  This script puts the
two runs side by side so that question can be answered rather than asserted.

Writes work/sensitivity.json, which the page reads.
"""
from __future__ import annotations

import argparse
import json
import os

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
DOCS = "/mnt/storage/bzhang/AIVC/cellhippo/docs"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def comp(d):
    import numpy as np
    ph = np.array(d["tile_pheno"])
    sl = np.array(d["slides"])
    tags = list(d["tags"])
    G = int(ph.max()) + 1
    M = np.zeros((G, len(tags)))
    for j, t in enumerate(tags):
        sel = sl == t
        n = int(sel.sum())
        for k in range(G):
            M[k, j] = float((ph[sel] == k).sum()) / max(n, 1)
    return M.tolist(), tags, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", default=DOCS,
                    help="directory holding the published run's JSONs")
    ap.add_argument("--relaxed", default=".",
                    help="directory holding the relaxed run's JSONs")
    ap.add_argument("--out", default="sensitivity.json")
    a = ap.parse_args()

    pairs = {
        "strict": (os.path.join(a.strict, "fused.json"),
                   os.path.join(a.strict, "trajectory.json"),
                   os.path.join(a.strict, "transitions.json"),
                   os.path.join(a.strict, "cross.json")),
        "relaxed": (os.path.join(a.relaxed, "fused.json"),
                    os.path.join(a.relaxed, "trajectory.json"),
                    os.path.join(a.relaxed, "transitions_relaxed.json"),
                    os.path.join(a.relaxed, "cross_relaxed.json")),
    }

    out = {"strict": {}, "relaxed": {}}
    for key, (fp, tp, rp, cp) in pairs.items():
        f = load(fp)
        if not f:
            continue
        M, tags, G = comp(f)
        e = {"tiles": f["n_tiles"], "patterns": G, "samples": tags,
             "composition": M}
        t = load(tp)
        if t:
            e["order"] = {k: v["order"] for k, v in t["slides"].items()}
        tr = load(rp)
        if tr:
            e["handovers"] = {k: v["handovers"] for k, v in
                              tr["slides"].items()}
            # who is the hub, and what share does P3 carry
            hub = {}
            for k, v in tr["slides"].items():
                m = v["matrix"]
                n = len(m)
                tot = sum(sum(r) for r in m) / 2
                rows = [sum(m[i]) for i in range(n)]
                hub[k] = {"biggest_hub": int(max(range(n), key=lambda i: rows[i])),
                          "hub_share": max(rows) / tot if tot else 0,
                          "p3_share": (rows[3] / tot) if n > 3 and tot else 0}
            e["hub"] = hub
        c = load(cp)
        if c:
            e["sample_corr"] = c.get("sample_correlation")
            e["separation"] = c.get("pattern_distance_summary")
            e["modality_r"] = {k: v["sample_pair_r"]
                               for k, v in (c.get("stability") or {}).items()}
        out[key] = e

    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    print("=== strict vs relaxed ===")
    for k in ("strict", "relaxed"):
        e = out.get(k)
        if not e:
            print(f"  {k}: missing")
            continue
        print(f"\n  {k}:  {e['tiles']} tiles -> {e['patterns']} patterns")
        print("        " + "".join(f"{t:>9s}" for t in e["samples"]))
        for i, row in enumerate(e["composition"]):
            print(f"     P{i} " + "".join(f"{v*100:8.1f}%" for v in row))
        if "hub" in e:
            print("        hub: " + "  ".join(
                f"{s}: P{h['biggest_hub']} {h['hub_share']:.0%} "
                f"(P3 {h['p3_share']:.0%})" for s, h in e["hub"].items()))
        if "handovers" in e:
            print("        handovers: " + json.dumps(e["handovers"]))
        if "order" in e:
            print("        order: " + json.dumps(e["order"]))
        if "modality_r" in e:
            print("        modality r: " + json.dumps(
                {k2: round(v, 2) for k2, v in e["modality_r"].items()}))
        if "separation" in e:
            print("        separation: " + json.dumps(e["separation"]))

    print(f"\nwrote {WORK}/sensitivity.json")


if __name__ == "__main__":
    main()
