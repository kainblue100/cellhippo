"""Gene-panel sizes and overlaps for the three original slides.

The pipeline has been pooling these on independently computed 32-PC blocks, which
sidesteps the question of a common gene axis.  This quantifies how much of one
there actually is.
"""
import itertools
import json

import h5py

TENX = "/mnt/storage/bzhang/AIVC/cellhippo/tenx"
OLD = {
    "5k": f"{TENX}/5k__3.0.0__Xenium_Prime_Human_Lung_Cancer_FFPE/"
          "Xenium_Prime_Human_Lung_Cancer_FFPE_cell_feature_matrix.h5",
    "lc": f"{TENX}/lc__2.0.0__Xenium_V1_Human_Lung_Cancer_Addon_FFPE/"
          "Xenium_V1_Human_Lung_Cancer_Addon_FFPE_cell_feature_matrix.h5",
    "luad": f"{TENX}/luad__2.0.0__Xenium_V1_humanLung_Cancer_FFPE/"
            "Xenium_V1_humanLung_Cancer_FFPE_cell_feature_matrix.h5",
    "v4": f"{TENX}/v4__3.0.0__Xenium_V1_Human_Lung_Cancer_FFPE/"
          "Xenium_V1_Human_Lung_Cancer_FFPE_cell_feature_matrix.h5",
}
NEW = ("/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw/"
       "Xenium_LUAD_No2/cell_feature_matrix.h5")


def genes(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        ft = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["feature_type"][:]]
        nm = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["name"][:]]
        shape = tuple(m["shape"][:])
    return set(g for g, t in zip(nm, ft) if t == "Gene Expression"), shape[1]


G = {}
print(f"{'sample':8s} {'genes':>7s} {'cells':>9s}")
for k, p in OLD.items():
    s, n = genes(p)
    G[k] = s
    print(f"  {k:6s} {len(s):7,d} {n:9,d}")

new, _ = genes(NEW)
print(f"  {'new×17':6s} {len(new):7,d}   (all 17 share this panel)")

print("\n=== pairwise overlap among the three originals ===")
for a, b in itertools.combinations(("5k", "lc", "luad"), 2):
    print(f"  {a:4s} ∩ {b:4s} = {len(G[a] & G[b]):5,d}")
three = G["5k"] & G["lc"] & G["luad"]
print(f"  5k ∩ lc ∩ luad = {len(three):5,d}")

print("\n=== the 302-gene cohort against each original ===")
for k in ("5k", "lc", "luad"):
    print(f"  new ∩ {k:4s} = {len(new & G[k]):5,d}   "
          f"({len(new & G[k])/len(new)*100:.0f}% of the new panel)")

print("\n=== what the existing pipeline actually used ===")
print("  cluster_tiles.py builds 32 PCs *per slide, independently*, so the")
print("  expression block never needed a common gene axis.  The page already")
print("  criticises this in the Annotile table: PCs from separate slides are")
print("  not on a common axis, so the expression block is not comparable")
print("  across slides.  The 133 shared genes above are the axis it would need.")

json.dump({"sizes": {k: len(v) for k, v in G.items()},
           "new_panel": len(new),
           "three_way": sorted(three),
           "new_x_three": sorted(new & three)},
          open("/mnt/storage/bzhang/AIVC/cellhippo/work/gene_overlap.json", "w"),
          indent=2)
print("\nwrote work/gene_overlap.json")
