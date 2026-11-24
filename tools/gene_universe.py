"""What genes are comparable between the 17-sample cohort and the external 3?"""
import json
import os

import h5py

NEW = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw/Xenium_LUAD_No2/cell_feature_matrix.h5"
OLD = {
    "5k": "/mnt/storage/bzhang/AIVC/cellhippo/tenx/5k__3.0.0__Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_cell_feature_matrix.h5",
    "lc": "/mnt/storage/bzhang/AIVC/cellhippo/tenx/lc__2.0.0__Xenium_V1_Human_Lung_Cancer_Addon_FFPE/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_cell_feature_matrix.h5",
    "luad": "/mnt/storage/bzhang/AIVC/cellhippo/tenx/luad__2.0.0__Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_cell_feature_matrix.h5",
}


def genes(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        ft = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["feature_type"][:]]
        nm = [x.decode() if isinstance(x, bytes) else str(x)
              for x in m["features"]["name"][:]]
    return [g for g, t in zip(nm, ft) if t == "Gene Expression"]


new = genes(NEW)
print(f"new panel: {len(new)} genes")
print("  first 40:", ", ".join(new[:40]))

MARK = {
    "AT1": ["AGER", "PDPN", "CAV1", "HOPX", "CAV2"],
    "AT2": ["SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "ABCA3", "NKX2-1"],
    "epi": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "cil": ["FOXJ1", "TPPP3", "PCDHAC2", "CAPS", "RSPH1"],
    "endo": ["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7", "PLVAP"],
    "fib": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "nkt": ["CD3D", "CD3E", "CD2", "CD8A", "IL7R", "TRAC"],
    "b": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "plasma": ["MZB1", "JCHAIN", "DERL3", "SDC1"],
    "mph": ["CD68", "CD163", "MRC1", "C1QA", "CSF1R"],
    "mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
    "prolif": ["MKI67", "TOP2A", "UBE2C"],
}
print("\n=== marker availability in the new 302-gene panel ===")
S = set(new)
tot = 0
for k, v in MARK.items():
    have = [g for g in v if g in S]
    tot += len(have)
    print(f"  {k:7s} {len(have)}/{len(v)}  {have}")

print("\n=== overlap with the external 3 ===")
sets = {k: set(genes(p)) for k, p in OLD.items()}
for k, s in sets.items():
    inter = S & s
    print(f"  new ∩ {k:5s} = {len(inter):4d} genes "
          f"(of {len(s)} in {k})")
allthree = set.intersection(*sets.values())
print(f"  external 3 shared      = {len(allthree)} genes")
print(f"  new ∩ external-shared  = {len(S & allthree)} genes")

# how many of the markers survive the external intersection
print("\n=== markers present in new ∩ all 3 external ===")
common = S & allthree
for k, v in MARK.items():
    have = [g for g in v if g in common]
    print(f"  {k:7s} {have}")

json.dump({"new_panel": new,
           "external_shared": sorted(allthree),
           "new_x_external": sorted(common)},
          open("/mnt/storage/bzhang/AIVC/cellhippo/work/gene_universe.json", "w"))
print("\nwrote work/gene_universe.json")
