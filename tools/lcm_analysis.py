"""Everything downstream of the clustering, for the 17-sample cohort.

Written as one script rather than reusing trajectory.py / transitions.py /
cross_sample.py, because three assumptions in those are wrong for this cohort:

  NORMAL_LABELS   trajectory.py hardcodes ("normal_epi", "AT2_like", "AT1", "cil").
                  This cohort's derived classes are AT1/AT2/epi/cil/fib/... so the
                  root score would sum over one label and pick arbitrarily.

  images          there is no registered H&E.  transitions.py and migration.py
                  both draw on one.  Here the spatial maps are drawn on the
                  morphology MIP, which needs no registration.

  scale           tiles are 190 um, not 243/112/104, so any statement about
                  spatial autocorrelation has to be recomputed rather than
                  carried over.

Produces: figures/lcm/*.jpg|png and work/lcm_analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import tifffile
from PIL import Image, ImageDraw

WORK = "/mnt/storage/bzhang/AIVC/cellhippo/work"
FIGS = "/mnt/storage/bzhang/AIVC/cellhippo/figures"
RAW = "/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw"
OUT = os.path.join(FIGS, "lcm")
TILEDIR = os.path.join(WORK, "lcm_tiles")

PALETTE = ["#c2410c", "#1d4ed8", "#0f766e", "#7c3aed", "#db2777", "#a16207",
           "#0369a1", "#15803d"]

# Classes that count as differentiated airway/alveolar epithelium when choosing a
# root.  This is the cohort's own vocabulary, not the three-slide cohort's.
NORMAL_LABELS = ("AT1", "AT2", "epi", "cil")


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load(name):
    return json.load(open(os.path.join(WORK, f"{name}.json"), encoding="utf-8"))


def per_sample_pheno(res):
    ph = np.array(res["tile_pheno"])
    sl = np.array(res["slides"])
    out, off = {}, 0
    for t in res["tags"]:
        n = int((sl == t).sum())
        out[t] = ph[off:off + n]
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lcm_fused",
                    help="clustering result in work/{name}.json")
    ap.add_argument("--tiledir", default=TILEDIR)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--figures", action="store_true", default=True)
    ap.add_argument("--no-figures", dest="figures", action="store_false")
    a = ap.parse_args()
    name, tiledir, outdir = a.name, a.tiledir, a.outdir
    os.makedirs(outdir, exist_ok=True)
    res = load(name)
    phs = per_sample_pheno(res)
    tags = list(res["tags"])
    G = int(np.array(res["tile_pheno"]).max()) + 1
    print(f"{G} phenotypes over {res['n_tiles']} tiles, {len(tags)} samples "
          f"({os.path.basename(tiledir)})")

    # ---------------------------------------------------------------- composition
    M = np.zeros((G, len(tags)))
    for j, t in enumerate(tags):
        n = len(phs[t])
        for k in range(G):
            M[k, j] = (phs[t] == k).sum() / max(n, 1)
    print("\n=== composition (fraction of each sample's tiles) ===")
    print("        " + "".join(f"{t[-6:]:>8s}" for t in tags))
    for k in range(G):
        print(f"   P{k}  " + "".join(f"{v*100:7.1f}%" for v in M[k]))

    # ---------------------------------------------------------------- signatures
    anns = None
    sig = np.zeros((G, 12))
    cnt = np.zeros(G)
    for t in tags:
        z = np.load(os.path.join(tiledir, f"{t}_tiles.npz"))
        if anns is None:
            anns = [str(a) for a in z["annotations"]]
        A = np.zeros((len(z["ann"]), len(anns)))
        for j, lab in enumerate(z["annotations"]):
            A[:, anns.index(str(lab))] = z["ann"][:, j]
        for k in range(G):
            sel = phs[t] == k
            if sel.sum():
                sig[k] += A[sel].sum(axis=0)
                cnt[k] += sel.sum()
    sig = sig / np.maximum(cnt, 1)[:, None]
    print("\n=== cell-type signature per phenotype (derived annotation) ===")
    for k in range(G):
        top = np.argsort(-sig[k])[:5]
        print(f"   P{k}  " + ", ".join(f"{anns[i]} {sig[k,i]*100:.0f}%"
                                       for i in top if sig[k, i] > 0.001))

    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(sig, metric="cosine"))
    print(f"\n  separation (cosine): min {D[D>0].min():.3f}  max {D.max():.3f}")
    for k in range(G):
        j = int(np.argmin(np.where(np.arange(G) == k, np.inf, D[k])))
        print(f"   P{k} closest to P{j} at {D[k, j]:.3f}")

    # ---------------------------------------------------------------- adjacency
    print("\n=== adjacent-tile handovers (within a sample) ===")
    trans = {}
    for t in tags:
        z = np.load(os.path.join(tiledir, f"{t}_tiles.npz"))
        grid = {(int(c), int(r)): int(p)
                for (c, r, _n, _u), p in zip(z["meta"], phs[t])}
        pairs = Counter()
        for (ix, iy), p in grid.items():
            for dx, dy in ((1, 0), (0, 1)):
                q = grid.get((ix + dx, iy + dy))
                if q is None or q == p:
                    continue
                pairs[(min(p, q), max(p, q))] += 1
        trans[t] = pairs
        tot = sum(pairs.values())
        top = pairs.most_common(3)
        print(f"   {t:22s} {len(grid):5d} tiles, {tot:5d} handovers  "
              + ", ".join(f"P{a}-P{b} {n}" for (a, b), n in top))

    # pooled handover matrix: which pairs recur across samples
    rec = Counter()
    per_pair = {}
    for t, pairs in trans.items():
        for key, n in pairs.items():
            rec[key] += 1
            per_pair.setdefault(key, {})[t] = n
    print("\n  pairs appearing in the most samples:")
    for key, nsamples in rec.most_common(6):
        print(f"   P{key[0]}-P{key[1]}  in {nsamples}/{len(tags)} samples, "
              f"total {sum(per_pair[key].values())}")

    # ---------------------------------------------------------------- figures
    def gallery():
        cell = 96
        per = 6
        W = per * cell
        H = len(tags) * cell
        sheet = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(sheet)
        for i, t in enumerate(tags):
            z = np.load(os.path.join(tiledir, f"{t}_tiles.npz"))
            em = np.load(os.path.join(WORK, f"{t}_cnn.npz"))["emb"]
            em = em / (np.linalg.norm(em, axis=1, keepdims=True) + 1e-9)
            ph = phs[t]
            k = int(np.bincount(ph).argmax())
            sel = np.where(ph == k)[0]
            c = em[sel].mean(axis=0)
            c /= np.linalg.norm(c) + 1e-9
            order = sel[np.argsort(-(em[sel] @ c))][:per]
            for j, idx in enumerate(order):
                im = Image.fromarray(z["img"][idx])
                sheet.paste(im.resize((cell, cell), Image.LANCZOS),
                            (j * cell, i * cell))
            dr.text((2, i * cell + 2), f"P{k}", fill=(255, 255, 0))
        p = os.path.join(outdir, "lcm_gallery.jpg")
        sheet.save(p, quality=92)
        print(f"\n  wrote {p} ({W}x{H})")

    def spatial():
        cell = 3
        cols = 5
        rows = (len(tags) + cols - 1) // cols
        pw, ph_ = 300, 300
        sheet = Image.new("RGB", (cols * pw, rows * ph_), "white")
        dr = ImageDraw.Draw(sheet)
        for i, t in enumerate(tags):
            z = np.load(os.path.join(tiledir, f"{t}_tiles.npz"))
            m = z["meta"]
            W = (m[:, 0].max() + 2) * cell
            Hh = (m[:, 1].max() + 2) * cell
            im = Image.new("RGB", (W, Hh), "white")
            d2 = ImageDraw.Draw(im)
            for (c, r, _n, _u), p in zip(m, phs[t]):
                x0, y0 = int(c) * cell, int(r) * cell
                d2.rectangle([x0, y0, x0 + cell, y0 + cell],
                             fill=rgb(PALETTE[int(p) % len(PALETTE)]))
            im.thumbnail((pw - 8, ph_ - 18), Image.LANCZOS)
            r_, c_ = divmod(i, cols)
            sheet.paste(im, (c_ * pw + 4, r_ * ph_ + 14))
            dr.text((c_ * pw + 4, r_ * ph_ + 2), t, fill=(20, 20, 20))
        p = os.path.join(outdir, "lcm_spatial.jpg")
        sheet.save(p, quality=92)
        print(f"  wrote {p}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def composition():
        fig, ax = plt.subplots(figsize=(13, 4.2), facecolor="white")
        im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=90, fontsize=8)
        ax.set_yticks(range(G))
        ax.set_yticklabels([f"P{k}" for k in range(G)])
        for k in range(G):
            for j in range(len(tags)):
                v = M[k, j]
                if v > 0.01:
                    ax.text(j, k, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "#1b1f24")
        ax.set_title("17-sample cohort: phenotype composition per sample", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()
        p = os.path.join(outdir, "lcm_composition.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {p}")

    def signature():
        fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="white")
        im = ax.imshow(sig, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(anns)))
        ax.set_xticklabels(anns, rotation=60, fontsize=8)
        ax.set_yticks(range(G))
        ax.set_yticklabels([f"P{k}" for k in range(G)])
        for k in range(G):
            for j in range(len(anns)):
                if sig[k, j] > 0.01:
                    ax.text(j, k, f"{sig[k,j]*100:.0f}", ha="center",
                            va="center", fontsize=6)
        ax.set_title("phenotype cell-type signature", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()
        p = os.path.join(outdir, "lcm_signature.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {p}")

    if a.figures:
        gallery()
        spatial()
        composition()
        signature()
    else:
        print("  figures skipped")

    json.dump({
        "cohort": "takano_kero 17 samples",
        "n_samples": len(tags), "n_tiles": int(res["n_tiles"]),
        "patterns": G,
        "samples": tags,
        "composition": [[round(float(v), 4) for v in r] for r in M],
        "signature": {str(k): {anns[i]: round(float(sig[k, i]), 4)
                               for i in np.argsort(-sig[k])[:6]}
                      for k in range(G)},
        "separation": {"min": round(float(D[D > 0].min()), 4),
                       "max": round(float(D.max()), 4)},
        "handovers": {t: {f"{a}-{b}": n for (a, b), n in p.most_common()}
                      for t, p in trans.items()},
        "handover_recurrence": {f"{k[0]}-{k[1]}": {"samples": n,
                                                   "per_sample": per_pair[k]}
                                for k, n in rec.items()},
        "annotation_source": "derived by marker scoring on the 302-gene hLung_100g "
                             "panel; not a pathologist label",
    }, open(os.path.join(WORK, f"{name}_analysis.json"), "w"), indent=2)
    print(f"\nwrote work/{name}_analysis.json")


if __name__ == "__main__":
    main()
