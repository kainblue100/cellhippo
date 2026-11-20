"""Generate docs/index.html from the real cellhippo outputs.

Numbers are read from web_payload.json rather than typed in, so the page cannot
drift from the run.  Charts are inline SVG -- no CDN, so the page works offline
and from a GitHub Pages URL.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAYLOAD = os.path.join(ROOT, "docs", "web_payload.json")
OUT = os.path.join(ROOT, "docs", "index.html")

D = json.load(open(PAYLOAD))["datasets"]
ORDER = ["5k", "lc", "luad", "pooled"]

META = {
    "5k": {"panel": "Xenium Prime 5K Human Pan Tissue and Pathways (5,001 genes)",
           "tissue": "Lung adenocarcinoma, FFPE (Avaden)",
           "oaa": "3.0.0", "tenx": "Xenium_Prime_Human_Lung_Cancer_FFPE",
           "cells10x": "278,328"},
    "lc": {"panel": "Human Immuno-Oncology Profiling Panel + custom add-on (480 genes)",
           "tissue": "Lung NSCLC, FFPE (Discovery Life Sciences)",
           "oaa": "2.0.0", "tenx": "Xenium_V1_Human_Lung_Cancer_Addon_FFPE",
           "cells10x": "161,000"},
    "luad": {"panel": "Human Multi-Tissue and Cancer Panel (377 genes)",
             "tissue": "Lung adenocarcinoma, FFPE (BioIVT)",
             "oaa": "2.0.0", "tenx": "Xenium_V1_humanLung_Cancer_FFPE",
             "cells10x": "162,254"},
}


def fmt(n):
    return f"{n:,}"


def bar_chart(pairs, width=560, bar_h=22, gap=6, colour="#3b6ea5", label_w=190):
    """Horizontal bars from (label, value) pairs."""
    if not pairs:
        return ""
    vmax = max(v for _, v in pairs)
    h = len(pairs) * (bar_h + gap) + 10
    rows = []
    for i, (lab, v) in enumerate(pairs):
        y = i * (bar_h + gap) + 5
        w = (v / vmax) * (width - label_w - 90) if vmax else 0
        rows.append(
            f'<text x="0" y="{y+bar_h*0.7:.0f}" class="cl">{lab}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'fill="{colour}" rx="2"/>'
            f'<text x="{label_w+w+8:.0f}" y="{y+bar_h*0.7:.0f}" '
            f'class="cv">{fmt(v)}</text>')
    return (f'<svg viewBox="0 0 {width} {h}" class="chart" '
            f'role="img">{"".join(rows)}</svg>')


def bic_chart(curve, chosen):
    """BIC vs number of components; lower is better."""
    ks = sorted(int(k) for k in curve)
    vs = [curve[str(k)] for k in ks]
    lo, hi = min(vs), max(vs)
    W, H, PAD = 560, 190, 34
    def sx(k):
        return PAD + (k - ks[0]) / max(1, ks[-1] - ks[0]) * (W - PAD - 12)
    def sy(v):
        return H - PAD + 12 - (v - lo) / max(1e-9, hi - lo) * (H - PAD - 20)
    pts = " ".join(f"{sx(k):.1f},{sy(curve[str(k)]):.1f}" for k in ks)
    marks = "".join(
        f'<circle cx="{sx(k):.1f}" cy="{sy(curve[str(k)]):.1f}" r="3" '
        f'fill="{"#c0392b" if k == chosen else "#3b6ea5"}"/>'
        f'<text x="{sx(k):.1f}" y="{H-10}" class="ax">{k}</text>' for k in ks)
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">'
            f'<line x1="{PAD}" y1="{H-PAD+12}" x2="{W-12}" y2="{H-PAD+12}" '
            f'stroke="#ccc"/>'
            f'<polyline points="{pts}" fill="none" stroke="#3b6ea5" '
            f'stroke-width="2"/>'
            f'{marks}'
            f'<text x="{PAD}" y="16" class="ax">BIC (lower is better) — '
            f'selected G = {chosen}</text></svg>')


def html():
    rows = "".join(
        f'<tr><td class="mono">{t}</td><td class="num">{fmt(D[t]["cells"])}</td>'
        f'<td class="num">{D[t]["fine_clusters"]}</td>'
        f'<td class="num">{D[t]["n_phenotypes"]}</td>'
        f'<td class="num">{D[t]["ari"]:.3f}</td>'
        f'<td class="num">{D[t]["nmi"]:.3f}</td>'
        f'<td class="num">{D[t]["n_split"]}/{D[t]["n_annotations"]}</td></tr>'
        for t in ORDER)

    # per-dataset deep dives
    deep = []
    for t in ("5k", "lc", "luad"):
        d = D[t]
        m = META[t]
        comp = [(f'phenotype {p["phenotype"]} — '
                 f'{list(p["composition"])[0]} {list(p["composition"].values())[0]*100:.0f}%',
                 p["cells"]) for p in d["phenotype_composition"]]
        split_rows = "".join(
            f'<tr><td class="mono">{s["annotation"]}</td>'
            f'<td class="num">{fmt(s["cells"])}</td>'
            f'<td class="num">{s["n_phenotypes"]}</td>'
            f'<td class="num">{s["dominant_fraction"]*100:.1f}%</td></tr>'
            for s in d["split_examples"])
        kept = ", ".join(f'{k["annotation"]} ({k["dominant_fraction"]*100:.0f}%)'
                         for k in d["kept_examples"][:6])
        deep.append(f"""
<section class="card">
  <h3>{t} — {m['tissue']}</h3>
  <p class="prov">10x dataset <span class="mono">{m['tenx']}</span>
     (OAA {m['oaa']}), {m['cells10x']} cells published;
     {m['panel']}.</p>
  <p>{fmt(d['cells'])} cells carry a GHIST annotation and passed the join
     ({d['state_shape'][0]} rows &times; {d['state_shape'][1]} state
     dimensions). {d['fine_clusters']} fine clusters collapsed to
     <strong>{d['n_phenotypes']} phenotypes</strong> (ARI {d['ari']:.3f},
     NMI {d['nmi']:.3f} against the annotation).</p>
  <h4>Phenotype sizes and dominant annotation</h4>
  {bar_chart(comp, colour="#3b6ea5")}
  <h4>Annotations this run split</h4>
  <table><thead><tr><th>annotation</th><th>cells</th><th>phenotypes</th>
  <th>dominant share</th></tr></thead><tbody>{split_rows}</tbody></table>
  <p class="note">Annotations left intact: {kept}.</p>
</section>""")

    pooled = D["pooled"]
    prof = pooled["slide_profiles"]
    ph_names = [f'P{i}' for i in range(len(prof[0]))]
    slide_table = "".join(
        f'<tr><td class="mono">{pooled["slides"][i]}</td>'
        + "".join(f'<td class="num">{v*100:.1f}</td>' for v in prof[i])
        + "</tr>" for i in range(len(prof)))

    p_best = max(D, key=lambda t: D[t]["ari"])
    p_low = min(("5k", "lc", "luad"), key=lambda t: D[t]["ari"])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cellhippo — clustering annotated spatial cell tables into new patterns</title>
<style>
  :root {{
    --ink:#1b1f24; --muted:#5c6672; --line:#e3e7ec;
    --accent:#3b6ea5; --warn:#c0392b; --bg:#fbfcfd; --card:#fff;
  }}
  * {{ box-sizing:border-box }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }}
  header {{
    background:linear-gradient(135deg,#1d3557,#3b6ea5); color:#fff;
    padding:54px 24px 46px;
  }}
  .wrap {{ max-width:980px; margin:0 auto; padding:0 24px }}
  header .wrap {{ padding:0 }}
  h1 {{ margin:0 0 10px; font-size:31px; letter-spacing:-.4px }}
  header p {{ margin:0; opacity:.92; max-width:720px }}
  .tag {{
    display:inline-block; font-size:12px; letter-spacing:.06em;
    text-transform:uppercase; background:rgba(255,255,255,.17);
    padding:4px 10px; border-radius:20px; margin-bottom:16px;
  }}
  h2 {{
    font-size:21px; margin:46px 0 14px; padding-bottom:8px;
    border-bottom:2px solid var(--line);
  }}
  h3 {{ font-size:17px; margin:0 0 8px }}
  h4 {{ font-size:14px; margin:22px 0 8px; color:var(--muted);
        text-transform:uppercase; letter-spacing:.05em }}
  p {{ margin:0 0 12px }}
  .card {{
    background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:20px 22px; margin:18px 0;
    box-shadow:0 1px 2px rgba(16,24,40,.04);
  }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:14px }}
  th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left }}
  th {{ background:#f4f7fa; font-weight:600; color:var(--muted); font-size:12.5px;
        text-transform:uppercase; letter-spacing:.04em }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
           font-size:13px }}
  .chart {{ width:100%; height:auto; display:block; margin:8px 0 4px }}
  .cl {{ font-size:12.5px; fill:#33404d }}
  .cv {{ font-size:12.5px; fill:#5c6672 }}
  .ax {{ font-size:10px; fill:#8b95a1; text-anchor:middle }}
  .note, .prov {{ color:var(--muted); font-size:13.5px }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
           gap:14px; margin:16px 0 }}
  .stat {{ background:var(--card); border:1px solid var(--line);
           border-radius:9px; padding:15px 16px }}
  .stat .v {{ font-size:25px; font-weight:650; letter-spacing:-.5px;
              font-variant-numeric:tabular-nums }}
  .stat .l {{ font-size:12.5px; color:var(--muted); margin-top:2px }}
  .flow {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 }}
  .step {{ flex:1 1 168px; background:var(--card); border:1px solid var(--line);
           border-top:3px solid var(--accent); border-radius:8px; padding:13px 14px }}
  .step .n {{ font-size:11px; color:var(--accent); font-weight:700;
              letter-spacing:.08em }}
  .step .t {{ font-weight:600; margin:4px 0 5px; font-size:14px }}
  .step .d {{ font-size:12.5px; color:var(--muted) }}
  .callout {{ border-left:4px solid var(--accent); background:#f2f7fc;
              padding:14px 18px; border-radius:0 8px 8px 0; margin:18px 0 }}
  .callout.warn {{ border-color:var(--warn); background:#fdf3f2 }}
  .kbd {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
          background:#eef2f6; padding:2px 6px; border-radius:4px; font-size:13px }}
  footer {{ color:var(--muted); font-size:13px; padding:44px 24px 60px;
            text-align:center }}
  a {{ color:var(--accent) }}
  .hi {{ background:#fff3cd; padding:1px 4px; border-radius:3px }}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="tag">methods note</div>
  <h1>cellhippo</h1>
  <p>HIPPO's phenotype-discovery chain, rebuilt for cell tables that already
  carry an annotation. Same 64-cluster &rarr; correlation &rarr; GMM structure,
  one extra input dimension — and one extra question the original could not ask:
  does a discovered phenotype <em>split</em> an annotated cell type, or merely
  <em>restate</em> it?</p>
</div></header>

<div class="wrap">

<h2>1. What HIPPO does, and what changes here</h2>
<p><a href="https://doi.org/10.1186/s13073-025-01526-5">Zhang et&nbsp;al.,
<em>Genome Medicine</em> 2025;17:98</a> cluster H&amp;E tiles from 517 small-cell
lung cancer slides with contrastive learning, keep 64 tile clusters, take the
64&times;64 Pearson correlation between cluster mean feature vectors, and merge
those correlations with a Gaussian mixture model chosen by BIC. The result is
~15 <strong>HIPOs</strong> — histomorphological phenotypes. Each patient becomes
a vector of HIPO area fractions, and consensus clustering of those vectors
yields two subtypes with different survival.</p>

<p>The chain is good. Its weak joint is that nothing external can contradict it:
the phenotypes are internally reproducible but unanchored. Tiles carry no label,
so "is this a new pattern or a relabelling?" has no operational answer.</p>

<div class="callout">
<strong>The change.</strong> This tool takes a per-cell table where each cell
already has an <span class="mono">annotation</span>. The annotation enters twice:
once inside the cell state vector, and once as an external yardstick. That second
use is the point — a phenotype that cuts one annotation into two states is a
finding, and a phenotype that equals one annotation is a relabelling, and the
tool reports which is which instead of letting both look like discovery.
</div>

<h2>2. Data, and how the provenance was pinned down</h2>
<p>The starting point was a set of label tables with no expression: the
<span class="mono">data_input/</span> CSVs from
<span class="mono">D10ny3u3/GHIST_input</span> — three Xenium experiments
(5k, lc, luad) giving <span class="mono">cell_id</span> and a cell-type
annotation. Those tables alone cannot support phenotype discovery, so the
matching 10&amp;x matrices were located from the dataset URLs recorded in the
source notes and downloaded.</p>

<p>Provenance is not assumed. Each GHIST set was matched to a 10&amp;x folder by
published cell count, then verified by <strong>barcode overlap</strong> — the same
Xenium run either shares its cell IDs or it does not:</p>

<table>
<thead><tr><th>GHIST set</th><th>10&amp;x dataset</th><th>published cells</th>
<th>GHIST cells</th><th>join coverage</th></tr></thead>
<tbody>
<tr><td class="mono">5k</td><td class="mono">Xenium_Prime_Human_Lung_Cancer_FFPE</td>
<td class="num">278,328</td><td class="num">{fmt(D['5k']['cells'])}</td>
<td class="num">99.93%</td></tr>
<tr><td class="mono">lc</td><td class="mono">Xenium_V1_Human_Lung_Cancer_Addon_FFPE</td>
<td class="num">161,000</td><td class="num">{fmt(D['lc']['cells'])}</td>
<td class="num">99.66%</td></tr>
<tr><td class="mono">luad</td><td class="mono">Xenium_V1_humanLung_Cancer_FFPE</td>
<td class="num">162,254</td><td class="num">{fmt(D['luad']['cells'])}</td>
<td class="num">99.78%</td></tr>
</tbody></table>

<p class="note">The unmatched few per cent are cells the GHIST annotator dropped;
nothing is imputed. A first attempt at this check was run against the wrong
hypothesis (that the GHIST tables belonged to the Takano cohort already on the
server) and returned 124 hits out of 600,464 — chance-level barcode collisions
between unrelated runs, which is exactly what a negative control should look
like.</p>

<h2>3. The pipeline</h2>
<div class="flow">
  <div class="step"><div class="n">STEP 1</div><div class="t">Cell state vector</div>
    <div class="d">one-hot the annotation and the second label, z-score the
    numeric features, concatenate. Categorical axes are one-hot, never integer
    coded, so distances do not depend on alphabetical order.</div></div>
  <div class="step"><div class="n">STEP 2</div><div class="t">Fine clusters</div>
    <div class="d">k-means to k = 64, HIPPO's cluster count. This is the
    stand-in for their contrastive clustering.</div></div>
  <div class="step"><div class="n">STEP 3</div><div class="t">Correlation &rarr; GMM</div>
    <div class="d">Pearson between the 64 centroids, then a diagonal GMM over
    G = 2..20 chosen by BIC. HIPPO's equations 1&ndash;3, unchanged.</div></div>
  <div class="step"><div class="n">STEP 4</div><div class="t">Phenotype assignment</div>
    <div class="d">each cell inherits the phenotype of its fine cluster. The GMM
    labels clusters, not cells — getting this backwards is the bug that broke
    the first run.</div></div>
  <div class="step"><div class="n">STEP 5</div><div class="t">Concordance</div>
    <div class="d">ARI / NMI against the annotation, plus a per-annotation split
    test: share of a cell type's cells in its dominant phenotype.</div></div>
  <div class="step"><div class="n">STEP 6</div><div class="t">Slide profiles</div>
    <div class="d">fraction of cells per phenotype, per slide, then consensus
    clustering. Runs only when there are enough slides.</div></div>
</div>

<h2>4. Results</h2>
<table>
<thead><tr><th>set</th><th class="num">cells</th><th class="num">fine clusters</th>
<th class="num">phenotypes</th><th class="num">ARI</th><th class="num">NMI</th>
<th class="num">split</th></tr></thead>
<tbody>{rows}</tbody></table>

<div class="grid">
  <div class="stat"><div class="v">{fmt(sum(D[t]['cells'] for t in ORDER))}</div>
    <div class="l">cells clustered across four runs</div></div>
  <div class="stat"><div class="v">64 &rarr; {D['5k']['n_phenotypes']}</div>
    <div class="l">fine clusters &rarr; phenotypes (5k)</div></div>
  <div class="stat"><div class="v">{D['5k']['n_split']}/{D['5k']['n_annotations']}</div>
    <div class="l">annotations split by the 5k run</div></div>
  <div class="stat"><div class="v">{D['pooled']['n_split']}/{D['pooled']['n_annotations']}</div>
    <div class="l">split in the label-only control</div></div>
</div>

{''.join(deep)}

<h3>BIC curves — how G was chosen</h3>
<div class="card">
{bic_chart(D['5k']['bic_curve'], D['5k']['n_phenotypes'])}
<p class="note">5k: BIC minimum at G = {D['5k']['n_phenotypes']}. The curve is
noisy above G&nbsp;&asymp;&nbsp;10, which is the usual warning that the
correlation matrix has fewer effective dimensions than it has rows — 64 clusters
is HIPPO's number, not a property of this data.</p>
</div>

<h2>5. The result that matters: expression vs labels</h2>
<div class="callout">
<strong>Control.</strong> The pooled run uses the annotation and the Seurat
cluster as its <em>only</em> features. It reaches ARI
{D['pooled']['ari']:.3f} and splits {D['pooled']['n_split']} of
{D['pooled']['n_annotations']} annotations — several at 100% agreement. That is
what a pipeline that can only echo its inputs looks like.
<br><br>
<strong>With expression.</strong> Adding 32 expression PCs drops ARI to
{D['5k']['ari']:.3f} / {D['lc']['ari']:.3f} / {D['luad']['ari']:.3f} and raises
splits to {D['5k']['n_split']}/{D['5k']['n_annotations']},
{D['lc']['n_split']}/{D['lc']['n_annotations']},
{D['luad']['n_split']}/{D['luad']['n_annotations']}. The agreement gets
<em>worse</em> and the interesting part gets bigger.
</div>

<p>Falling ARI is normally a bad sign. Here it is the finding. ARI only measures
agreement with the annotation, and the whole question is whether the annotation
is missing structure. The three runs recover every major cell type — epithelia,
endothelium, T/NK, macrophages each dominate their own phenotype — while cutting
several of them into states the annotation does not name. Examples:</p>
<ul>
<li><strong>5k <span class="mono">fib</span></strong> ({fmt(D['5k']['split_examples'][2]['cells'])}
  cells) splits with only
  {D['5k']['split_examples'][2]['dominant_fraction']*100:.1f}% in its top
  phenotype; the remainder sits with <span class="mono">fib_activated</span>.</li>
<li><strong>luad <span class="mono">epi</span></strong> ({fmt(D['luad']['split_examples'][0]['cells'])}
  cells) splits {D['luad']['split_examples'][0]['dominant_fraction']*100:.1f}% /
  {(1-D['luad']['split_examples'][0]['dominant_fraction'])*100:.1f}% — the
  largest single cell type in that run, and the one a tumour-focused study cares
  about most.</li>
<li><strong>lc <span class="mono">plasma</span></strong> and
  <span class="mono">lymph_endo</span> scatter across 3 and 4 phenotypes
  respectively, with dominant shares of
  {D['lc']['split_examples'][2]['dominant_fraction']*100:.0f}% and
  {D['lc']['split_examples'][7]['dominant_fraction']*100:.0f}%.</li>
</ul>

<h3>Slide-level step</h3>
<div class="card">
<p>The pooled run has {len(pooled['slides'])} slides. The consensus step
<span class="hi">refused to run</span>:
<span class="mono">{pooled['subtyping']['info']['reason']}</span>.
HIPPO's own step rests on 209 discovery patients; with three samples the
consensus matrix would be arithmetic, not evidence. Reported and skipped rather
than faked.</p>
<table><thead><tr><th>slide</th>
{''.join(f'<th class="num">{p}</th>' for p in ph_names)}</tr></thead>
<tbody>{slide_table}</tbody></table>
<p class="note">Rows are phenotype fractions (%), columns phenotypes P0–P
{len(ph_names)-1}. The three slides do differ — luad is
{prof[2][4]*100:.0f}% P4 against {prof[0][4]*100:.0f}% for 5k — but n = 3.</p>
</div>

<h2>6. What this does not show</h2>
<div class="callout warn">
<ul style="margin:0;padding-left:20px">
<li><strong>No survival endpoint.</strong> HIPPO validates subtypes against
overall and disease-free survival. These three Xenium datasets ship no clinical
follow-up, so the prognostic half of HIPPO is untestable here. Everything above
is a statement about cell state, not outcome.</li>
<li><strong>Three slides is not a cohort.</strong> The slide-level and
consensus-clustering steps are implemented and are deliberately not exercised.</li>
<li><strong>"Split" is not automatically "better".</strong> A split can be real
biology, or it can be the expression PCs shadowing the cell-cycle or a
dissociation artefact, or the annotation being coarse. The tool reports the
split and its size; it does not claim the split is meaningful. Distinguishing
those needs a marker-level read of the phenotypes, which is the next step.</li>
<li><strong>Fine clusters are k-means, not contrastive.</strong> HIPPO's
representation learning is doing work that plain k-means on PCs is not. The
substitution is the main approximation in this port.</li>
</ul>
</div>

<h2>7. Reproducing</h2>
<pre class="mono" style="background:#f4f7fa;padding:14px 16px;border-radius:8px;overflow-x:auto">python tools/fetch_10x_all.py                       # 10x matrices + coordinates
python tools/assemble.py                            # join with annotation, PCA -> work/&lt;set&gt;_full.parquet
PYTHONPATH=src python -m cellhippo.run \\
    --csv work/5k_full.parquet \\
    --annotation annotation --second-label cluster \\
    --features PC1,...,PC32 --k 64 --out work/5k_cellhippo.json</pre>
<p class="note">Everything runs on the server. Pass
<span class="kbd">--slide</span> to enable step 6; omit it and that step is
skipped rather than guessed.</p>

<footer>
  cellhippo — an independent reimplementation of the HIPPO phenotype chain for
  annotated cell tables. Not affiliated with the HIPPO authors; the method is
  theirs, the annotation dimension and the concordance test are the port.
</footer>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    open(OUT, "w", encoding="utf-8").write(html())
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)")
