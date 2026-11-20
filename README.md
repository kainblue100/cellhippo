# cellhippo

HIPPO's histomorphological phenotype chain, rebuilt for cell tables that already
carry an annotation.

HIPPO ([Zhang et al., *Genome Medicine* 2025;17:98](https://doi.org/10.1186/s13073-025-01526-5))
takes unlabelled H&E tiles, contrastively clusters them into 64 groups, merges the
64×64 inter-cluster correlation matrix with a BIC-selected Gaussian mixture model
into ~15 phenotypes, turns each patient into a phenotype-fraction vector, and
consensus-clusters those vectors into subtypes. It works, and nothing external can
contradict it: tiles carry no label, so "new pattern" and "relabelling" are
indistinguishable by construction.

This tool keeps the chain and adds one input dimension — a per-cell annotation. The
annotation enters twice: inside the cell state vector, and as an external yardstick.
The second use answers a question the original cannot: **does a discovered phenotype
split an annotated cell type, or merely restate it?**

**[Read the walkthrough](docs/index.html)** — full pipeline, per-step results, and
the run-by-run numbers.

## Method

| step | what happens | HIPPO analogue |
|---|---|---|
| 1 | cell state vector: one-hot labels + z-scored numeric features | tile features |
| 2 | k-means to k = 64 fine clusters | contrastive clustering |
| 3 | Pearson between cluster centroids → diagonal GMM over G = 2..20 by BIC | eq. 1–3, unchanged |
| 4 | each cell inherits its fine cluster's phenotype | HIPO assignment |
| 5 | ARI/NMI vs the annotation + per-annotation split share | not in the original |
| 6 | slide phenotype profiles → consensus clustering | eq. 4 + ConsensusClusterPlus |

Step 6 runs only when there are enough slides to make a consensus meaningful;
otherwise it reports why it was skipped.

## Install

```bash
pip install numpy pandas scipy scikit-learn h5py pyarrow
```

## Use

```bash
PYTHONPATH=src python -m cellhippo.run \
    --csv cells.parquet \
    --annotation annotation \
    --second-label cluster \
    --features PC1,PC2,PC3 \
    --slide slide \
    --k 64 \
    --out result.json
```

Input contract:

- `cell_id` — required, unique
- `annotation` — required; the extra dimension
- a second categorical column (`--second-label`) — optional
- numeric columns named in `--features` — optional
- `slide` (`--slide`) — optional; without it step 6 is skipped, not faked

Accepts CSV or parquet. Output is a JSON with the BIC curve, phenotype sizes,
phenotype ↔ annotation composition, ARI/NMI, and the per-annotation split table.

## Results

Four runs, all on real 10x Xenium data joined to the
[GHIST_input](https://github.com/D10ny3u3/GHIST_input) annotations:

| set | cells | fine clusters | phenotypes | ARI | NMI | annotations split |
|---|---|---|---|---|---|---|
| 5k | 278,146 | 64 | 10 | 0.572 | 0.657 | 13 / 17 |
| lc | 160,448 | 64 | 6 | 0.555 | 0.592 | 9 / 14 |
| luad | 161,894 | 64 | 5 | 0.488 | 0.554 | 7 / 10 |
| pooled (label-only control) | 600,488 | 64 | 9 | 0.761 | 0.800 | 4 / 23 |

The control is the point. With label-only features the pipeline reaches ARI 0.761
and leaves 19 of 23 annotations intact at 100% agreement — it can only echo its
inputs. Adding 32 expression PCs *lowers* ARI to 0.49–0.57 and raises splits to
7–13 per run. Agreement with the annotation drops while the pipeline recovers every
major cell type and cuts several into states the annotation does not name.

## Provenance

The GHIST tables carry no expression, so the matching 10x matrices were located
from the dataset URLs in the source notes and downloaded. The mapping is verified by
barcode overlap, not assumed:

| GHIST set | 10x dataset | join coverage |
|---|---|---|
| 5k | `Xenium_Prime_Human_Lung_Cancer_FFPE` | 99.93% |
| lc | `Xenium_V1_Human_Lung_Cancer_Addon_FFPE` | 99.66% |
| luad | `Xenium_V1_humanLung_Cancer_FFPE` | 99.78% |

## Limitations

- **No survival endpoint.** HIPPO validates subtypes against OS/DFS. These datasets
  ship no clinical follow-up, so the prognostic half is untestable here.
- **Three slides.** The slide-level consensus step is implemented and deliberately
  not exercised.
- **A split is not automatically meaningful.** It can be biology, cell-cycle
  structure, a dissociation artefact, or a coarse annotation. The tool reports the
  split and its size and does not adjudicate.
- **k-means, not contrastive learning** for step 2 — the main approximation in
  this port.

## Layout

```
src/cellhippo/run.py          the tool
tools/find_10x_datasets.py    locate the source datasets on the 10x CDN
tools/fetch_10x_all.py        pull the 10x matrices + coordinates
tools/map_ghist_to_10x.py     verify the mapping by barcode overlap
tools/inspect_h5.py           dump the h5 layout
tools/assemble.py             join expression + annotation + coordinates, PCA
tools/make_payload.py         distil results for the page
tools/build_page.py           generate docs/index.html
docs/index.html               the walkthrough
```
