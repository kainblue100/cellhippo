# cellhippo

HIPPO's phenotype-discovery chain, run on three modalities at once: the **H&E
image**, the **Xenium expression matrix**, and the **cell-type annotation** the
original method never had.

HIPPO ([Zhang et al., *Genome Medicine* 2025;17:98](https://doi.org/10.1186/s13073-025-01526-5))
tiles H&E slides, clusters the tiles contrastively into 64 groups, correlates the
cluster centroids, and merges them with a BIC-selected Gaussian mixture into
histomorphological phenotypes. Its tiles carry no label, so nothing external can
contradict them. Here each tile also knows which cell types it contains, and the
image sits in the feature vector alongside the expression — so the two can be
ablated against each other instead of asserted.

**[Read the walkthrough](docs/index.html)** — registration, tiling, clustering,
figures, ablations.

## Pipeline

| step | what happens | reference |
|---|---|---|
| 1 | register cells to the H&E: rotation from `he_imagealignment.csv`, scale from the OME `PhysicalSizeX`, translation by second-moment matching | this port |
| 2 | tile 224×224 px at ~1 µm/px, non-overlapping, ≥60% stained tissue, ≥10 cells | HIPOS |
| 3 | Reinhard stain normalisation in LAB, cross-slide (source stats per slide, target fixed) | HIPOS |
| 4 | contrastive CNN trained from scratch on the tiles, 128-dim embedding | HIPOS step 1 |
| 5 | fuse image ⊕ expression PCs ⊕ annotation fractions, z-scored per block | GHIST+ input contract |
| 6 | k-means (k=64) → Pearson between centroids → diagonal GMM with G by BIC | HIPPO eq. 1–3 |
| 7 | representative-tile gallery, spatial unit map, composition heatmaps | HIPPO Figs 2C, 3A, 5B |

## Install

```bash
pip install numpy pandas scipy scikit-learn h5py pyarrow tifffile \
            imagecodecs pillow scikit-image torch
```

## Use

```bash
python tools/fetch_10x_all.py            # matrices + coordinates + H&E
python tools/register_final.py           # rotation, scale, translation
python tools/extract_tiles.py --tag 5k   # tiling + three modalities
python tools/train_cnn.py --tiles work/5k_tiles.npz work/lc_tiles.npz \
    --epochs 80 --clusters 64 --out work/cnn_tiles.npz
python tools/cluster_tiles.py --tags 5k lc --img --expr --ann --name fused
python tools/make_figures.py --name fused --tags 5k lc
```

`cluster_tiles.py` takes `--img/--expr/--ann` independently, so any subset is an
ablation.

## Results

2,086 tiles from two Xenium slides (5k: 722, lc: 1,364). The fused run gives 64
fine clusters and 5 phenotypes:

| phenotype | n tiles | dominant morphology |
|---|---|---|
| P0 | 39 | tumour/epithelial nests, gland-like |
| P1 | 275 | dense collagenous stroma, paucicellular |
| P2 | 42 | fibrous stroma, more uniform, less pigment |
| P3 | 338 | mixed: stroma + lymphocyte clusters + haemorrhage |
| P4 | 28 | gland-forming epithelium, desmoplastic stroma |

The spatial map shows strong autocorrelation — neighbouring tiles mostly share a
phenotype, with smooth boundaries between large domains — so the phenotypes
correspond to anatomical regions rather than noise.

## Registration

`he_imagealignment.csv` could not be used as shipped: its linear part implies
1.288 px/µm while the OME metadata on the same TIFF says 0.2738 µm/px, a factor
2.84 apart, and the translation term corresponds to neither.

Each component was fixed from independent evidence:

- **rotation** — the matrix's own −89.6°, which second-moment matching reproduces
  at −85.6° and −93.4° on the two slides (within 4°)
- **scale** — `cell_area` gives an 11.4 µm cell on lc; the nuclei measured in the
  H&E at the OME pixel size are 12.1 µm. At the matrix's scale they would be
  34 µm, which is not a lung cell. Moment matching gives 0.271 µm/px against the
  OME's 0.274
- **translation** — second-moment matching of the cell cloud's covariance ellipse
  to the tissue mask

Earlier attempts failed in instructive ways, all documented in
`tools/register_final.py` and the page: IoU and "fraction of cloud on tissue" both
reward shrinking the cloud and slid to the search boundary, and the first tissue
mask found 30% tissue where the eye sees 60% because Otsu on luminance misses pale
stroma.

## Limitations

- **Two slides, 2,086 tiles.** HIPPO used 227,000 tiles from 209 patients. Nothing
  here supports a population claim.
- **No survival endpoint.** These datasets ship no clinical follow-up, so HIPPO's
  prognostic half is untestable.
- **The image encoder is small** — 1.21 M parameters against HIPPO's ResNet50 on
  a 100× larger tile set. Contrastive loss fell 4.08 → 0.85, but the image branch
  is the weakest part of this port.
- **G is not a result.** Phenotype counts move with the BIC cap and the block
  weighting; the ablations are for reading the galleries side by side.

## Layout

```
src/cellhippo/run.py          HIPPO chain for annotated cell tables (no image)
tools/fetch_10x_all.py        matrices + coordinates + H&E from the 10x CDN
tools/register_final.py       rotation / scale / translation, with the evidence
tools/extract_tiles.py        HIPPO-style tiling, Reinhard, three modalities
tools/train_cnn.py            contrastive CNN on tiles
tools/cluster_tiles.py        HIPPO eq. 1-3 on the fused block, with ablations
tools/make_figures.py         gallery / spatial / heatmap panels
tools/build_page.py           generate docs/index.html
docs/index.html               the walkthrough
```
