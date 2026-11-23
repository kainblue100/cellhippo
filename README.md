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
| 8 | 3D model: infer each cell's z from the sectioning relation, render as spheres coloured by phenotype | CellSqueeze3D; Alon et al. |

## The 3D model

A 2D section still carries the third axis. Sectioning a sphere of radius `R` at
distance `z` from its centre leaves a disc of radius `a = sqrt(R² − z²)`, so the
apparent area is the size of the cut, not the size of the cell —
[CellSqueeze3D](https://doi.org/10.1002/advs.202518706) (Kong & Lu, *Adv Sci*
2026) turns that into an inverse problem. The result is rendered the way
[Alon et al.](https://arxiv.org/abs/2101.11600) build theirs: explicit spheres
carrying position, radius and the learned phenotype, not a voxel grid.

One number per cell cannot determine both `R` and `z`. What breaks the degeneracy
is that **cells of one phenotype share a true radius**, so a small apparent area
reads as a cut away from the equator rather than a small cell. The sign of `z` is
free and is chosen to reduce overlap.

Two limits are reported rather than hidden:

- the group radius is capped by the packing the density can hold — `N` spheres of
  radius `R` in area `A` and slab `2t` cannot exceed a packing fraction near 0.6
- at tissue density the 3D adjustment cannot clear every conflict. It resolves
  31% of the 2D conflicts on 5k and 18% on lc; the rest is the sphere assumption
  at the jamming limit, not a tuning failure

The page has an interactive viewer (plain canvas 2D, no three.js, no CDN): drag
to rotate, scroll to zoom, step through phenotypes, and a cross-section toggle
that hides everything below `z = 0` — the cells that remain are the ones the knife
actually cut.

### Volume render

Free spheres and metaballs were tried first, in Blender, and both capped out at
3–5/10 — the expected ceiling for a surface mesh, which cannot produce internal
scattering or depth falloff. `tools/volume_render_cells.py` is a front-to-back
emission-absorption ray march instead: per-sample opacity, gradient shading,
filmic shoulder. Pixel statistics for both slides: `mean 24–32, max 207–224,
corner [0,0,0], lit 26–29%`.

It is **not** at the level of a figure rendered from a real 3D microscopy stack,
and the limitation is the data. This model is ~20 uniform spheres whose positions
were inferred from a single section, so the only structure available is the sphere
and the core drawn inside it. Cells sit ~12 µm apart at this density, so a field
wide enough to hold forty of them is ~85 µm across while `z` spans only ±6 µm — a
wafer, not a block.

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
python tools/build_3d.py --tag 5k --block 1 --slab-um 6   # 2D -> 3D
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
tools/build_3d.py             2D -> 3D cell inference (sectioning relation)
tools/volume_render_cells.py  emission-absorption volume render of that model
tools/blender_scene.py        Blender scene for the surface-based attempt
tools/viewer3d.py             the interactive canvas viewer for the page
tools/build_page.py           generate docs/index.html
docs/index.html               the walkthrough
```
