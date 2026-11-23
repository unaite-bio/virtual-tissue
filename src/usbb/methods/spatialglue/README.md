# SpatialGlue (`spatialglue`)

Verdict C. Tier T1. Issue: to open.

## What it is
A graph neural network with dual attention. For each of two modalities it combines a spatial neighbour graph and a feature graph, then fuses the modalities by attention. It returns a joint latent `SpatialGlue` (64 values, L2-normalised), one latent per modality, and per-unit modality weights `alpha` (n, 2) (`SpatialGlue_pyG.py`, `model.py`). Paper: [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02316-4); code: [JinmiaoChenLab/SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue).

## Weights
None: fitted on our data, on the units it embeds.

## Inputs
- One feature matrix per modality (`obsm["feat"]`). Widths may differ, and both are mapped to `dim_output=64` (`SpatialGlue_pyG.py`); the working note's "equal dimensions" is the tutorial's choice. So it runs as a fuser on two upstream embeddings (`upstream: {kind: embedding}`), which gives the encoder by fuser grid of proposal §4.1.
- The [tutorial](https://spatialglue-tutorials.readthedocs.io/en/latest/) builds RNA features with seurat_v3 HVGs (3,000), normalise, log1p, scale and PCA, and protein features with CLR, scale and PCA, both to n_proteins - 1 components. To reproduce it, set `hvg_pca` and `pca_protein` upstream to the same steps (to check that they apply them).
- Both modalities on the same units of the same section; spots or pseudospots. Spatial graph: 3 neighbours on `obsm["spatial"]`; feature graph: 20 neighbours, correlation (`preprocess.py`).
- Fit mode for the card: transductive (fits on the test units, no labels).

## Pitfalls on our data
- Same spots only: C on Simkin and Haist, D on Causer and CORAL, whose modalities are on separate sections (working note, 16 Nov 2026). A bundle does not say whether its modalities share a section (`pairing` stays in the store's attrs), so who refuses adjacent-section input is open.
- Old environment: `requirements.txt` pins python==3.8, anndata==0.8.0, scanpy==1.9.1, numpy==1.22.3 and torch>=1.8.0, but `setup.py` installs none of them (`install_requires` is only `requests`): whether newer versions work is to verify. R 4.0.3 and rpy2 serve only `utils.mclust_R` (clustering), which we do not need. Own env group.
- The four adjacency matrices are dense n x n (`toarray()` in `preprocess.py`). One Visium section fits (shown on 2,568 to 9,752 spots, about 5 min for 9,752; working note), but 100,000 units need 40 GB per matrix in float32 (our arithmetic). Haist's 1.5M cells need pseudospots or one run per core: to decide.
- `datatype` overrides `epochs` and `weight_factors` without warning (`'10x'`: 200 epochs, `[1, 5, 1, 10]`; default `'SPOTS'`: 600 epochs), so the card exposes `datatype`, not `epochs`. Defaults were tuned on the authors' lymph-node CytAssist data (GEO GSE263617): keep one declared setting for every dataset (proposal §3.2, §4.1).
- `Train_SpatialGlue` stores `random_seed` but never uses it: call `fix_seed(self.seed)` first. The tutorial warns of slight sensitivity to seed and GPU.
- The per-modality latents use one modality at inference but are trained with cross-reconstruction of the other, so their `informed_by` lists both (our reading of `model.py`). `alpha` has no slot in the output file yet.
- Licence: `LICENSE.md` is AGPL-3.0 (GitHub), while `setup.py` and PyPI say MIT: to verify before copying any of its code.

## Why the code looks like this
To write with the implementation.

## Done when
- [ ] The contract test passes: `pytest src/usbb/methods/spatialglue`.
- [ ] The cluster test ran in the method's env group, and its `RUN.json` summary is in the PR.
- [ ] No pretrained weights, so no §3.3 row; the PR says whether any dataset it ran on is the authors' development data (GSE263617, proposal §3.2).
- [ ] With the tutorial's features and `fix_seed`, our `SpatialGlue` latent equals the tutorial code's on the same spots and GPU.
