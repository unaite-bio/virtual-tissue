# HVG + PCA (`hvg_pca`)

Verdict C. Tier v0. Issue: to open.

## What it is
Baseline RNA embedding: counts per 10,000 then log1p, the `n_top_genes` genes of highest variance, then PCA with `n_components` components, all fitted on the train bundle and applied to the test bundle (working note, 16 Nov 2026: log-normalised HVGs and PCA, fitted per dataset).

## Weights
None.

## Inputs
RNA raw counts at spots, the only unit our stores hold RNA for (`spot_rna`, `raw_counts` in the tonsil and synthetic stores). Genes keep the provider's ids; no vocabulary.

## Pitfalls on our data
- A unit without counts cannot be normalised, so `check` refuses its bundle (none among the 4,194 tonsil spots).
- With fewer train units or kept genes than `n_components`, the latent has fewer columns (the synthetic store has 20 genes).
- The count matrix is read dense: 4,194 x 18,085 values on tonsil, about 0.6 GB per float64 copy.
- Train units from several sections or patients put their batch effects into the components; the baseline measures that too.

## Why the code looks like this
- numpy only, so it runs in the core env on Python 3.8+ without scanpy or scikit-learn.
- Genes are ranked by the plain variance of the log-normalised values. Scanpy's `highly_variable_genes` ([docs](https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pp.highly_variable_genes.html)) ranks differently: its default `seurat` flavour, like `cell_ranger`, uses the dispersion normalised within 20 bins of mean expression, which removes the link between mean and variance that our ranking keeps; `seurat_v3` reads counts and ranks by the variance after a regularised z-score.
- The target is a fixed 10,000 counts, where scanpy's `normalize_total` defaults to the median total of the units: a median would change with the units in the bundle, so train and test units would not be scaled alike.
- Genes are centred, not scaled, before PCA. Train and test bundles come from one `prepare` call, so they hold the same genes in the same order and the gene indices kept by `fit` apply to the test bundle.

## Done when
- [x] `pytest tests/test_contract.py -k hvg_pca` passes: refusals before `load`, row permutation, same seed, output spec.
- [ ] On the tonsil spots with a patient or section split, the latent has 50 columns and the run fits only on the train bundle (`RUN.json` lists both bundles).
- [ ] The evaluation's probes on it beat `random_features`.
