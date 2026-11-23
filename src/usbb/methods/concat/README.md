# Early fusion (`concat`)

Verdict A. Tier v0. Issue: to open.

## What it is
Baseline fuser: every latent of every upstream output in the bundle, side by side, ordered by file name then latent name; `obsm["spatial"]` is left out. Nothing is fitted (working note, 16 Nov 2026: early fusion by concatenation, verdict A).

## Weights
None.

## Inputs
The outputs of the upstream methods the experiment names, copied into the bundle as `<upstream_method_id>.h5ad`. Any unit type, as long as every upstream output holds the same units in the same order (`read_input` refuses the bundle otherwise).

## Pitfalls on our data
- Fusion needs the same spot or cell in every upstream output, so adjacent sections (Causer) first need a cell-to-spot mapping (working note).
- A wide block outweighs a narrow one, for example a 1,536-dimension image vector next to 53 markers (working note).

## Why the code looks like this
- No scaling: scaling needs statistics fitted on training units, and a frozen fuser sees only the test bundle; fitting them there would make it transductive. The evaluation's probes standardise every column with statistics from the training units. Balancing blocks of different widths, which the working note suggests (scaling or PCA per block), would be a fitted fuser of its own.
- Sorted file names, then sorted latent names, so the column order follows from the bundle alone.

## Done when
- [x] `pytest tests/test_contract.py -k concat` passes, with `random_features` and `mean_intensity` as upstream outputs.
- [ ] On the tonsil spots, the latent's width is the sum of the upstream widths, and its first columns equal the latent of the upstream file that sorts first.
