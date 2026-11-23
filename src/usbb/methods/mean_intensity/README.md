# Mean marker intensity (`mean_intensity`)

Verdict C. Tier v0. Issue: to open.

## What it is
Baseline protein embedding: each unit's value for every marker of its protein table, as `arcsinh(value / cofactor)`, one column per marker (working note, 16 Nov 2026: mean intensity, any panel).

## Weights
None.

## Inputs
Protein at spots, `raw_counts`: the antibody counts of the tonsil CytAssist store (35 markers, 4 of them isotype controls) and of the synthetic store. Markers keep the provider's ids; no vocabulary.

## Pitfalls on our data
- A card has one value scale, so cells and pseudospots, which hold `raw_intensity` (synthetic store, PhenoCycler data later), are refused until the card format accepts several scales or a second card is added.
- The cofactor below suits counts; fluorescence intensities need their own, to choose on the PhenoCycler data.

## Why the code looks like this
- arcsinh, not the values as they are: tonsil counts span orders of magnitude within a marker (KRT5: 648 at the 5th percentile, 180,655 at the 99th), and the probes' standardisation rescales a column but does not undo its skew. Unlike log, arcsinh accepts zeros (46 tonsil values).
- Cofactor 5: values well above it are compressed like a log, values below stay close to linear. On tonsil, 99.9 % of counts are at least 100, where arcsinh(x / 5) equals log(2x / 5) to within 0.001, so the cofactor only shifts each marker by a constant that the probes' standardisation removes.
- Isotype controls are kept: they measure non-specific binding in each unit, which a probe can use. Choosing markers belongs to the evaluation.
- Frozen, although the working note files mean intensity with the fitted baselines (verdict C): with a fixed cofactor nothing is learnt from the data.

## Done when
- [x] `pytest tests/test_contract.py -k mean_intensity` passes: refusals before `load`, row permutation, same seed, output spec.
- [ ] On the tonsil spots, the latent has 35 columns, one per row of `spot_protein.var`.
- [ ] The evaluation's probes on it beat `random_features`.
