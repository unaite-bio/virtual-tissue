# Random features (`random_features`)

Verdict n/a. Tier v0. Issue: to open.

## What it is
Negative control: each unit gets `n_features` standard normal values from a generator seeded by the run seed and the unit id. No value, position or image is read, so the features carry no biology and set the floor of every probe.

## Weights
None.

## Inputs
Protein at spots (`raw_counts`), read only for its unit ids. Protein is the one modality our stores hold at every unit type (`spot_protein`, `cell_protein`, `pseudospot_protein`). Its bundle is also the one `mean_intensity` reads, so the control costs no extra preparation.

## Pitfalls on our data
- A card has one value scale, so cells and pseudospots (`raw_intensity` in the synthetic store, PhenoCycler data later) are refused for now.
- Its output declares `informed_by: []` (allowed for controls only): no protein value is read, so the evaluation can probe it for every target, protein included.

## Why the code looks like this
- One generator per unit, not one per bundle: a unit gets the same features whatever the other units, their order or the split, so outputs of different runs join on unit ids.
- The generator seed is the sha256 of `<seed>:<unit id>`, which is the same in every process and on every machine, unlike Python's `hash()` of a string.
- `n_features` is 50, the `hvg_pca` default, so probes on the control and on that baseline get the same number of columns.

## Done when
- [x] `pytest tests/test_contract.py -k random_features` passes: refusals before `load`, row permutation, same seed, output spec.
- [ ] Two runs with different seeds on the tonsil spots give different features for every unit.
- [ ] The evaluation reports it next to every method, on every probe.
