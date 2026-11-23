# Eva (`eva`)

Verdict B. Tier T1. Issue: to open.

## What it is
A masked-autoencoder vision transformer pretrained on multiplexed immunofluorescence with matched H&E (about 64M cells, 4,159 regions). Every channel is tagged with its marker name. It takes one 224 px patch holding any named protein channels plus three H&E channels and returns one vector per patch, the mean of its tokens, 768 values ([repository](https://github.com/YAndrewL/Eva), [preprint](https://doi.org/10.64898/2025.12.10.693553), tutorial 01).

## Weights
- [`yandrewl/Eva`](https://huggingface.co/yandrewl/Eva), gated with automatic approval (Hub API; working note, 16 Nov 2026). Use `Eva_model.ckpt`; `Eva_ft.ckpt` is the fine-tune for virtual staining. On 23 Nov 2026 `main` = `4341b0296ada53b02ce277e0430a6c5760a5a2ec` (last change 19 Aug 2026, a re-upload with no changelog, proposal §3.3) and `Eva_model.ckpt` sha256 = `6c507aa4aa704d3b4f33c200fcc9bf5de1b511a781116de3edd380633419132f`. Pin both in the card.
- Licence of weights and code: none on the Hub card, and the repository has no licence file. To verify before use. Its training data is CC BY-NC-ND 4.0.
- Also needed: the GenePT marker embeddings, file `GenePT_gene_protein_embedding_model_3_text.pickle` from Zenodo [10833191](https://zenodo.org/records/10833191) (CC BY 4.0; Eva README), and the code at a pinned commit (`main` = `35dd50ce557e72474fe48da7e871afcb8e63298a` on 23 Nov 2026). The card's `weights` names one repository, so how to pin these two is open.

## Inputs
- Two image modalities, `protein` and `he`, cut around the same unit at the same µm per pixel and 224 px: Eva stacks them on one pixel grid, H&E last, as channels `HECHA1`, `HECHA2`, `HECHA3` (tutorial 02). Spots and pseudospots; cells wait for the open point on protein images at cell scale.
- Pretraining pixel sizes ranged from 0.3775 to 1 µm/px by cohort, patches about 100 µm wide (preprint, Methods); our value is to choose with the data producers. Protein intensities are clipped and scaled to [0, 1] with bounds from histogram thresholding (exact rule: to verify); H&E is divided by 255, then inverted to `1 - x` (preprint, Methods).
- Marker names must be keys of `marker_to_gene` in Eva's `utils/constant.py` (about 200, e.g. `CD3e`, `PanCK`); any other name raises `KeyError`. Keep an explicit map from our `protein_name` to these keys, and make `check` refuse unmapped markers, naming all of them.
- Our bundle format gives image patches no channel names and stores them as uint8; both must be settled for protein patches before this method (open design point).

## Pitfalls on our data
- Pretraining exposure: UPMC-HNC, a head and neck PhenoCycler cohort, is in Eva's training (§3.3, Y). Badoual's PhenoCycler data was processed on Enable Medicine and Eva also used unspecified additional images, so written confirmation is needed before data freeze; spora cohorts, Haist among them, count as possibly seen (§3.1, §3.2).
- Same section only: Badoual and Haist (H&E after CODEX on the same section) fit. Causer's protein is on an adjacent section and its H&E is 3.28 µm/px, so no joint input there; CORAL has no images (working note, 16 Nov 2026).
- In `Eva/utils.py`, `extract_features(channel_mode="HE")` keeps only the last three channels, and `"full"` keeps all of them without inverting H&E; tutorial 02 feeds H&E uninverted. A joint embedding is `"full"` with H&E already inverted.
- `Eva/mae.py` reads `marker_embeddings/GenePT_embedding.pkl` relative to the working directory, and the checkpoint is loaded with `weights_only=False` (check the sha256 first). `Eva/` and `utils/` have no `__init__.py` and `Eva/layers.py` imports a top-level `utils`, so the code needs its repository root on `sys.path`.

## Why the code looks like this
To write with the implementation.

## Done when
- [ ] The contract test passes: `pytest src/usbb/methods/eva`.
- [ ] The cluster test ran with the pinned weights, and its `RUN.json` summary is in the PR.
- [ ] The card's `heldout` entries are checked against proposal §3.3 (column "Eva"), with the Enable Medicine answer on Badoual recorded.
- [ ] On the cluster, the joint embedding equals Eva's own `forward_encoder` at the pinned commit on the same inverted patches (768 values each).
- [ ] `check` refuses a panel with a marker missing from Eva's table, naming each one; the card's `intended_usage` states how many of our markers map (Causer 11, Haist 53).
