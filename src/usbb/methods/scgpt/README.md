# scGPT (`scgpt`)

Verdict B. Tier T1. Issue: to open.

## What it is
A transformer pretrained on single-cell RNA; the `whole-human` checkpoint saw 33 million normal human cells. `scgpt.tasks.embed_data` takes one expression row per unit and returns its `<cls>` token, L2-normalised, as float32 (`scgpt/tasks/cell_emb.py`). Paper: [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02201-0); code: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT).

## Weights
- Google Drive folders, not Hugging Face (README model zoo; "Publish to huggingface model hub" is still unchecked). The README recommends `whole-human` ([folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)); `embed_data` reads `best_model.pt`, `args.json` and `vocab.json` from it. Not gated.
- Drive has no revision, so the card pins the folder id and the sha256 of those three files. The plan's `usbb.weights` fetches from Hugging Face only, so how a Drive checkpoint is fetched is open.
- Code MIT (repo `LICENSE`); no licence is stated for the checkpoints (README; working note, 16 Nov 2026): to verify before sharing embeddings.
- The zero-shot tutorials' README links `scGPT_CP`, which is further trained with Tabula Sapiens cell-type labels, to the same folder id as `whole-human`. Check the folder content before pinning.

## Inputs
- RNA (`rna`) on spots, `value_scale: raw_counts`. In schema v1 only spots carry RNA; images and protein are refused.
- `embed_data` reads `adata.X`, does not normalise, and bins each unit's non-zero values into 51 quantile bins (`data_collator.py`, `preprocess.py::binning`). So counts go in as they are; library-size scaling or log1p would barely change the bins (our reading).
- Vocabulary: genes match `vocab.json` by name through `gene_col`. Our `var_names` are provider ids, so pass `gene_col="gene_symbol"`. Unmatched genes are dropped and logged as "match x/y genes"; report that count. Two rows with the same symbol both get the same token (our reading of `cell_emb.py`), so resolve duplicates first.
- Fit mode for the card: frozen. A fine-tuned arm or another checkpoint (pan-cancer, 5.7M cancer cells) is a new method id.

## Pitfalls on our data
- Spots are out of distribution: scGPT saw single cells only, never Visium spots, and a spot holds several cells (working note, 16 Nov 2026).
- At most 1,200 genes per unit (`max_length=1200`). A unit with more matched genes is subsampled with `torch.randperm`, and ties in binning are broken with `np.random` (`data_collator.py`, `preprocess.py`). Seed torch and numpy from `self.seed`.
- Pretrained on normal tissue; our sections are head and neck tumours.
- Environment: `embed_data` defaults to `use_fast_transformer=True` (flash-attn); the README says the weights also load without flash-attn, so set it to False unless the env has it. The README asks for flash-attn<1.0.5 with CUDA 11.7, but `pyproject.toml` on `main` has a `fa2` extra with flash-attn>=2.8.0, plus scvi-tools<1.0 and cell-gears<0.0.3: to verify, likely its own env group. PyPI `scgpt` is 0.2.4 (31 Mar 2025); `main` at `cebd6fae655b9c585a4807daa3ac31bb764f06b4` (27 Apr 2026) says 0.2.5. Pin one.

## Why the code looks like this
To write with the implementation.

## Done when
- [ ] The contract test passes: `pytest src/usbb/methods/scgpt`.
- [ ] The cluster test ran with the pinned checkpoint, and its `RUN.json` summary is in the PR.
- [ ] The card's `heldout` entries are checked against proposal §3.3 (column "RNA FMs").
- [ ] The PR reports the matched gene count on the tonsil spots.
- [ ] On the cluster, embeddings equal `scgpt.tasks.embed_data` on the same spots and seed, and two runs with the same seed are identical.
