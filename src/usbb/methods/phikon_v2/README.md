# Phikon-v2 (`phikon_v2`)

Verdict B. Tier v0. Issue: to open.

The worked example: copy this folder to add an H&E patch encoder.

## What it is
Owkin's ViT-L/16 trained with DINOv2 on PANCAN-XL, 456M H&E tiles of 224 px at 20x from about 60,000 public slides (TCGA, CPTAC, GTEx and others) ([model card](https://huggingface.co/owkin/phikon-v2), [paper](https://arxiv.org/abs/2409.09173)). We use it frozen: one CLS token of 1024 values per patch. Official fine-tuning exists as a LoRA notebook (working note, 16 Nov 2026).

## Weights
`owkin/phikon-v2`, not gated, pinned in `card.yaml` to a revision and the sha256 of the two files `load` reads (`config.json`, `model.safetensors`); the weights file is unchanged since its upload on 17 May 2024. Owkin non-commercial licence (`LICENSE.pdf` in the repository); whether embeddings derived from it may be shared is to verify.

## Inputs
H&E patches of 224 px at 0.5 µm/px (20x), uint8 RGB, centred on spots or pseudospots and cut by the data processor. A patch covers 112 µm, about twice a Visium spot's 55 µm diameter, so the embedding also describes the tissue around the spot.

## Pitfalls on our data
- Cells are refused: a 112 µm patch holds many cells.
- H&E coarser than 0.5 µm/px is refused by the data processor: Causer's 3.28 µm/px image (proposal §3.2) and the tonsil hires PNG; tonsil needs the full-resolution scan. The Haist and Simkin pixel sizes are to verify (working note, §8).
- The tissue type was seen in pretraining: TCGA head and neck slides (proposal §3.3) and CPTAC-HNSCC (model card references). Our patients were not.
- Badoual's H&E is stained after PhenoCycler cycling and may look unlike routine H&E (proposal §3.1).

## Why the code looks like this
- `PatchEncoder` does the batching, device, eval mode and freeze check, so `method.py` holds only `load` and `embed`.
- `Dinov2Model` from transformers, never remote code: what runs is the code locked in the `torch` environment.
- Our own scaling and ImageNet normalisation instead of the official processor, which would silently resize a patch of the wrong size; the runner refuses such a bundle before `load`. On 224 px patches the processor's resize and crop do nothing, so both give the same pixels.
- Weights pinned by revision and sha256, fetched before the job and read from `--model`; a job never downloads.
- fp32, the dtype of the checkpoint and of the model card's example, so outputs do not depend on mixed precision.
- The CLS token after the final layer norm, as in the model card's feature extraction example.
- A fine-tuned arm is a new method id with its own card, never a parameter of this one.

## Done when
- [x] `pytest src/usbb/methods/phikon_v2` passes on CPU in the `torch` environment (tiny random `Dinov2Model`).
- [ ] `pytest -m "slow and gpu" src/usbb/methods/phikon_v2` passes on a GPU node: our CLS tokens equal the official processor and model at the pinned revision.
- [ ] Tonsil spots embedded from the full-resolution H&E; a rerun is a cache hit and gives bit-identical embeddings.
