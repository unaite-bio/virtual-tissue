# ResNet-50, ImageNet (`resnet50`)

Verdict A. Tier v0. Issue: to open.

## What it is
The ImageNet baseline of the H&E encoders (proposal §4.1): ResNet-50 trained on ImageNet-1k, 1.28M natural photos (working note, 16 Nov 2026), with torchvision's original weights as ported to timm ([model card](https://huggingface.co/timm/resnet50.tv_in1k), [paper](https://arxiv.org/abs/1512.03385)). Frozen; the embedding is the average-pooled output of the last stage, 2048 values per patch.

## Weights
`timm/resnet50.tv_in1k`, not gated, BSD-3-Clause, pinned in `card.yaml` to a revision and the sha256 of `model.safetensors`, the only file `load` reads; unchanged since its upload on 5 April 2023.

## Inputs
The same patches as Phikon-v2: 224 px at 0.5 µm/px, centred on spots or pseudospots. ImageNet photos have no pixel size, so no resolution is right for this model by construction. We chose 0.5 µm/px and 224 px so that it sees the same 112 µm of tissue as Phikon-v2 and both read one patch file; 224 px is the input size in the model's config.

## Pitfalls on our data
- Same geometry limits as Phikon-v2: cells, and H&E coarser than 0.5 µm/px such as Causer's 3.28 µm/px image, are refused.
- Its features were never trained on tissue, so stain and scanner differences between datasets may weigh more in them than in the pathology encoders (to verify with the evaluation).

## Why the code looks like this
- timm builds `resnet50` by name and `load_state_dict` reads the pinned file strictly, so a wrong or partial file fails instead of leaving random weights.
- `reset_classifier(0)` drops the 1000 ImageNet classes; the forward pass then returns the pooled features.
- No resize to 256 px and centre crop, unlike timm's eval transform (`crop_pct` 0.875): the patch already has the tissue extent we want, and the crop would cut its border.
- ImageNet mean and std, the values in the model's config at the pinned revision.
- `PatchEncoder` sets eval mode, which matters here: BatchNorm in train mode would update its running statistics, and the freeze check covers them too.

## Done when
- [x] `pytest src/usbb/methods/resnet50` passes on CPU in the `torch` environment (random ResNet-50 weights).
- [ ] `pytest -m "slow and gpu" src/usbb/methods/resnet50` passes on a GPU node: our features equal timm's model at the pinned revision with `crop_pct` 1.0.
- [ ] Tonsil spots embedded from the same patch file as Phikon-v2.
