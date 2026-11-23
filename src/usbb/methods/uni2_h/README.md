# UNI2-h (`uni2_h`)

Verdict B. Tier T1. Issue: to open.

## What it is
A ViT-H/14 with 8 register tokens, pretrained with DINOv2 on more than 200M tiles from more than 350,000 H&E and IHC slides of Mass General Brigham. It takes one 224 px RGB patch and returns its class token, 1536 values ([model card](https://huggingface.co/MahmoodLab/UNI2-h)).

## Weights
- [`MahmoodLab/UNI2-h`](https://huggingface.co/MahmoodLab/UNI2-h), gated: request access on that page; the account's primary email must be institutional (personal domains are denied) and each person registers separately. Licence CC BY-NC-ND 4.0, non-commercial, no redistribution of the weights (model card).
- On 23 Nov 2026 the Hub lists `main` = `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` (last change 16 Jan 2025) and one weights file, `pytorch_model.bin`, sha256 `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e`. Pin both in the card and check the sha256 after download.
- The architecture is built with timm >= 0.9.8 (model card); no remote code is needed.

## Inputs
- H&E patches (`he`) for spots and pseudospots; cells refused, as in `phikon_v2`.
- 224 px, ImageNet mean and std (model card). The card does not give the pretraining µm per pixel; the authors' released UNI2-h features use 256 px tiles at 20x ([features card](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)), and the working note (16 Nov 2026) takes about 0.5 µm/px. Exact value: to verify. 224 px at 0.5 µm/px, as in `phikon_v2`, would share the same patches.

## Pitfalls on our data
- Causer's H&E at 3.28 µm/px is 6.6 times coarser than 0.5 µm/px: refuse it; until the full-resolution image arrives it can only be a low-resolution stress test. The pixel size of the Haist and Simkin H&E is not confirmed yet (working note, 16 Nov 2026).
- H&E stained after PhenoCycler or CODEX cycling (Badoual, Haist) may differ from routine H&E (proposal §3.1, §3.2).
- Pretraining slides come from Mass General Brigham (model card); §3.3 marks UNI2 N for every candidate dataset with H&E. Whether they include head and neck tissue is not stated: to verify.
- Access is per person: whoever runs the weight fetch on the HPC needs approved access, so request it early (working note, 16 Nov 2026). The terms count datasets created from the model as derivatives, so whether our embeddings can be shared is still an open licence point.

## Why the code looks like this
To write with the implementation.

## Done when
- [ ] The contract test passes: `pytest src/usbb/methods/uni2_h`.
- [ ] The cluster test ran with the pinned weights, and its `RUN.json` summary is in the PR.
- [ ] The card's `heldout` entries are checked against proposal §3.3 (column "H&E FMs").
- [ ] On the cluster, embeddings equal the model card's timm path on the same 224 px patches (1536 values each).
- [ ] The pretraining magnification is confirmed (paper or authors) and matches the card's `um_per_px`; Causer's 3.28 µm/px patches are refused.
