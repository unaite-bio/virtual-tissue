"""Phikon-v2, Owkin's ViT-L/16 DINOv2 pathology encoder, used frozen on 224 px H&E patches."""

from __future__ import annotations

import torch
from transformers import Dinov2Model

from usbb.patch_encoder import IMAGENET_MEAN, IMAGENET_STD, PatchEncoder, normalise


class PhikonV2(PatchEncoder):
    def load(self):
        self.model = Dinov2Model.from_pretrained(self.model_dir, dtype=torch.float32)

    def embed(self, patches):
        pixels = normalise(patches, IMAGENET_MEAN, IMAGENET_STD)
        return self.model(pixel_values=pixels).last_hidden_state[:, 0]  # CLS token
