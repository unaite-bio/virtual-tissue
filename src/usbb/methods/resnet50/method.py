"""ImageNet ResNet-50 (torchvision weights, through timm): the baseline H&E patch encoder."""

from __future__ import annotations

from pathlib import Path

import timm
from safetensors.torch import load_file

from usbb.patch_encoder import IMAGENET_MEAN, IMAGENET_STD, PatchEncoder, normalise


class ResNet50(PatchEncoder):
    def load(self):
        self.model = timm.create_model("resnet50")
        self.model.load_state_dict(load_file(Path(self.model_dir) / "model.safetensors"))
        self.model.reset_classifier(0)  # forward then returns the pooled features, not 1000 classes

    def embed(self, patches):
        return self.model(normalise(patches, IMAGENET_MEAN, IMAGENET_STD))
