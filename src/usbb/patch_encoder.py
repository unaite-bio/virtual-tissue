"""Base class of the frozen H&E patch encoders; the only usbb core module that imports torch."""

from __future__ import annotations

import numpy as np
import torch

from usbb.method import Method

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalise(patches, mean, std):
    """uint8 patches (n, channels, size, size) scaled to [0, 1], then standardised per channel."""
    mean = torch.tensor(mean, device=patches.device).view(-1, 1, 1)
    std = torch.tensor(std, device=patches.device).view(-1, 1, 1)
    return (patches.float() / 255 - mean) / std


class PatchEncoder(Method):
    """An encoder of the card's one image input: load() sets self.model, embed() encodes a batch."""

    batch_size = 64

    def embed(self, patches):
        """Embeddings (n, d) of uint8 patches (n, channels, size, size) on the model's device."""
        raise NotImplementedError(f"{type(self).__name__} has no embed()")

    def transform(self, inp) -> dict:
        """Embed the patches batch by batch, on a GPU when there is one, in eval mode."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device).eval().requires_grad_(False)
        (modality,) = self.card.inputs
        (output,) = self.card.outputs
        embeddings = []
        with torch.inference_mode():
            for _, batch in inp.patches(modality, self.batch_size):
                # Contiguous like official processors' tensors, so the same GPU kernels run.
                patches = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).contiguous()
                embeddings.append(self.embed(patches).float().cpu().numpy())
        return {output.latent: np.concatenate(embeddings)}

    def state(self) -> dict:
        """Parameters and buffers of self.model, which a frozen run must leave unchanged."""
        return {name: value.cpu().numpy() for name, value in self.model.state_dict().items()}
