"""Early fusion: every upstream latent side by side, unscaled."""

from __future__ import annotations

import numpy as np

from usbb.method import Method


class Concat(Method):
    def transform(self, inp):
        blocks = [
            inp.modalities[name].obsm[latent]
            for name in sorted(inp.modalities)
            for latent in sorted(inp.modalities[name].obsm)
            if latent != "spatial"
        ]
        return {"concat": np.concatenate(blocks, axis=1)}
