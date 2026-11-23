"""Each unit's protein values, arcsinh-transformed, as its embedding."""

from __future__ import annotations

import numpy as np

from usbb.method import Method


class MeanIntensity(Method):
    def transform(self, inp):
        return {"intensity": np.arcsinh(inp.matrix("protein") / self.params["cofactor"])}
