"""Negative control: features drawn from the seed and the unit id, never from the data."""

from __future__ import annotations

import hashlib

import numpy as np

from usbb.method import Method


def unit_features(seed, unit_id, n_features) -> np.ndarray:
    """Standard normal values from a generator seeded by (seed, unit id)."""
    digest = hashlib.sha256(f"{seed}:{unit_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).standard_normal(n_features)


class RandomFeatures(Method):
    def transform(self, inp):
        n_features = self.params["n_features"]
        return {"random": np.stack([unit_features(self.seed, u, n_features) for u in inp.unit_ids])}
