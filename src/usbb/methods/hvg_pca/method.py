"""Principal components of the most variable log-normalised genes, fitted on the train bundle."""

from __future__ import annotations

import numpy as np

from usbb.method import Method


def log_normalise(counts) -> np.ndarray:
    """log1p of counts per 10,000 counts of the unit."""
    return np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)


class HvgPca(Method):
    def check(self, inp):
        empty = int((inp.matrix("rna").sum(axis=1) == 0).sum())
        return [f"{empty} units have no RNA counts, so they cannot be normalised"] if empty else []

    def fit(self, inp):
        values = log_normalise(inp.matrix("rna"))
        self.genes = np.argsort(-values.var(axis=0), kind="stable")[: self.params["n_top_genes"]]
        values = values[:, self.genes]
        self.mean = values.mean(axis=0)
        _, _, components = np.linalg.svd(values - self.mean, full_matrices=False)
        self.components = components[: self.params["n_components"]]

    def transform(self, inp):
        values = log_normalise(inp.matrix("rna"))[:, self.genes]
        return {"pca": (values - self.mean) @ self.components.T}
