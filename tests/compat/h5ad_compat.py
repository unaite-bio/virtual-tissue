"""Write the pinned h5ad layouts with usbb.io.write_h5ad, or check files written by another stack.

Usage: PYTHONPATH=src python tests/compat/h5ad_compat.py write|check <dir>
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from usbb.io import write_h5ad

IDS = ["tonsil:CGGTTAGGCCTGGACG-1", "tonsil:AACCTTGGAACCTTGG-1", "tonsil:TTGGCCAATTGGCCAA-1"]


def layouts() -> dict:
    """The input files of a bundle (rna, he) and a method output, as in the pinned layout."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "store_id": ["tonsil"] * 3,
            "section_id": ["s1", "s1", "s2"],
            "sample_id": ["a", "a", "b"],
            "patient_id": ["p1", "p2", "p3"],
        },
        index=IDS,
    )
    spatial = rng.random((3, 2)) * 1000.0
    rna = anndata.AnnData(
        obs=obs.copy(),
        var=pd.DataFrame(
            {"gene_symbol": ["CD3E", "MS4A1"]}, index=["ENSG00000198851", "ENSG00000156738"]
        ),
        layers={"counts": sp.csr_matrix(rng.poisson(2.0, size=(3, 2)).astype(np.int32))},
        obsm={"spatial": spatial},
        uns={
            "dataset_id": "tenx_tonsil",
            "modality": "rna",
            "unit_type": "spot",
            "value_scale": "raw_counts",
            "normalization_id": "raw_counts",
        },
    )
    he = anndata.AnnData(
        obs=obs.copy(),
        obsm={"spatial": spatial, "patches": rng.integers(0, 256, (3, 4, 4, 3), dtype=np.uint8)},
        uns={
            "dataset_id": "tenx_tonsil",
            "modality": "he",
            "unit_type": "spot",
            "value_scale": "raw",
            "normalization_id": "raw",
            "patch": {"um_per_px": 0.5, "size_px": 4},
        },
    )
    output = anndata.AnnData(
        obs=obs.copy(),
        obsm={"joint": rng.random((3, 5)).astype(np.float32)},
        uns={
            "dataset_id": "tenx_tonsil",
            "method_id": "spatialglue",
            "normalization_id": "raw_counts",
            "unit_type": "spot",
            "run_key": "0" * 64,
            "informed_by": {"joint": ["rna", "protein"]},
        },
    )
    return {"rna": rna, "he": he, "output": output}


def plain(value):
    """Python values for comparison: anndata reads lists back as arrays and ints as numpy scalars."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def differences(name: str, got: anndata.AnnData, want: anndata.AnnData) -> list:
    """Every way `got` differs from `want`, as messages."""
    found = []
    if got.X is not None:
        found.append("X is not empty")
    for axis in ("obs", "var"):
        got_df, want_df = getattr(got, axis), getattr(want, axis)
        if list(got_df.index) != list(want_df.index):
            found.append(f"{axis}_names differ")
        for column in want_df.columns:
            if column not in got_df or list(got_df[column].astype(str)) != list(want_df[column]):
                found.append(f"{axis}[{column!r}] differs")
    for slot in ("obsm", "layers"):
        for key, want_array in getattr(want, slot).items():
            got_array = getattr(got, slot).get(key)
            if got_array is None or got_array.dtype != want_array.dtype:
                found.append(f"{slot}[{key!r}] missing or dtype differs")
            elif not np.array_equal(dense(got_array), dense(want_array)):
                found.append(f"{slot}[{key!r}] values differ")
    if plain(got.uns) != plain(want.uns):
        found.append(f"uns differs: {plain(got.uns)!r}")
    return [f"{name}.h5ad: {message}" for message in found]


def write(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, adata in layouts().items():
        write_h5ad(adata, folder / (name + ".h5ad"))


def check(folder: Path) -> None:
    wanted = layouts()
    problems = []
    for name, want in wanted.items():
        problems += differences(name, anndata.read_h5ad(folder / (name + ".h5ad")), want)
    with h5py.File(folder / "he.h5ad", "r") as f:  # methods read patches lazily, without anndata
        if not np.array_equal(f["obsm/patches"][1:3], wanted["he"].obsm["patches"][1:3]):
            problems.append("he.h5ad: obsm/patches differs when read with h5py")
    if problems:
        sys.exit("h5ad compatibility problems:\n" + "\n".join(problems))


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    {"write": write, "check": check}[command](folder)
    print(f"{command} ok with anndata {version('anndata')}, Python {sys.version.split()[0]}")
