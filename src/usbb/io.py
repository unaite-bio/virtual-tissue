"""Reading and writing the files exchanged between steps (specs in usbb/api/file_*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import anndata
import h5py
import numpy as np
import pandas as pd
import yaml

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11, as in the SpatialGlue environment
    from anndata.experimental import read_elem

API = Path(__file__).parent / "api"
NUMBER_KINDS = {"integer": "iu", "double": "f", "boolean": "b"}


def write_h5ad(adata: anndata.AnnData, path) -> None:
    """Write an h5ad file that the oldest supported anndata (0.8) can read."""
    settings = getattr(anndata, "settings", None)  # anndata < 0.11 has no settings
    if settings is None:
        adata.write_h5ad(path)
        return
    with settings.override(
        allow_write_nullable_strings=False
    ):  # anndata 0.8 cannot read nullable strings
        adata.write_h5ad(path)


@dataclass
class MethodInput:
    """A bundle: one AnnData per file, all with the same units in the same order."""

    path: Path
    unit_ids: list[str]
    unit_type: str
    obs: pd.DataFrame
    modalities: dict[str, anndata.AnnData]

    def matrix(self, modality) -> np.ndarray:
        """Raw values of a matrix modality (layer counts or intensity) as a dense array."""
        layers = self.modalities[modality].layers
        values = layers["counts"] if "counts" in layers else layers["intensity"]
        return values.toarray() if hasattr(values, "toarray") else np.asarray(values)

    def patches(self, modality, batch_size) -> Iterator[tuple[list[str], np.ndarray]]:
        """Yield (unit ids, uint8 patches) batches in unit order, read from disk batch by batch."""
        with h5py.File(self.path / f"{modality}.h5ad", "r") as f:
            patches = f["obsm/patches"]
            for start in range(0, len(self.unit_ids), batch_size):
                stop = start + batch_size
                yield self.unit_ids[start:stop], patches[start:stop]


def _read_without_patches(path) -> anndata.AnnData:
    # Patches can exceed memory, so obsm/patches is left on disk for MethodInput.patches.
    with h5py.File(path, "r") as f:
        return anndata.AnnData(
            obs=read_elem(f["obs"]),
            var=read_elem(f["var"]),
            obsm={key: read_elem(value) for key, value in f["obsm"].items() if key != "patches"},
            layers=read_elem(f["layers"]),
            uns=read_elem(f["uns"]),
        )


def read_input(bundle_dir) -> MethodInput:
    """Read every <name>.h5ad of a bundle folder; all must hold the same units in the same order."""
    path = Path(bundle_dir)
    modalities = {file.stem: _read_without_patches(file) for file in sorted(path.glob("*.h5ad"))}
    if not modalities:
        raise ValueError(f"{path} holds no h5ad file")
    first_name, first = next(iter(modalities.items()))
    misaligned = [
        name
        for name, adata in modalities.items()
        if not adata.obs_names.equals(first.obs_names)
        or adata.uns["unit_type"] != first.uns["unit_type"]
    ]
    if misaligned:
        raise ValueError(
            f"{path}: unit ids, order or unit_type of {misaligned} differ from {first_name}.h5ad"
        )
    return MethodInput(path, list(first.obs_names), first.uns["unit_type"], first.obs, modalities)


def write_output(path, inp, latents, uns) -> None:
    """Write a method output for the units of inp and check it against api/file_output.yaml."""
    write_h5ad(anndata.AnnData(obs=inp.obs.copy(), obsm=latents, uns=uns), path)
    check_file(path, "output")


def _has_type(node, type_) -> bool:
    """Whether an h5ad element holds values of an Open Problems file-format type."""
    if type_ == "object":
        return isinstance(node, h5py.Group)
    if isinstance(
        node, h5py.Group
    ):  # categorical, nullable and sparse encodings keep values in a child
        children = [node[key] for key in ("categories", "values", "data") if key in node]
        if not children:
            return False
        node = children[0]
    if type_ == "string":
        return h5py.check_string_dtype(node.dtype) is not None
    return node.dtype.kind in NUMBER_KINDS[type_]


def check_file(path, spec) -> None:
    """Raise one ValueError listing every way the h5ad at path breaks api/file_<spec>.yaml."""
    slots = yaml.safe_load((API / f"file_{spec}.yaml").read_text(encoding="utf-8"))["info"][
        "format"
    ]
    problems = []
    with h5py.File(path, "r") as f:
        for slot in ("obs", "var", "obsm", "layers", "uns"):
            for entry in slots.get(slot, []):
                key = f"{slot}/{entry['name']}"
                if key not in f:
                    if entry["required"]:
                        problems.append(f"missing {key}")
                elif not _has_type(f[key], entry["type"]):
                    problems.append(f"{key} is not of type {entry['type']}")
    if problems:
        raise ValueError(f"{path} breaks file_{spec}.yaml: " + "; ".join(problems))
