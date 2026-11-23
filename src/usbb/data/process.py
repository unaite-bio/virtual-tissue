"""Data processor: stores, a method card and a split become train and test input bundles."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from PIL import Image

from usbb.data.schema import OBS_COLUMNS, full_resolution, global_affine, validate
from usbb.io import write_h5ad
from usbb.run import folder_hash as bundle_key  # the run key hashes bundles the same way

UPSAMPLING_TOLERANCE = 0.05


def prepare(store_paths, card, unit_type: str, split, out_root):
    """Write the input bundles of `card` and return (train_dir or None, test_dir).

    `split` is None (every unit is a test unit) or {"by": <obs column>, "test": [values]}.
    """
    if unit_type not in card.units:
        raise ValueError(
            f"{card.method_id} does not take {unit_type} units (card units: {card.units})"
        )
    stores = [sd.read_zarr(path) for path in store_paths]
    for sdata in stores:
        validate(sdata)
    datasets = {sdata.attrs["usbb"]["dataset_key"] for sdata in stores}
    if len(datasets) != 1:
        raise ValueError(f"stores come from several datasets: {sorted(datasets)}")
    files = {}
    for name, spec in card.inputs.items():
        if spec["kind"] == "embedding":  # upstream outputs are copied in by the launcher
            continue
        parts = [_modality_file(sdata, unit_type, name, spec) for sdata in stores]
        if any(not part.var_names.equals(parts[0].var_names) for part in parts):
            raise ValueError(f"stores have different {name} features")
        files[name] = ad.concat(parts, merge="same", uns_merge="same")
    if not files:
        raise ValueError(f"{card.method_id} has no matrix or image input to prepare")
    if split is None:
        return None, _write_bundle(files, out_root)
    test = next(iter(files.values())).obs[split["by"]].isin(split["test"]).to_numpy()
    if test.all() or not test.any():
        raise ValueError(f"split {split} leaves no {'train' if test.all() else 'test'} units")
    train_dir = _write_bundle(
        {name: adata[~test].copy() for name, adata in files.items()}, out_root
    )
    test_dir = _write_bundle({name: adata[test].copy() for name, adata in files.items()}, out_root)
    return train_dir, test_dir


def cut_patches(image, xy_um, um_per_px: float, size_px: int) -> np.ndarray:
    """uint8 patches (n, s, s, c) of side `size_px` at `um_per_px` centred on `xy_um`; 0 outside."""
    matrix = global_affine(image)
    if matrix[0, 1] or matrix[1, 0]:
        raise ValueError("patches need an image transformation without rotation or shear")
    source_um = matrix[0, 0]
    if source_um > um_per_px * (1 + UPSAMPLING_TOLERANCE):
        raise ValueError(
            f"the image is {source_um:.3f} µm/px, coarser than the {um_per_px} µm/px patches;"
            " upsampling is refused"
        )
    # Read once: one dask read per patch costs about 20 ms (90 s for the 4,194 tonsil spots).
    pixels = np.moveaxis(np.asarray(full_resolution(image).data), 0, -1)
    height, width, channels = pixels.shape
    half = size_px * um_per_px / source_um / 2
    patches = np.zeros((len(xy_um), size_px, size_px, channels), np.uint8)
    for i, (x, y) in enumerate((np.asarray(xy_um) - matrix[:2, 2]) / source_um):
        x0, y0 = int(np.floor(x - half)), int(np.floor(y - half))
        x1, y1 = int(np.ceil(x + half)), int(np.ceil(y + half))
        crop = np.zeros((y1 - y0, x1 - x0, channels), np.uint8)
        ys, xs = slice(max(y0, 0), min(y1, height)), slice(max(x0, 0), min(x1, width))
        if ys.start < ys.stop and xs.start < xs.stop:
            crop[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = pixels[ys, xs]
        box = (x - half - x0, y - half - y0, x + half - x0, y + half - y0)
        resized = Image.fromarray(crop).resize(
            (size_px, size_px), Image.Resampling.BILINEAR, box=box
        )
        patches[i] = np.asarray(resized)
    return patches


def _units(sdata, unit_type: str):
    """Ids, obs and µm centres of the units in every `<unit_type>_*` table, sorted by id."""
    store_id = sdata.attrs["usbb"]["store_id"]
    tables = [t for key, t in sorted(sdata.tables.items()) if key.split("_")[0] == unit_type]
    if not tables:
        raise ValueError(f"store {store_id} has no {unit_type} tables")
    ids = sorted(set.intersection(*(set(t.obs["instance_id"]) for t in tables)))
    rows = pd.Index(tables[0].obs["instance_id"]).get_indexer(ids)
    obs = tables[0].obs.iloc[rows][[c for c in OBS_COLUMNS + ("core_id",) if c in tables[0].obs]]
    obs.insert(0, "store_id", store_id)
    obs.index = [f"{store_id}:{i}" for i in ids]
    return ids, obs, tables[0].obsm["spatial"][rows]


def _modality_file(sdata, unit_type: str, name: str, spec) -> ad.AnnData:
    ids, obs, spatial = _units(sdata, unit_type)
    uns = {
        "dataset_id": sdata.attrs["usbb"]["dataset_key"],
        "modality": name,
        "unit_type": unit_type,
    }
    if spec["kind"] == "image":
        patches = cut_patches(sdata.images[name], spatial, spec["um_per_px"], spec["patch_px"])
        uns.update(value_scale="raw", normalization_id="raw")
        uns["patch"] = {"um_per_px": float(spec["um_per_px"]), "size_px": int(spec["patch_px"])}
        return ad.AnnData(obs=obs, obsm={"spatial": spatial, "patches": patches}, uns=uns)
    source = sdata.tables[f"{unit_type}_{name}"]
    value_scale = source.uns["value_scale"]
    layer = "counts" if value_scale == "raw_counts" else "intensity"
    rows = pd.Index(source.obs["instance_id"]).get_indexer(ids)
    uns.update(value_scale=value_scale, normalization_id=value_scale)
    layers = {layer: source.layers[layer][rows]}
    return ad.AnnData(
        obs=obs, var=source.var.copy(), layers=layers, obsm={"spatial": spatial}, uns=uns
    )


def _write_bundle(files: dict, out_root) -> Path:
    """Write one h5ad per modality to `<out_root>/bundles/<bundle_key[:16]>/`; return the folder."""
    bundles = Path(out_root) / "bundles"
    staging = bundles / f"tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, adata in files.items():
        write_h5ad(adata, staging / f"{name}.h5ad")
    final = bundles / bundle_key(staging)[:16]
    if final.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(final)
    return final
