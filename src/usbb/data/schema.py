"""SpatialData store schema v1: constants, element constructors and the validator."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from scipy import sparse
from spatialdata.models import ShapesModel, TableModel, get_channel_names
from spatialdata.transformations import Identity, get_transformation
from xarray import DataArray

SCHEMA_VERSION = 1
SPOT_RADIUS_UM = 27.5
UNIT_ELEMENTS = {"spot": "spots", "pseudospot": "pseudospots", "cell": "cells"}
ELEMENT_NAMES = {
    "images": {"he", "protein"},
    "labels": {"cells"},
    "points": set(),
    "shapes": {"spots", "pseudospots", "annotations", "cores"},
}
ATTRS_KEYS = (
    "schema_version",
    "dataset_key",
    "store_id",
    "sections",
    "tissue",
    "fixation",
    "licence",
    "release_date",
    "pairing",
    "registration",
    "provenance",
)
VALUE_SCALES = {"raw_counts", "raw_intensity", "background_subtracted", "scaled", "arcsinh"}
OBS_COLUMNS = ("section_id", "sample_id", "patient_id")
PROTEIN_VAR_COLUMNS = ("protein_name", "is_control", "mapping_source")
PAIRINGS = ["adjacent_section", "same_section"]
REGISTRATION_KEYS = ("moving", "reference", "kind", "baked", "method", "software", "qc")
REGISTRATION_KINDS = ["affine", "identity", "nonrigid"]
UM_PER_PX_RANGE = (0.05, 20.0)


class SchemaError(ValueError):
    """A store breaks schema v1; the message lists every violation."""


def circles(xy, ids, radius_um: float):
    """Circle shapes in µm (identity transform to `global`), indexed by `ids`."""
    index = pd.Index([str(i) for i in ids])
    radius = np.full(len(index), float(radius_um))
    return ShapesModel.parse(
        np.asarray(xy, float),
        geometry=0,
        radius=radius,
        index=index,
        transformations={"global": Identity()},
    )


def table(name, ids, obs, var, values, spatial, value_scale) -> AnnData:
    """Table `<unit>_<modality>` annotating its unit element; obs holds section, sample, patient."""
    unit, modality = name.split("_", 1)
    element = UNIT_ELEMENTS[unit]
    obs = obs.set_axis([str(i) for i in ids]).assign(
        region=pd.Categorical([element] * len(ids)), instance_id=list(ids)
    )
    adata = AnnData(
        obs=obs,
        var=var,
        layers={"counts" if value_scale == "raw_counts" else "intensity": values},
        obsm={"spatial": np.asarray(spatial, float)},
        uns={"modality": modality, "value_scale": value_scale},
    )
    return TableModel.parse(adata, region=element, region_key="region", instance_key="instance_id")


def full_resolution(raster) -> DataArray:
    """The finest level of an image or labels element."""
    return raster if isinstance(raster, DataArray) else sd.get_pyramid_levels(raster, n=0)


def global_affine(element) -> np.ndarray:
    """The 3 x 3 (x, y) affine matrix from an element to the global frame."""
    return get_transformation(element, "global").to_affine_matrix(("x", "y"), ("x", "y"))


def um_per_px(raster) -> np.ndarray:
    """Pixel size (x, y) in µm of a raster element in the global frame."""
    matrix = global_affine(raster)
    return np.hypot(matrix[0, :2], matrix[1, :2])


def validate_store(path) -> None:
    """Read a zarr store and validate it."""
    validate(sd.read_zarr(path))


def validate(sdata) -> None:
    """Raise one SchemaError listing every way `sdata` breaks schema v1."""
    attrs = sdata.attrs.get("usbb") or {}
    problems = _attrs_problems(sdata, attrs) + _element_problems(sdata)
    for name, adata in sdata.tables.items():
        problems += [f"table {name}: {p}" for p in _table_problems(sdata, attrs, name, adata)]
    if problems:
        raise SchemaError("store breaks usbb schema v1:\n- " + "\n- ".join(problems))


def _attrs_problems(sdata, attrs) -> list:
    problems = [f'attrs["usbb"] lacks {key}' for key in ATTRS_KEYS if key not in attrs]
    if attrs.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        problems.append(f"schema_version is {attrs['schema_version']!r}, not {SCHEMA_VERSION}")
    if not isinstance(attrs.get("sections", [None]), list) or attrs.get("sections") == []:
        problems.append("sections must be a non-empty list")
    pairing = attrs.get("pairing") or {}
    modalities = {t.uns.get("modality") for t in sdata.tables.values()} | set(sdata.images)
    for modality in sorted(m for m in modalities if m):
        entry = pairing.get(modality) or {}
        if entry.get("kind") not in PAIRINGS or not entry.get("source"):
            problems.append(f"pairing[{modality!r}] needs kind (one of {PAIRINGS}) and source")
    elements = {name for _, name, _ in sdata.gen_spatial_elements()}
    registration = attrs.get("registration", [])
    if not isinstance(registration, list):
        return problems + ["registration must be a list"]
    for i, entry in enumerate(registration):
        where = f"registration[{i}]"
        missing = [key for key in REGISTRATION_KEYS if key not in entry]
        if missing:
            problems.append(f"{where} lacks {', '.join(missing)}")
            continue
        if entry["kind"] not in REGISTRATION_KINDS:
            problems.append(f"{where} kind {entry['kind']!r} is not in {REGISTRATION_KINDS}")
        if entry["kind"] == "nonrigid" and entry["baked"] is not True:
            problems.append(f"{where} is non-rigid and must be baked into the moving element")
        problems += [
            f"{where} {role} {entry[role]!r} is not an element"
            for role in ("moving", "reference")
            if entry[role] not in elements
        ]
        qc = entry["qc"] or {}
        missing_qc = qc.get("status") == "missing" and qc.get("reason")
        if not isinstance(qc.get("residual_um"), (int, float)) and not missing_qc:
            problems.append(f"{where} qc needs residual_um or {{status: missing, reason}}")
    return problems


def _element_problems(sdata) -> list:
    problems = []
    for kind, allowed in ELEMENT_NAMES.items():
        for name, element in getattr(sdata, kind).items():
            where = f"{kind} {name}"
            if name not in allowed:
                problems.append(f"{where} is not a schema element (allowed: {sorted(allowed)})")
            if "global" not in get_transformation(element, get_all=True):
                problems.append(f"{where} has no transformation to global")
            elif kind in ("images", "labels"):
                x, y = um_per_px(element)
                if abs(x - y) > 0.01 * x or not UM_PER_PX_RANGE[0] <= x <= UM_PER_PX_RANGE[1]:
                    problems.append(
                        f"{where} pixels ({x}, {y}) µm: not square in {UM_PER_PX_RANGE}"
                    )
            elif kind == "shapes" and not np.allclose(global_affine(element), np.eye(3)):
                problems.append(f"{where} must be in µm with an identity transformation")
    he = sdata.images.get("he")
    if he is not None:
        he = full_resolution(he)
        if he.dims != ("c", "y", "x") or he.dtype != np.uint8 or he.shape[0] != 3:
            problems.append("images he must be cyx uint8 RGB")
    protein = sdata.images.get("protein")
    if protein is not None and not all(isinstance(c, str) for c in get_channel_names(protein)):
        problems.append("images protein needs channel names")
    spots = sdata.shapes.get("spots")
    if spots is not None and not np.allclose(spots.get("radius", np.nan), SPOT_RADIUS_UM):
        problems.append(f"shapes spots must be circles of radius {SPOT_RADIUS_UM} µm")
    annotations = sdata.shapes.get("annotations")
    if annotations is not None and not {"label", "annotator"} <= set(annotations.columns):
        problems.append("shapes annotations needs columns label and annotator")
    return problems


def _table_problems(sdata, attrs, name, adata) -> list:
    unit, _, modality = name.partition("_")
    element = UNIT_ELEMENTS.get(unit)
    if element is None or not modality:
        return [f"name is not <unit>_<modality> with unit in {sorted(UNIT_ELEMENTS)}"]
    problems = []
    keys = adata.uns.get("spatialdata_attrs", {})
    expected = {"region": element, "region_key": "region", "instance_key": "instance_id"}
    if any(keys.get(key) != value for key, value in expected.items()):
        problems.append(
            f'must annotate {element} with region_key "region", instance_key "instance_id"'
        )
    if adata.uns.get("modality") != modality:
        problems.append(f'uns["modality"] must be {modality!r}')
    value_scale = adata.uns.get("value_scale")
    if value_scale not in VALUE_SCALES:
        problems.append(f'uns["value_scale"] {value_scale!r} is not in {sorted(VALUE_SCALES)}')
    if adata.X is not None:
        problems.append("X must be empty; values go in layers")
    layer = "counts" if value_scale == "raw_counts" else "intensity"
    if layer not in adata.layers:
        problems.append(f'layers["{layer}"] is missing')
    else:
        values = adata.layers[layer]
        values = values.data if sparse.issparse(values) else np.asarray(values)
        if not np.isfinite(values).all():
            problems.append(f'layers["{layer}"] has non-finite values')
        elif layer == "counts" and ((values < 0).any() or (values != np.round(values)).any()):
            problems.append('layers["counts"] must hold whole non-negative counts')
    columns = OBS_COLUMNS + (("core_id",) if "cores" in sdata.shapes else ())
    problems += [f"obs lacks {c}" for c in columns if c not in adata.obs]
    sections = set(adata.obs.get("section_id", [])) - set(attrs.get("sections", []))
    if sections:
        problems.append(f"obs section_id values {sorted(sections)} are not in attrs sections")
    var_columns = ("gene_symbol",) + (PROTEIN_VAR_COLUMNS if modality == "protein" else ())
    problems += [f"var lacks {c}" for c in var_columns if c not in adata.var]
    if "is_control" in adata.var and adata.var["is_control"].dtype != bool:
        problems.append("var is_control must be boolean")
    return problems + _location_problems(sdata, element, adata)


def _location_problems(sdata, element, adata) -> list:
    if element not in sdata.shapes and element not in sdata.labels:
        return [f"annotates {element}, which is not in the store"]
    problems = []
    ids = adata.obs["instance_id"]
    if not ids.is_unique:
        problems.append("instance_id values are not unique")
    if element in sdata.shapes:
        known = set(sdata.shapes[element].index)
    else:
        known = set(da.unique(full_resolution(sdata.labels[element]).data).compute().tolist())
    if not set(ids) <= known:
        problems.append(f"{len(set(ids) - known)} instance_id values are not in {element}")
    spatial = np.asarray(adata.obsm.get("spatial", np.empty((0, 0))), float)
    if spatial.shape != (adata.n_obs, 2) or not np.isfinite(spatial).all():
        return problems + ['obsm["spatial"] must be finite (n, 2) µm coordinates']
    extent = sd.get_extent(sdata[element])
    for axis, column in (("x", 0), ("y", 1)):
        low, high = extent[axis]
        if (spatial[:, column] < low - 1e-6).any() or (spatial[:, column] > high + 1e-6).any():
            problems.append(f'obsm["spatial"] {axis} lies outside {element}')
    if "pseudospot_id" in adata.obs:
        pseudospots = sdata.shapes.get("pseudospots")
        known = set() if pseudospots is None else set(pseudospots.index)
        unknown = set(adata.obs["pseudospot_id"]) - known - {""}
        if unknown:
            problems.append(f"{len(unknown)} pseudospot_id values are not in pseudospots")
    return problems
