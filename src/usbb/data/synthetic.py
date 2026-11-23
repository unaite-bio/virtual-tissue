"""A small valid store for tests: spots with RNA and protein, an H&E image, cells with protein."""

from __future__ import annotations

import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from scipy.spatial import cKDTree
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from usbb.data.schema import SCHEMA_VERSION, SPOT_RADIUS_UM, circles, table

IMAGE_PX = 600  # at 1 µm per pixel
PROTEINS = pd.DataFrame(
    {
        "gene_symbol": ["CD3E", "MS4A1", "PAX5", "KRT5", ""],
        "protein_name": ["CD3", "CD20", "PAX5", "PanCK", "mouse_IgG1k"],
        "is_control": [False, False, False, False, True],
        "mapping_source": ["synthetic"] * 4 + ["none"],
    },
    index=["CD3E", "MS4A1", "PAX5", "KRT5", "mouse_IgG1k"],
)


def synthetic_store(store_id: str = "synthetic_a", seed: int = 0) -> sd.SpatialData:
    """36 spots (20 genes, 5 proteins), a 600 x 600 px H&E at 1 µm/px, 150 cells, two patients."""
    rng = np.random.default_rng(seed)
    row, col = np.divmod(np.arange(36), 6)
    spot_xy = np.column_stack([40.0 + 100 * col + 50 * (row % 2), 40.0 + 50 * np.sqrt(3) * row])
    row, col = np.divmod(np.arange(150), 15)
    cell_xy = np.column_stack([20.0 + 40 * col, 20.0 + 60 * row]) + rng.uniform(-5, 5, (150, 2))
    pixel_centres = np.stack(np.meshgrid(np.arange(IMAGE_PX), np.arange(IMAGE_PX)), -1) + 0.5
    distance, nearest = cKDTree(cell_xy).query(pixel_centres.reshape(-1, 2))
    cells = np.where(distance <= 5, nearest + 1, 0).reshape(IMAGE_PX, IMAGE_PX).astype(np.int32)
    units = {
        "spot": ([f"spot{i:02d}" for i in range(36)], spot_xy),
        "cell": (np.arange(1, 151), cell_xy),
    }
    genes = pd.DataFrame(
        {"gene_symbol": [f"GENE{i}" for i in range(20)]}, index=[f"ENSG{i:011d}" for i in range(20)]
    )
    rna = sparse.csr_matrix(rng.poisson(rng.uniform(0.5, 5, 20), (36, 20)))
    spot_protein = rng.poisson(rng.uniform(5, 50, 5), (36, 5))
    cell_protein = rng.gamma(2.0, rng.uniform(10, 100, 5), (150, 5))
    values = {
        "spot_rna": (genes, rna, "raw_counts"),
        "spot_protein": (PROTEINS, spot_protein, "raw_counts"),
        "cell_protein": (PROTEINS, cell_protein, "raw_intensity"),
    }
    tables = {}
    for name, (var, matrix, value_scale) in values.items():
        ids, xy = units[name.split("_")[0]]
        part = np.where(xy[:, 0] < IMAGE_PX / 2, "1", "2").astype(object)  # left half is patient 1
        obs = pd.DataFrame(
            {
                "section_id": f"{store_id}_s" + part,
                "sample_id": f"{store_id}_sample" + part,
                "patient_id": f"{store_id}_p" + part,
            }
        )
        matrix = matrix.astype(np.int32 if value_scale == "raw_counts" else np.float32)
        tables[name] = table(name, ids, obs, var.copy(), matrix, xy, value_scale)
    same_section = {"kind": "same_section", "source": "synthetic"}
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": "synthetic",
        "store_id": store_id,
        "sections": [f"{store_id}_s1", f"{store_id}_s2"],
        "tissue": "synthetic",
        "fixation": "none",
        "licence": "CC0-1.0",
        "release_date": "2026-11-23",
        "pairing": {"he": same_section, "rna": same_section, "protein": same_section},
        "registration": [
            {
                "moving": "cells",
                "reference": "he",
                "kind": "identity",
                "baked": False,
                "method": "none, drawn in the H&E frame",
                "software": "usbb.data.synthetic",
                "qc": {"residual_um": 0.0},
            }
        ],
        "provenance": {"generator": "usbb.data.synthetic.synthetic_store", "seed": seed},
    }
    pixels = rng.integers(0, 256, (3, IMAGE_PX, IMAGE_PX), dtype=np.uint8)
    one_um = {"global": Scale([1.0, 1.0], axes=("y", "x"))}
    he = Image2DModel.parse(
        pixels, dims=("c", "y", "x"), c_coords=list("rgb"), transformations=one_um
    )
    labels = Labels2DModel.parse(cells, dims=("y", "x"), transformations={"global": Identity()})
    return sd.SpatialData(
        images={"he": he},
        labels={"cells": labels},
        shapes={"spots": circles(spot_xy, units["spot"][0], SPOT_RADIUS_UM)},
        tables=tables,
        attrs={"usbb": attrs},
    )
