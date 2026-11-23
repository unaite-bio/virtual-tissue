import re

import numpy as np
import pytest

pytest.importorskip("spatialdata")

from usbb.data.pseudospots import make_pseudospots
from usbb.data.schema import validate
from usbb.data.synthetic import synthetic_store


def test_a_cell_joins_the_spot_circle_containing_its_centroid():
    sdata = synthetic_store()
    make_pseudospots(sdata, geometry="spots", diameter_um=55.0)
    spots = sdata.shapes["spots"]
    centres = np.column_stack([spots.geometry.x, spots.geometry.y])
    cells = sdata.tables["cell_protein"]
    distance = np.linalg.norm(cells.obsm["spatial"][:, None] - centres[None], axis=2)
    nearest = spots.index.to_numpy()[distance.argmin(axis=1)]
    expected = np.where(distance.min(axis=1) <= 27.5, nearest, "")
    assert list(cells.obs["pseudospot_id"]) == list(expected)


def test_pseudospot_values_are_the_mean_of_member_cells():
    sdata = synthetic_store()
    make_pseudospots(sdata)
    cells, pooled = sdata.tables["cell_protein"], sdata.tables["pseudospot_protein"]
    members = (cells.obs["pseudospot_id"] == pooled.obs_names[0]).to_numpy()
    expected = cells.layers["intensity"][members].mean(axis=0)
    np.testing.assert_allclose(pooled.layers["intensity"][0], expected, rtol=1e-6)
    assert pooled.obs["n_cells"].iloc[0] == members.sum()


def test_store_with_pseudospots_is_valid():
    sdata = synthetic_store()
    make_pseudospots(sdata, geometry="hex")
    validate(sdata)


def test_hex_lattice_ids_and_parameters_are_recorded():
    sdata = synthetic_store()
    make_pseudospots(sdata, geometry="hex", diameter_um=80.0, spacing_um=120.0)
    pooled = sdata.tables["pseudospot_protein"]
    assert all(re.fullmatch(r"r\d{4}c\d{4}", i) for i in pooled.obs_names)
    assert pooled.uns["pseudospots"]["spacing_um"] == 120.0
    assert np.allclose(sdata.shapes["pseudospots"].radius, 40.0)


def test_same_store_gives_the_same_pseudospots():
    first, second = synthetic_store(), synthetic_store()
    make_pseudospots(first, geometry="hex")
    make_pseudospots(second, geometry="hex")
    a, b = first.tables["pseudospot_protein"], second.tables["pseudospot_protein"]
    assert a.obs.equals(b.obs)
    np.testing.assert_array_equal(a.layers["intensity"], b.layers["intensity"])
