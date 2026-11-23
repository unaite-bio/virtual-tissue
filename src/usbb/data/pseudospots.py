"""Pseudospots: Visium-like circles that pool the cells whose centroid they contain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from usbb.data.schema import OBS_COLUMNS, circles, table


def make_pseudospots(
    sdata, geometry: str = "spots", diameter_um: float = 55.0, spacing_um: float = 100.0
) -> None:
    """Add `pseudospots`, table `pseudospot_protein` and cell_protein.obs["pseudospot_id"] in place.

    geometry "spots" reuses the spot centres and ids; "hex" uses a lattice anchored at the origin.
    Only circles that contain at least one cell are kept.
    """
    cells = sdata.tables["cell_protein"]
    if geometry == "spots":
        spots = sdata.shapes["spots"]
        ids, centres = (
            spots.index.to_numpy(str),
            np.column_stack([spots.geometry.x, spots.geometry.y]),
        )
    elif geometry == "hex":
        ids, centres = hex_lattice(cells.obsm["spatial"].max(axis=0), spacing_um)
    else:
        raise ValueError(f"geometry must be 'spots' or 'hex', not {geometry!r}")
    distance, nearest = cKDTree(centres).query(cells.obsm["spatial"])
    member = np.where(distance <= diameter_um / 2, ids[nearest], "")
    cells.obs["pseudospot_id"] = member
    inside = member != ""
    columns = [c for c in OBS_COLUMNS + ("core_id",) if c in cells.obs]
    groups = cells.obs.loc[inside, columns].groupby(member[inside])
    mixed = groups.nunique().max(axis=1) > 1
    if mixed.any():
        raise ValueError(
            f"pseudospots span several sections, samples or patients: {list(mixed.index[mixed])}"
        )
    obs = groups.first().assign(n_cells=groups.size())
    means = pd.DataFrame(cells.layers["intensity"][inside]).groupby(member[inside]).mean()
    xy = centres[pd.Index(ids).get_indexer(obs.index)]
    pooled = table(
        "pseudospot_protein",
        obs.index,
        obs,
        cells.var.copy(),
        means.to_numpy(np.float32),
        xy,
        cells.uns["value_scale"],
    )
    pooled.uns["pseudospots"] = {
        "geometry": geometry,
        "diameter_um": float(diameter_um),
        "spacing_um": float(spacing_um),
        "membership": "cell centroid inside the circle (nearest centre when circles overlap)",
        "aggregation": "mean of member cells",
        "source_table": "cell_protein",
    }
    sdata.shapes["pseudospots"] = circles(xy, obs.index, diameter_um / 2)
    sdata.tables["pseudospot_protein"] = pooled


def hex_lattice(corner_um, spacing_um: float):
    """Ids (r0000c0000) and centres of a hexagonal lattice from the origin to `corner_um` (x, y)."""
    row_step = spacing_um * np.sqrt(3) / 2
    n_rows, n_cols = int(corner_um[1] // row_step) + 2, int(corner_um[0] // spacing_um) + 2
    row, col = np.divmod(np.arange(n_rows * n_cols), n_cols)
    ids = np.array([f"r{r:04d}c{c:04d}" for r, c in zip(row, col)])
    return ids, np.column_stack([spacing_um * (col + (row % 2) / 2), row_step * row])
