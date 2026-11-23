"""Reader for 10x Visium CytAssist Gene and Protein Expression outputs (Space Ranger 2.x)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile
from PIL import Image
from scipy import sparse
from scipy.spatial import cKDTree
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from usbb.data.schema import SCHEMA_VERSION, SPOT_RADIUS_UM, circles, table

SPOT_PITCH_UM = 100.0  # centre-to-centre distance of Visium v1/v2 spots (10x Visium documentation)
MAX_PIXEL_SIZE_DISAGREEMENT = 0.03
UM_PER_TIFF_UNIT = {2: 25400.0, 3: 10000.0, 4: 1000.0, 5: 1.0}  # inch, cm, mm, µm
DATASETS = {
    "tenx_tonsil": {
        "tissue": "tonsil",
        "fixation": "FFPE",
        "licence": "CC BY 4.0",
        "release_date": "2023-05-15",
    }
}


def read_tenx_cytassist(
    raw_dir, store_id: str, he_image, dataset_key: str = "tenx_tonsil"
) -> sd.SpatialData:
    """Store with spots, spot_rna and spot_protein (raw counts) and the H&E image at `he_image`."""
    raw_dir, he_image = Path(raw_dir), Path(he_image)
    spatial = next(raw_dir.rglob("scalefactors_json.json")).parent
    positions = pd.read_csv(spatial / "tissue_positions.csv", index_col="barcode")
    counts, features, barcodes, software = read_filtered_matrix(
        next(raw_dir.glob("*filtered_feature_bc_matrix.h5"))
    )
    pixels, recorded_um_per_px = read_rgb(he_image)
    um_per_px, image_per_fullres_px, pitch_px = he_pixel_size(
        spatial, positions, pixels.shape, recorded_um_per_px
    )
    xy = positions.loc[barcodes, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    xy *= image_per_fullres_px * um_per_px
    obs = pd.DataFrame(
        {"section_id": store_id, "sample_id": store_id, "patient_id": store_id}, index=barcodes
    )
    is_rna = (features["feature_type"] == "Gene Expression").to_numpy()
    antibodies = features[~is_rna]
    is_control = antibodies["isotype_control"].eq("TRUE").to_numpy()
    rna_var = pd.DataFrame(
        {"gene_symbol": features["name"][is_rna].to_numpy()}, index=features["id"][is_rna]
    )
    protein_var = pd.DataFrame(
        {
            "gene_symbol": np.where(is_control, "", antibodies["name"]),
            "protein_name": antibodies["secondary_name"].to_numpy(),
            "is_control": is_control,
            "mapping_source": np.where(is_control, "none", "10x feature reference name"),
        },
        index=antibodies["id"],
    )
    he = Image2DModel.parse(
        np.moveaxis(pixels, -1, 0),
        dims=("c", "y", "x"),
        c_coords=["r", "g", "b"],
        transformations={"global": Scale([um_per_px, um_per_px], axes=("y", "x"))},
    )
    same_section = {
        "kind": "same_section",
        "source": "10x CytAssist Gene and Protein Expression of one section",
    }
    registration = {
        "moving": "spots",
        "reference": "he",
        "kind": "affine",
        "baked": True,
        "method": "Space Ranger CytAssist to microscope image registration",
        "software": software,
        "qc": {
            "status": "missing",
            "reason": "Space Ranger outputs report no registration residual",
        },
    }
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "store_id": store_id,
        "sections": [store_id],
        **DATASETS[dataset_key],
        "pairing": {"he": same_section, "rna": same_section, "protein": same_section},
        "registration": [registration],
        "provenance": {
            "reader": "usbb.data.tenx.read_tenx_cytassist",
            "raw_dir": raw_dir.name,
            "he_image": he_image.name,
            "he_um_per_px_source": "image file"
            if recorded_um_per_px
            else "spot lattice (image records none)",
            "spot_pitch_fullres_px": pitch_px,
        },
    }
    return sd.SpatialData(
        images={"he": he},
        shapes={"spots": circles(xy, barcodes, SPOT_RADIUS_UM)},
        tables={
            "spot_rna": table(
                "spot_rna", barcodes, obs, rna_var, counts[:, is_rna], xy, "raw_counts"
            ),
            "spot_protein": table(
                "spot_protein", barcodes, obs, protein_var, counts[:, ~is_rna], xy, "raw_counts"
            ),
        },
        attrs={"usbb": attrs},
    )


def he_pixel_size(spatial: Path, positions, image_shape, recorded_um_per_px):
    """µm per image pixel, image pixels per full-resolution pixel, spot pitch in full-res pixels.

    The image scale comes from the Space Ranger scale factors. The µm come from the image file when
    it records a pixel size, else from the spot lattice; the two must agree within 3 %.
    """
    scalefactors = json.loads((spatial / "scalefactors_json.json").read_text())
    # The hires image is the full-resolution one, longest side scaled by tissue_hires_scalef.
    fullres_side = (
        max(Image.open(spatial / "tissue_hires_image.png").size)
        / scalefactors["tissue_hires_scalef"]
    )
    image_per_fullres_px = max(image_shape[:2]) / fullres_side
    fullres_xy = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    # Mean distance to the six lattice neighbours; the 1.2 cut drops farther spots at array edges.
    distances = cKDTree(fullres_xy).query(fullres_xy, k=7)[0][:, 1:]
    pitch_px = float(distances[distances < 1.2 * distances[:, :1]].mean())
    lattice_um_per_px = SPOT_PITCH_UM / (pitch_px * image_per_fullres_px)
    if recorded_um_per_px is None:
        return lattice_um_per_px, image_per_fullres_px, pitch_px
    if abs(recorded_um_per_px / lattice_um_per_px - 1) > MAX_PIXEL_SIZE_DISAGREEMENT:
        raise ValueError(
            f"the H&E image records {recorded_um_per_px:.4f} µm/px but the"
            f" {SPOT_PITCH_UM:.0f} µm spot lattice gives {lattice_um_per_px:.4f} µm/px,"
            f" more than {MAX_PIXEL_SIZE_DISAGREEMENT:.0%} apart"
        )
    return recorded_um_per_px, image_per_fullres_px, pitch_px


def read_filtered_matrix(path):
    """Counts (barcodes x features, CSR), features, barcodes and Space Ranger version of an h5."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        counts = sparse.csc_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape
        ).T.tocsr()
        columns = {
            k: v[:].astype(str) for k, v in m["features"].items() if isinstance(v, h5py.Dataset)
        }
        features = pd.DataFrame({k: v for k, v in columns.items() if len(v) == shape[0]})
        return (
            counts,
            features,
            m["barcodes"][:].astype(str).tolist(),
            str(f.attrs["software_version"]),
        )


def read_rgb(path: Path):
    """RGB pixels (y, x, 3) and the µm per pixel the file records, or None when it records none."""
    if path.suffix.lower() not in (".tif", ".tiff", ".btf"):
        return np.asarray(Image.open(path).convert("RGB")), None
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        um_per_unit = UM_PER_TIFF_UNIT.get(int(page.resolutionunit))
        recorded = "XResolution" in page.tags and um_per_unit is not None
        return page.asarray(), (um_per_unit / page.resolution[0] if recorded else None)
