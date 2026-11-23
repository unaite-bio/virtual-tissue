from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("spatialdata")

import tifffile
from PIL import Image
from scipy.spatial import cKDTree

from usbb.data.schema import um_per_px, validate
from usbb.data.tenx import read_tenx_cytassist

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "tenx_tonsil_cytassist_gex_protein"
HIRES = next(RAW.rglob("tissue_hires_image.png"), None) if RAW.exists() else None
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(HIRES is None, reason="10x tonsil files are not local"),
]


@pytest.fixture(scope="module")
def tonsil():
    return read_tenx_cytassist(RAW, "tonsil", HIRES)


def as_tiff(folder, um):
    """The hires image as a TIFF that records `um` µm per pixel."""
    path = folder / "he.tif"
    pixels = np.asarray(Image.open(HIRES).convert("RGB"))
    tifffile.imwrite(path, pixels, resolution=(1e4 / um, 1e4 / um), resolutionunit="CENTIMETER")
    return path


def test_tonsil_store_is_valid(tonsil):
    validate(tonsil)


def test_png_pixel_size_comes_from_the_spot_lattice(tonsil):
    # 100 µm over a 365.9 px full-resolution pitch, divided by tissue_hires_scalef 0.0511
    assert um_per_px(tonsil.images["he"])[0] == pytest.approx(5.347, rel=0.002)


def test_spots_are_55_um_circles_100_um_apart(tonsil):
    xy = tonsil.tables["spot_rna"].obsm["spatial"]
    assert np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]) == pytest.approx(100, rel=0.01)
    assert np.allclose(tonsil.shapes["spots"].radius, 27.5)


def test_feature_types_are_split_and_isotype_controls_flagged(tonsil):
    rna, protein = tonsil.tables["spot_rna"], tonsil.tables["spot_protein"]
    assert rna.shape == (4194, 18085) and protein.shape == (4194, 35)
    assert protein.var["is_control"].sum() == 4
    assert list(protein.var.loc["PTPRC_1", ["gene_symbol", "protein_name"]]) == ["PTPRC", "CD45RA"]


def test_recorded_pixel_size_is_used_when_it_agrees_with_the_lattice(tmp_path):
    sdata = read_tenx_cytassist(RAW, "tonsil", as_tiff(tmp_path, 5.40))
    assert um_per_px(sdata.images["he"])[0] == pytest.approx(5.40)


def test_recorded_pixel_size_far_from_the_lattice_fails(tmp_path):
    with pytest.raises(ValueError, match="spot lattice"):
        read_tenx_cytassist(RAW, "tonsil", as_tiff(tmp_path, 6.0))
