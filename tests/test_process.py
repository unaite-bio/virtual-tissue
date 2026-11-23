from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("spatialdata")

from usbb.data.process import bundle_key, prepare
from usbb.data.synthetic import synthetic_store
from usbb.io import check_file, read_input

RNA = {"kind": "matrix", "value_scale": "raw_counts"}
HE = {"kind": "image", "um_per_px": 1.0, "patch_px": 32}


def card(**inputs):
    """Stand-in for usbb.card.MethodCard: only the attributes prepare reads."""
    return SimpleNamespace(method_id="stand_in", units=["spot", "cell"], inputs=inputs)


def checked(bundle):
    """Check each bundle file against api/file_input.yaml, then read it as the method step does."""
    for path in sorted(bundle.glob("*.h5ad")):
        check_file(path, "input")
    return read_input(bundle)


def first_patch(bundle):
    return next(checked(bundle).patches("he", 1))[1][0]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("stores") / "synthetic_a.zarr"
    synthetic_store().write(path)
    return path


def test_split_by_patient_separates_train_and_test(store, tmp_path):
    split = {"by": "patient_id", "test": ["synthetic_a_p2"]}
    train, test = prepare([store], card(rna=RNA), "spot", split, tmp_path)
    assert set(checked(test).obs["patient_id"]) == {"synthetic_a_p2"}
    assert set(checked(train).obs["patient_id"]) == {"synthetic_a_p1"}


def test_files_of_a_bundle_share_unit_ids(store, tmp_path):
    train, test = prepare([store], card(rna=RNA, he=HE), "spot", None, tmp_path)
    inp = checked(test)
    assert train is None and sorted(inp.modalities) == ["he", "rna"]
    assert inp.unit_ids[0] == "synthetic_a:spot00" and len(inp.unit_ids) == 36
    assert inp.matrix("rna").shape == (36, 20)


def test_cell_units_carry_raw_intensities(store, tmp_path):
    protein = {"kind": "matrix", "value_scale": "raw_intensity"}
    _, test = prepare([store], card(protein=protein), "cell", None, tmp_path)
    adata = checked(test).modalities["protein"]
    assert adata.layers["intensity"].shape == (150, 5)
    assert adata.uns["normalization_id"] == "raw_intensity"


def test_patch_is_the_image_centred_on_the_unit(store, tmp_path):
    _, test = prepare([store], card(he=HE), "spot", None, tmp_path)
    image = synthetic_store().images["he"].values  # spot00 is at (40, 40) µm, 1 µm per pixel
    np.testing.assert_array_equal(first_patch(test), np.moveaxis(image[:, 24:56, 24:56], 0, -1))
    assert checked(test).modalities["he"].uns["patch"] == {"um_per_px": 1.0, "size_px": 32}


def test_patch_is_zero_outside_the_image(store, tmp_path):
    _, test = prepare([store], card(he=dict(HE, patch_px=128)), "spot", None, tmp_path)
    patch = first_patch(test)  # 64 px half-width around (40, 40): 24 px fall outside
    assert patch[:24].max() == 0 and patch[:, :24].max() == 0 and patch[24:, 24:].max() > 0


def test_bundle_folder_is_named_by_its_content(store, tmp_path):
    _, first = prepare([store], card(rna=RNA), "spot", None, tmp_path)
    _, again = prepare([store], card(rna=RNA), "spot", None, tmp_path)
    assert first == again and first.name == bundle_key(first)[:16]


def test_image_coarser_than_the_card_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="upsampling"):
        prepare([store], card(he=dict(HE, um_per_px=0.5)), "spot", None, tmp_path)
