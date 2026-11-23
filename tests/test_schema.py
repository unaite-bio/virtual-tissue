import pytest

pytest.importorskip("spatialdata")

from usbb.data.schema import SchemaError, validate, validate_store
from usbb.data.synthetic import synthetic_store


def test_synthetic_store_is_valid():
    validate(synthetic_store())


def test_store_validates_after_a_zarr_round_trip(tmp_path):
    synthetic_store().write(tmp_path / "a.zarr")
    validate_store(tmp_path / "a.zarr")


def test_one_error_lists_every_violation():
    sdata = synthetic_store()
    del sdata.attrs["usbb"]["licence"]
    sdata.shapes["spots"]["radius"] = 30.0
    sdata.tables["spot_rna"].layers["counts"].data[0] = -1
    with pytest.raises(SchemaError) as error:
        validate(sdata)
    assert all(word in str(error.value) for word in ("licence", "radius", "non-negative"))


def test_unbaked_nonrigid_registration_is_refused():
    sdata = synthetic_store()
    sdata.attrs["usbb"]["registration"][0].update(kind="nonrigid", baked=False)
    with pytest.raises(SchemaError, match="non-rigid"):
        validate(sdata)


def test_every_modality_needs_a_pairing():
    sdata = synthetic_store()
    del sdata.attrs["usbb"]["pairing"]["protein"]
    with pytest.raises(SchemaError, match="pairing\\['protein'\\]"):
        validate(sdata)


def test_pseudospot_ids_must_exist():
    sdata = synthetic_store()
    sdata.tables["cell_protein"].obs["pseudospot_id"] = "nowhere"
    with pytest.raises(SchemaError, match="pseudospot_id"):
        validate(sdata)
