from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest
import yaml

from usbb.io import API, check_file, read_input, write_h5ad, write_output


def write_modality(folder, modality, n=4, start=0):
    """Write spots s1:u<start>... to <folder>/<modality>.h5ad: 8 px patches at 0.5 um/px for he, counts otherwise."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = [f"s1:u{i}" for i in range(start, start + n)]
    patients = [f"p{i % 2}" for i in range(n)]
    obs = pd.DataFrame(
        {"store_id": "s1", "section_id": "a", "sample_id": "a", "patient_id": patients}, index=ids
    )
    uns = {"dataset_id": "toy", "modality": modality, "unit_type": "spot"}
    obsm = {"spatial": np.arange(2.0 * n).reshape(n, 2)}
    if modality == "he":
        obsm["patches"] = np.repeat(np.arange(start, start + n, dtype=np.uint8), 8 * 8 * 3).reshape(
            n, 8, 8, 3
        )
        uns.update(
            value_scale="raw", normalization_id="raw", patch={"um_per_px": 0.5, "size_px": 8}
        )
        adata = anndata.AnnData(obs=obs, obsm=obsm, uns=uns)
    else:
        uns.update(value_scale="raw_counts", normalization_id="raw_counts")
        var = pd.DataFrame({"gene_symbol": ["A", "B", "C"]}, index=["g1", "g2", "g3"])
        counts = np.arange(3 * start, 3 * (start + n)).reshape(n, 3)
        adata = anndata.AnnData(obs=obs, var=var, obsm=obsm, layers={"counts": counts}, uns=uns)
    write_h5ad(adata, folder / f"{modality}.h5ad")
    return folder


@pytest.fixture
def bundle(tmp_path):
    write_modality(tmp_path, "rna")
    return write_modality(tmp_path, "he")


def test_read_input_keeps_unit_ids_in_file_order(bundle):
    inp = read_input(bundle)
    assert inp.unit_ids == ["s1:u0", "s1:u1", "s1:u2", "s1:u3"]
    assert inp.unit_type == "spot"
    assert sorted(inp.modalities) == ["he", "rna"]
    assert list(inp.obs.index) == inp.unit_ids


def test_matrix_returns_the_raw_counts(bundle):
    assert np.array_equal(read_input(bundle).matrix("rna"), np.arange(12).reshape(4, 3))


def test_patches_stay_on_disk_and_come_in_batches_with_their_ids(bundle):
    inp = read_input(bundle)
    assert "patches" not in inp.modalities["he"].obsm
    batches = list(inp.patches("he", batch_size=3))
    assert [ids for ids, _ in batches] == [inp.unit_ids[:3], inp.unit_ids[3:]]
    assert batches[1][1].dtype == np.uint8
    assert batches[1][1].shape == (1, 8, 8, 3)
    assert (batches[1][1] == 3).all()


def test_read_input_rejects_files_with_different_units(tmp_path):
    write_modality(tmp_path, "rna", n=4)
    write_modality(tmp_path, "he", n=3)
    with pytest.raises(ValueError, match="differ from he.h5ad"):
        read_input(tmp_path)


def test_bundle_files_match_the_input_spec(bundle):
    check_file(bundle / "rna.h5ad", "input")
    check_file(bundle / "he.h5ad", "input")


def test_check_file_lists_every_problem(tmp_path):
    adata = anndata.AnnData(
        obs=pd.DataFrame({"store_id": ["s1"]}, index=["s1:u0"]),
        obsm={"spatial": np.zeros((1, 2), int)},
    )
    write_h5ad(adata, tmp_path / "bad.h5ad")
    with pytest.raises(ValueError) as error:
        check_file(tmp_path / "bad.h5ad", "input")
    for problem in (
        "missing obs/patient_id",
        "missing uns/unit_type",
        "obsm/spatial is not of type double",
    ):
        assert problem in str(error.value)


def test_write_output_keeps_ids_and_writes_float32_latents(bundle, tmp_path):
    inp = read_input(bundle)
    uns = {
        "dataset_id": "toy",
        "method_id": "toy",
        "normalization_id": "raw",
        "unit_type": "spot",
        "run_key": "k",
    }
    uns["informed_by"] = {"z": ["he"]}
    write_output(tmp_path / "output.h5ad", inp, {"z": np.ones((4, 2), np.float32)}, uns)
    out = anndata.read_h5ad(tmp_path / "output.h5ad")
    assert list(out.obs_names) == inp.unit_ids
    assert out.obsm["z"].dtype == np.float32
    assert list(out.uns["informed_by"]["z"]) == ["he"]


def test_api_files_follow_the_open_problems_file_format():
    for spec in ("input", "output"):
        info = yaml.safe_load((API / f"file_{spec}.yaml").read_text(encoding="utf-8"))
        assert info["type"] == "file" and info["label"] and info["summary"]
        assert info["info"]["format"]["type"] == "h5ad"
        for slot in ("obs", "var", "obsm", "layers", "uns"):
            for entry in info["info"]["format"].get(slot, []):
                assert set(entry) == {"name", "type", "description", "required"}
