from __future__ import annotations

import sys
import types

import anndata
import pytest
import yaml

pytest.importorskip("spatialdata")

import usbb.card
from tests.test_card import write_card
from usbb import launch
from usbb.cli import main
from usbb.data.synthetic import synthetic_store
from usbb.executors import LocalExecutor
from usbb.weights import fetch_weights

SPLIT = {"by": "store_id", "test": ["synthetic_b"]}
FUSED = ["random_features", "hvg_pca", {"concat": {"inputs": ["random_features", "hvg_pca"]}}]


@pytest.fixture(scope="module", autouse=True)
def results(tmp_path_factory):
    """A usbb.local.toml that sends every run of this module to a temporary results folder."""
    root = tmp_path_factory.mktemp("launch")
    config = root / "usbb.local.toml"
    config.write_text(f'results = "{(root / "results").as_posix()}"\n', encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(launch, "LOCAL_CONFIG", config)
        yield root / "results"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A folder with two synthetic stores of one dataset."""
    folder = tmp_path_factory.mktemp("synthetic")
    for seed, store_id in enumerate(["synthetic_a", "synthetic_b"]):
        synthetic_store(store_id, seed).write(folder / f"{store_id}.zarr")
    return folder


def write_experiment(path, dataset, methods):
    content = {"dataset": str(dataset), "unit": "spot", "split": SPLIT, "methods": methods}
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fused(dataset, tmp_path_factory):
    """An experiment with two methods and a fuser of their outputs, launched once."""
    path = write_experiment(tmp_path_factory.mktemp("fused") / "fused.yaml", dataset, FUSED)
    assert launch.launch_experiment(path)
    return path


@pytest.fixture
def toy_methods(monkeypatch, tmp_path):
    monkeypatch.setattr(usbb.card, "METHODS_ROOT", tmp_path / "methods")
    return tmp_path / "methods"


def test_fuser_reads_the_outputs_of_the_methods_above_it(fused, results):
    paths = results.glob("runs/*/*/output/output.h5ad")
    out = {path.parents[2].name: anndata.read_h5ad(path) for path in paths}
    assert sorted(out) == ["concat", "hvg_pca", "random_features"]
    assert set(out["concat"].obs["store_id"]) == {"synthetic_b"}
    assert out["concat"].obsm["concat"].shape[1] == out["hvg_pca"].obsm["pca"].shape[1] + 50


def test_second_launch_does_no_work(fused, monkeypatch):
    ran = []
    monkeypatch.setattr(LocalExecutor, "run", lambda self, tasks: ran.extend(tasks))
    assert launch.launch_experiment(fused)
    assert ran == []


def test_failing_method_writes_failed_json_and_exits_with_1(dataset, results, tmp_path):
    methods = [{"mean_intensity": {"cofactor": 0}}]
    path = write_experiment(tmp_path / "failing.yaml", dataset, methods)
    with pytest.raises(SystemExit) as stop:
        main(["launch", str(path)])
    assert stop.value.code == 1
    assert len(list(results.glob("runs/mean_intensity/*/FAILED.json"))) == 1


def test_groups_without_a_command_are_refused_together(dataset, toy_methods, tmp_path):
    write_card(toy_methods, method_id="on_gpu", env_group="torch")
    write_card(toy_methods, method_id="on_tpu", env_group="jax")
    path = write_experiment(tmp_path / "groups.yaml", dataset, ["on_gpu", "on_tpu"])
    with pytest.raises(ValueError, match=r"no \[groups\] command for \['jax', 'torch'\]"):
        launch.launch_experiment(path)


def test_fitted_fuser_is_refused(dataset, toy_methods, tmp_path):
    write_card(toy_methods)  # toy: a fitted encoder of rna
    upstream = {"upstream": {"kind": "embedding"}}
    outputs = [{"latent": "z", "informed_by": ["upstream"]}]
    write_card(toy_methods, method_id="fuse", family="fuser", inputs=upstream, outputs=outputs)
    path = write_experiment(
        tmp_path / "fuser.yaml", dataset, ["toy", {"fuse": {"inputs": ["toy"]}}]
    )
    with pytest.raises(ValueError, match="fitted fusers are not supported"):
        launch.launch_experiment(path)


def test_weights_whose_sha256_differs_from_the_card_are_refused(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"not the pinned weights")
    hub = types.SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    weights = {"repo": "org/model", "revision": "0" * 40, "files": {"model.safetensors": "0" * 64}}
    with pytest.raises(ValueError, match="model.safetensors"):
        fetch_weights(weights)
