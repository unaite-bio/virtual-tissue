from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import types

import anndata
import numpy as np
import pytest

import usbb.card
from tests.test_card import write_card
from tests.test_io import write_modality
from usbb.card import card_by_id
from usbb.cli import main
from usbb.io import read_input
from usbb.method import Method
from usbb.run import InputRefused, run_key, run_method


class Centre(Method):
    """Subtracts the mean counts of the bundle it was fitted on."""

    def fit(self, inp):
        self.mean = inp.matrix("rna").mean(axis=0)

    def transform(self, inp):
        return {"z": (inp.matrix("rna") - self.mean) * self.params["scale"]}


class Identity(Method):
    """Frozen: returns the counts and must never be fitted."""

    def fit(self, inp):
        raise AssertionError("a frozen method was fitted")

    def transform(self, inp):
        return {"z": inp.matrix("rna")}


@pytest.fixture
def register(monkeypatch, tmp_path):
    """Make a test-only Method class loadable under a card written in tmp_path/methods."""
    monkeypatch.setattr(usbb.card, "METHODS_ROOT", tmp_path / "methods")

    def register(method_class, **changes):
        write_card(tmp_path / "methods", **changes)
        module = types.ModuleType("method")
        module.Toy = method_class
        monkeypatch.setitem(
            sys.modules, f"usbb.methods.{changes.get('method_id', 'toy')}.method", module
        )

    return register


@pytest.fixture
def train(tmp_path):
    return write_modality(tmp_path / "train", "rna", n=4)


@pytest.fixture
def test(tmp_path):
    return write_modality(tmp_path / "test", "rna", n=3, start=4)


def latent(path):
    return anndata.read_h5ad(path).obsm["z"]


def run_record(path):
    return json.loads((path.parent / "RUN.json").read_text(encoding="utf-8"))


def test_fitted_method_fits_on_train_and_embeds_the_test_units(register, train, test, tmp_path):
    register(Centre)
    path = run_method("toy", train, test, tmp_path / "run")
    out = anndata.read_h5ad(path)
    assert list(out.obs_names) == ["s1:u4", "s1:u5", "s1:u6"]
    assert out.obsm["z"].dtype == np.float32
    assert np.allclose(
        out.obsm["z"], read_input(test).matrix("rna") - read_input(train).matrix("rna").mean(axis=0)
    )
    assert dict(out.uns["informed_by"]) == {"z": ["rna"]}
    assert out.uns["run_key"] == run_record(path)["run_key"]


def test_transductive_method_fits_on_test_and_is_flagged(register, train, test, tmp_path):
    register(Centre, fit_mode="transductive")
    path = run_method("toy", train, test, tmp_path / "run")
    assert np.allclose(latent(path).mean(axis=0), 0.0)
    assert run_record(path)["transductive"] is True


def test_frozen_method_is_never_fitted_even_with_a_train_bundle(register, train, test, tmp_path):
    register(Identity, fit_mode="frozen")
    assert latent(run_method("toy", train, test, tmp_path / "run")).shape == (3, 3)


def test_success_publishes_only_output_and_clears_an_earlier_failure(
    register, train, test, tmp_path
):
    register(Centre)
    with pytest.raises(InputRefused):
        run_method("toy", None, test, tmp_path / "run")
    run_method("toy", train, test, tmp_path / "run")
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["output"]


def test_card_refusals_come_before_load_and_all_at_once(register, test, tmp_path):
    class Unloadable(Identity):
        def load(self):
            raise AssertionError("loaded")

    image = {"kind": "image", "um_per_px": 1.0, "patch_px": 16}
    inputs = {
        "rna": {"kind": "matrix", "value_scale": "raw_intensity"},
        "he": image,
        "protein": image,
    }
    register(Unloadable, fit_mode="frozen", units=["cell"], inputs=inputs)
    write_modality(test, "he", n=3, start=4)
    with pytest.raises(InputRefused) as error:
        run_method("toy", None, test, tmp_path / "run")
    for reason in (
        "unit type spot",
        "value scale raw_counts",
        "patches of 8 px at 0.5",
        "missing modality protein",
    ):
        assert reason in str(error.value)
    assert json.loads((tmp_path / "run" / "FAILED.json").read_text())["error"] == "InputRefused"


def test_method_check_refuses_before_load(register, test, tmp_path):
    class Picky(Identity):
        def check(self, inp):
            return ["needs at least 10 genes"]

    register(Picky, fit_mode="frozen")
    with pytest.raises(InputRefused, match="needs at least 10 genes"):
        run_method("toy", None, test, tmp_path / "run")


def test_fitted_method_without_train_bundle_is_refused(register, test, tmp_path):
    register(Centre)
    with pytest.raises(InputRefused, match="needs a train bundle"):
        run_method("toy", None, test, tmp_path / "run")


def test_same_run_key_is_a_cache_hit_and_a_new_seed_reruns(register, train, test, tmp_path):
    loads = []

    class Counted(Centre):
        def load(self):
            loads.append(self.seed)

    register(Counted)
    for seed in (0, 0, 1):
        run_method("toy", train, test, tmp_path / "run", seed=seed)
    assert loads == [0, 1]


def test_random_numpy_and_torch_are_seeded(register, test, tmp_path, monkeypatch):
    torch_seeds = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=torch_seeds.append))

    class Noise(Identity):
        def transform(self, inp):
            return {"z": np.random.rand(3, 2) + random.random()}

    register(Noise, fit_mode="frozen")
    runs = [
        latent(run_method("toy", None, test, tmp_path / f"run{i}", seed=seed))
        for i, seed in enumerate((0, 0, 1))
    ]
    assert np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[0], runs[2])
    assert torch_seeds == [0, 0, 1]


def test_frozen_method_whose_state_changes_fails(register, test, tmp_path):
    class Drifting(Identity):
        weights = np.zeros(2)

        def state(self):
            return {"weights": self.weights}

        def transform(self, inp):
            self.weights = self.weights + 1
            return super().transform(inp)

    register(Drifting, fit_mode="frozen")
    with pytest.raises(RuntimeError, match="state"):
        run_method("toy", None, test, tmp_path / "run")
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["FAILED.json"]


@pytest.mark.parametrize(
    "latents", [{"other": np.ones((3, 2))}, {"z": np.ones((2, 2))}, {"z": np.full((3, 2), np.nan)}]
)
def test_outputs_must_match_the_card_latents_rows_and_be_finite(register, test, tmp_path, latents):
    class Bad(Identity):
        def transform(self, inp):
            return latents

    register(Bad, fit_mode="frozen")
    with pytest.raises(ValueError, match="output check failed"):
        run_method("toy", None, test, tmp_path / "run")


def test_unknown_parameters_are_refused(register, train, test, tmp_path):
    register(Centre)
    with pytest.raises(ValueError, match="has no parameters"):
        run_method("toy", train, test, tmp_path / "run", params={"n_pcs": 3})


def test_fuser_output_is_informed_by_the_upstream_modalities(register, train, test, tmp_path):
    register(Centre)
    (tmp_path / "fuse_in").mkdir()
    shutil.copy(run_method("toy", train, test, tmp_path / "toy"), tmp_path / "fuse_in" / "toy.h5ad")

    class Fuse(Identity):
        def transform(self, inp):
            return {"z": inp.modalities["toy"].obsm["z"]}

    inputs = {"upstream": {"kind": "embedding"}}
    outputs = [{"latent": "z", "informed_by": ["upstream"]}]
    register(
        Fuse, method_id="fuse", family="fuser", fit_mode="frozen", inputs=inputs, outputs=outputs
    )
    out = anndata.read_h5ad(run_method("fuse", None, tmp_path / "fuse_in", tmp_path / "fuse"))
    assert dict(out.uns["informed_by"]) == {"z": ["rna"]}


def test_run_key_is_a_content_hash(register, train, test, tmp_path):
    register(Centre)
    card = card_by_id("toy")
    key = run_key(card, None, 0, train, test)
    (card.path.parent / "__pycache__").mkdir()
    (card.path.parent / "__pycache__" / "method.pyc").write_bytes(b"compiled")
    assert run_key(card, {"scale": 1.0}, 0, train, test) == key
    assert run_key(card, {"scale": 2.0}, 0, train, test) != key
    assert run_key(card, None, 1, train, test) != key
    assert run_key(card, None, 0, test, test) != key
    (card.path.parent / "README.md").write_text("A guide.\n")
    assert run_key(card, None, 0, train, test) != key


def test_cli_run_publishes_the_output(register, train, test, tmp_path):
    register(Centre)
    main(
        [
            "run",
            "--method",
            "toy",
            "--input_train",
            str(train),
            "--input_test",
            str(test),
            "--output",
            str(tmp_path / "run"),
            "--params",
            '{"scale": 2.0}',
        ]
    )
    assert run_record(tmp_path / "run" / "output" / "output.h5ad")["params"] == {"scale": 2.0}


def test_cli_run_failure_writes_failed_json_and_exits_1(test, tmp_path):
    command = [
        sys.executable,
        "-m",
        "usbb.cli",
        "run",
        "--method",
        "missing",
        "--input_test",
        str(test),
        "--output",
        str(tmp_path / "run"),
    ]
    assert subprocess.run(command, check=False, capture_output=True).returncode == 1
    assert (tmp_path / "run" / "FAILED.json").exists()
