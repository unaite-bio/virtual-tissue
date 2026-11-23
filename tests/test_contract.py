"""Contract suite: every method without weights, run through the method step on synthetic bundles."""

from __future__ import annotations

import shutil

import anndata
import numpy as np
import pytest

pytest.importorskip("spatialdata")

from usbb.card import UNIT_TYPES, all_cards, card_by_id, load_card
from usbb.data.process import prepare
from usbb.data.synthetic import synthetic_store
from usbb.io import check_file, read_input, write_h5ad
from usbb.method import Method, load_method
from usbb.run import InputRefused, run_method

SPLIT = {"by": "patient_id", "test": ["synthetic_a_p2"]}
UPSTREAM = ("random_features", "mean_intensity")  # run first to feed fusers


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "synthetic_a.zarr"
    synthetic_store().write(path)
    return path


@pytest.fixture(
    scope="module",
    params=[card for card in all_cards() if card.weights is None],
    ids=lambda card: card.method_id,
)
def card(request):
    return request.param


@pytest.fixture(scope="module")
def bundles(card, store, tmp_path_factory):
    """(train, test) bundles: prepared from the store, or upstream outputs for a fuser."""
    root = tmp_path_factory.mktemp(card.method_id)
    if all(spec["kind"] != "embedding" for spec in card.inputs.values()):
        return prepare([store], card, card.units[0], SPLIT, root)
    (root / "upstream").mkdir()
    for method_id in UPSTREAM:
        train, test = prepare([store], card_by_id(method_id), "spot", SPLIT, root)
        output = run_method(method_id, train, test, root / method_id)
        shutil.copy(output, root / "upstream" / f"{method_id}.h5ad")
    return None, root / "upstream"


@pytest.fixture(scope="module")
def output(card, bundles, tmp_path_factory):
    train, test = bundles
    return run_method(card.method_id, train, test, tmp_path_factory.mktemp("run"))


@pytest.fixture
def unloadable(card, monkeypatch):
    """Make the method's load() fail, so a refusal shows it came before load."""

    def load(self):
        raise AssertionError("load ran before the input was refused")

    monkeypatch.setattr(type(load_method(card, None, 0)), "load", load)


def rewritten(bundle, folder, edit):
    """A copy of bundle in folder with edit(adata) -> adata applied to every file."""
    folder.mkdir()
    for path in sorted(bundle.glob("*.h5ad")):
        write_h5ad(edit(anndata.read_h5ad(path)), folder / path.name)
    return folder


def test_card_validates_and_names_its_class(card):
    assert isinstance(load_method(load_card(card.path), None, 0), Method)


def test_wrong_unit_type_is_refused_before_load(card, bundles, unloadable, tmp_path):
    other = [unit for unit in UNIT_TYPES if unit not in card.units]
    if not other:
        pytest.skip(f"{card.method_id} takes every unit type")

    def relabel(adata):
        adata.uns["unit_type"] = other[0]
        return adata

    test = rewritten(bundles[1], tmp_path / "test", relabel)
    with pytest.raises(InputRefused, match=f"unit type {other[0]}"):
        run_method(card.method_id, bundles[0], test, tmp_path / "run")


def test_missing_modality_is_refused_before_load(card, bundles, unloadable, tmp_path):
    adata = anndata.read_h5ad(next(bundles[1].glob("*.h5ad")))
    adata.uns.pop("method_id", None)  # an upstream output without it stops counting as one
    (tmp_path / "test").mkdir()
    write_h5ad(adata, tmp_path / "test" / "unrelated.h5ad")
    with pytest.raises(InputRefused, match="missing modality|holds no upstream method output"):
        run_method(card.method_id, bundles[0], tmp_path / "test", tmp_path / "run")


def test_permuting_input_rows_permutes_output_rows(card, bundles, output, tmp_path):
    train, test = bundles
    order = np.random.default_rng(0).permutation(len(read_input(test).unit_ids))
    shuffled = rewritten(test, tmp_path / "shuffled", lambda adata: adata[order].copy())
    expected = anndata.read_h5ad(output)
    result = anndata.read_h5ad(run_method(card.method_id, train, shuffled, tmp_path / "run"))
    assert list(result.obs_names) == list(expected.obs_names[order])
    for latent in expected.obsm:
        np.testing.assert_allclose(result.obsm[latent], expected.obsm[latent][order], atol=1e-6)


def test_same_seed_gives_identical_output(card, bundles, output, tmp_path):
    train, test = bundles
    again = anndata.read_h5ad(run_method(card.method_id, train, test, tmp_path / "run", seed=0))
    expected = anndata.read_h5ad(output)
    for latent in expected.obsm:
        np.testing.assert_array_equal(again.obsm[latent], expected.obsm[latent])


def test_output_passes_the_output_spec(output):
    check_file(output, "output")
