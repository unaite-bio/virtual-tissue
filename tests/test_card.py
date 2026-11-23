from __future__ import annotations

import sys

import pytest
import yaml

from usbb.card import CardError, Output, all_cards, card_by_id, load_card

CARD = {
    "method_id": "toy",
    "class": "Toy",
    "label": "Toy",
    "summary": "A toy method used by the tests.",
    "description": "Centres the counts.",
    "references": {"doi": []},
    "links": {"repository": None, "documentation": None},
    "family": "encoder",
    "control": False,
    "fit_mode": "fitted",
    "units": ["spot"],
    "inputs": {"rna": {"kind": "matrix", "value_scale": "raw_counts"}},
    "outputs": [{"latent": "z", "informed_by": ["rna"]}],
    "parameters": {"scale": 1.0},
    "weights": None,
    "env_group": "core",
    "resources": {"cpu": "lowcpu", "memory": "lowmem", "time": "lowtime", "gpu": None},
    "heldout": {"tenx_tonsil": "?"},
    "intended_usage": "Tests only.",
}


def write_card(root, **changes):
    """Write CARD with changes to <root>/<method_id>/card.yaml and return its path."""
    card = dict(CARD, **changes)
    path = root / card["method_id"] / "card.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(card), encoding="utf-8")
    return path


def test_load_card_reads_every_field(tmp_path):
    card = load_card(write_card(tmp_path))
    assert card.class_name == "Toy"
    assert card.outputs == [Output("z", ["rna"])]
    assert card.heldout == {"tenx_tonsil": "?"}
    assert card.path == tmp_path / "toy" / "card.yaml"


def test_card_error_lists_every_problem(tmp_path):
    path = write_card(tmp_path, family="painter", summary="Too short.", units=["tile"])
    with pytest.raises(CardError) as error:
        load_card(path)
    for key in ("family", "summary", "units"):
        assert f"\n- {key} must " in str(error.value)


def test_missing_and_unknown_keys_are_reported(tmp_path):
    path = write_card(tmp_path, colour="red")
    card = yaml.safe_load(path.read_text())
    del card["label"]
    path.write_text(yaml.safe_dump(card))
    with pytest.raises(CardError, match="missing key label") as error:
        load_card(path)
    assert "unknown key colour" in str(error.value)


def test_method_id_must_match_its_folder(tmp_path):
    path = write_card(tmp_path)
    path.parent.rename(tmp_path / "other")
    with pytest.raises(CardError, match=r"method_id must be other \(its folder\)"):
        load_card(tmp_path / "other" / "card.yaml")


def test_image_input_needs_its_patch_geometry(tmp_path):
    path = write_card(tmp_path, inputs={"he": {"kind": "image", "um_per_px": 0.5}})
    with pytest.raises(CardError, match="inputs must map"):
        load_card(path)


def test_informed_by_must_name_inputs_of_the_card(tmp_path):
    path = write_card(tmp_path, outputs=[{"latent": "z", "informed_by": ["protein"]}])
    with pytest.raises(CardError, match="informed_by of z"):
        load_card(path)


def test_weights_need_a_full_revision_and_sha256s(tmp_path):
    weights = {
        "repo": "a/b",
        "revision": "main",
        "files": {"model.bin": "abc"},
        "licence": "x",
        "release_date": None,
    }
    with pytest.raises(CardError, match="weights must be"):
        load_card(write_card(tmp_path, weights=weights))


def test_all_cards_skips_guides_and_imports_no_method_code(tmp_path):
    write_card(tmp_path)
    (tmp_path / "toy" / "method.py").write_text("raise RuntimeError('imported')\n")
    (tmp_path / "_template").mkdir()
    (tmp_path / "_template" / "README.md").write_text("A guide.\n")
    assert [card.method_id for card in all_cards(tmp_path)] == ["toy"]
    assert "usbb.methods.toy.method" not in sys.modules


def test_all_cards_reports_every_broken_card_at_once(tmp_path):
    write_card(tmp_path, method_id="first", control="no")
    write_card(tmp_path, method_id="second", fit_mode="trained")
    with pytest.raises(CardError) as error:
        all_cards(tmp_path)
    assert "control must be" in str(error.value)
    assert "fit_mode must be" in str(error.value)


def test_card_by_id_reads_the_card_of_that_folder(tmp_path):
    write_card(tmp_path, method_id="other")
    assert card_by_id("other", tmp_path).method_id == "other"
