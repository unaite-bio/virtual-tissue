"""Phikon-v2 through the method step: a tiny random Dinov2Model on CPU, the real weights on a GPU."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from usbb.card import card_by_id
from usbb.io import write_h5ad
from usbb.methods.phikon_v2.method import PhikonV2
from usbb.run import run_method

PATCHES = np.random.default_rng(0).integers(0, 256, (5, 224, 224, 3), dtype=np.uint8)
IDS = [f"toy:spot{i}" for i in range(len(PATCHES))]


def write_bundle(folder, order=(0, 1, 2, 3, 4)):
    """A bundle holding he.h5ad: PATCHES as spots at 0.5 µm/px, rows in the given order."""
    order = list(order)
    ids = [IDS[i] for i in order]
    obs = pd.DataFrame(
        {"store_id": "toy", "section_id": "s1", "sample_id": "s1", "patient_id": "p1"}, index=ids
    )
    uns = {"dataset_id": "toy", "modality": "he", "unit_type": "spot", "value_scale": "raw"}
    uns.update(normalization_id="raw", patch={"um_per_px": 0.5, "size_px": 224})
    obsm = {"spatial": np.zeros((len(ids), 2)), "patches": PATCHES[order]}
    folder.mkdir()
    write_h5ad(anndata.AnnData(obs=obs, obsm=obsm, uns=uns), folder / "he.h5ad")
    return folder


def embeddings(bundle, run_dir, model_dir):
    """Run phikon_v2 on a bundle; return its embeddings indexed by unit id."""
    out = anndata.read_h5ad(run_method("phikon_v2", None, bundle, run_dir, model_dir=model_dir))
    return pd.DataFrame(out.obsm["cls"], index=out.obs_names)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A randomly initialised Dinov2Model, 2 layers of width 32, saved like the pinned weights."""
    path = tmp_path_factory.mktemp("tiny_phikon_v2")
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    transformers.Dinov2Model(config).save_pretrained(path)
    return path


def test_one_cls_vector_per_spot(tiny_model, tmp_path):
    cls = embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", tiny_model)
    assert list(cls.index) == IDS and cls.shape == (5, 32)


def test_embeddings_follow_unit_ids_across_batches(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr(PhikonV2, "batch_size", 2)
    first = embeddings(write_bundle(tmp_path / "a"), tmp_path / "run_a", tiny_model)
    permuted = embeddings(
        write_bundle(tmp_path / "b", (3, 0, 4, 2, 1)), tmp_path / "run_b", tiny_model
    )
    np.testing.assert_allclose(permuted.loc[IDS], first.loc[IDS], atol=1e-5)


def test_changing_a_weight_during_embed_fails_the_freeze_check(tiny_model, tmp_path, monkeypatch):
    original_embed = PhikonV2.embed

    def drifting(self, patches):
        self.model.embeddings.cls_token.add_(1.0)
        return original_embed(self, patches)

    monkeypatch.setattr(PhikonV2, "embed", drifting)
    with pytest.raises(RuntimeError, match="frozen"):
        embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", tiny_model)


@pytest.mark.slow
@pytest.mark.gpu
def test_matches_the_official_processor_and_model_at_the_pinned_revision(tmp_path):
    from huggingface_hub import snapshot_download

    weights = card_by_id("phikon_v2").weights
    repo, revision = weights["repo"], weights["revision"]
    model_dir = snapshot_download(repo, revision=revision, allow_patterns=list(weights["files"]))
    ours = embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = transformers.AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = transformers.AutoModel.from_pretrained(repo, revision=revision).to(device).eval()
    with torch.inference_mode():
        inputs = processor(images=list(PATCHES), return_tensors="pt").to(device)
        official = model(**inputs).last_hidden_state[:, 0].cpu().numpy()
    np.testing.assert_allclose(ours.loc[IDS], official, rtol=1e-4, atol=1e-4)
