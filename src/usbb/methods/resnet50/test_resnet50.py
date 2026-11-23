"""ResNet-50 through the method step: random weights on CPU, the pinned ImageNet weights on a GPU."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from safetensors.torch import save_file

from usbb.card import card_by_id
from usbb.io import write_h5ad
from usbb.methods.resnet50.method import ResNet50
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
    """Run resnet50 on a bundle; return its embeddings indexed by unit id."""
    out = anndata.read_h5ad(run_method("resnet50", None, bundle, run_dir, model_dir=model_dir))
    return pd.DataFrame(out.obsm["pooled"], index=out.obs_names)


@pytest.fixture(scope="module")
def random_model(tmp_path_factory):
    """Randomly initialised ResNet-50 weights saved like the pinned file; load() builds resnet50."""
    path = tmp_path_factory.mktemp("random_resnet50")
    save_file(timm.create_model("resnet50").state_dict(), path / "model.safetensors")
    return path


def test_one_pooled_vector_per_spot(random_model, tmp_path):
    pooled = embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", random_model)
    assert list(pooled.index) == IDS and pooled.shape == (5, 2048)


def test_embeddings_follow_unit_ids_across_batches(random_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ResNet50, "batch_size", 2)
    first = embeddings(write_bundle(tmp_path / "a"), tmp_path / "run_a", random_model)
    permuted = embeddings(
        write_bundle(tmp_path / "b", (3, 0, 4, 2, 1)), tmp_path / "run_b", random_model
    )
    np.testing.assert_allclose(permuted.loc[IDS], first.loc[IDS], atol=1e-5)


def test_changing_a_weight_during_embed_fails_the_freeze_check(random_model, tmp_path, monkeypatch):
    original_embed = ResNet50.embed

    def drifting(self, patches):
        self.model.conv1.weight.add_(1.0)
        return original_embed(self, patches)

    monkeypatch.setattr(ResNet50, "embed", drifting)
    with pytest.raises(RuntimeError, match="frozen"):
        embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", random_model)


@pytest.mark.slow
@pytest.mark.gpu
def test_matches_the_official_timm_model_at_the_pinned_revision(tmp_path):
    from huggingface_hub import snapshot_download
    from PIL import Image

    weights = card_by_id("resnet50").weights
    repo, revision = weights["repo"], weights["revision"]
    model_dir = snapshot_download(repo, revision=revision, allow_patterns=list(weights["files"]))
    ours = embeddings(write_bundle(tmp_path / "bundle"), tmp_path / "run", model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(f"hf-hub:{repo}@{revision}", pretrained=True, num_classes=0)
    model = model.to(device).eval()
    # crop_pct 1.0 keeps the whole 224 px patch; timm's 0.875 resizes to 256 px and crops 224.
    config = dict(timm.data.resolve_model_data_config(model), crop_pct=1.0)
    transform = timm.data.create_transform(**config)
    pixels = torch.stack([transform(Image.fromarray(patch)) for patch in PATCHES])
    with torch.inference_mode():
        official = model(pixels.to(device)).cpu().numpy()
    np.testing.assert_allclose(ours.loc[IDS], official, rtol=1e-4, atol=1e-4)
