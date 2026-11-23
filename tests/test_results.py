from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("spatialdata")

from usbb.card import card_by_id
from usbb.data.process import prepare
from usbb.data.synthetic import synthetic_store
from usbb.results import finished_runs
from usbb.run import run_method

# Run in a fresh interpreter: other tests of the session import method code themselves.
READ_RUNS = """
import json, sys
import anndata
from usbb.results import finished_runs
runs = finished_runs(sys.argv[1])
print(json.dumps({
    "methods": [run.card.method_id for run in runs],
    "rows": [anndata.read_h5ad(run.output).n_obs for run in runs],
    "imported": [name for name in sys.modules if name.startswith("usbb.methods")],
}))
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Results of two methods run on one synthetic store."""
    root = tmp_path_factory.mktemp("results")
    store = root / "synthetic_a.zarr"
    synthetic_store("synthetic_a", 0).write(store)
    for method_id in ("mean_intensity", "random_features"):
        _, test = prepare([store], card_by_id(method_id), "spot", None, root)
        run_method(method_id, None, test, root / "runs" / method_id / "run")
    return root


def test_runs_are_read_without_importing_method_code(results):
    done = subprocess.run(
        [sys.executable, "-c", READ_RUNS, str(results)], capture_output=True, text=True, check=True
    )
    read = json.loads(done.stdout)
    assert read["methods"] == ["mean_intensity", "random_features"]
    assert len(set(read["rows"])) == 1 and read["rows"][0] > 0
    assert read["imported"] == []


def test_failed_runs_are_left_out(results):
    failed = results / "runs" / "hvg_pca" / "failed"
    failed.mkdir(parents=True)
    (failed / "FAILED.json").write_text("{}", encoding="utf-8")
    assert [run.card.method_id for run in finished_runs(results)] == [
        "mean_intensity",
        "random_features",
    ]
