"""The method step: refuse or run one method on a bundle, then publish its output atomically."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import time
import traceback
from importlib.metadata import distributions
from pathlib import Path

import numpy as np

from usbb.card import card_by_id
from usbb.io import read_input, write_output
from usbb.method import load_method

PACKAGE = Path(__file__).parent
CORE_FILES = ("card.py", "method.py", "io.py", "run.py", "patch_encoder.py")


class InputRefused(Exception):
    """The method cannot run on this input; raised before anything is loaded."""


def _sha256(path, text=False) -> str:
    """sha256 of a file; text is hashed with LF line endings so Windows and Linux checkouts agree."""
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_hash(root, files, text=False) -> str:
    lines = sorted(f"{file.relative_to(root).as_posix()} {_sha256(file, text)}" for file in files)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def folder_hash(folder, text=False) -> str:
    """sha256 over a folder's files and their relative paths; also names the input bundles."""
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return _files_hash(folder, files, text)


def run_key(card, params, seed, train_dir, test_dir, env_lock=None) -> str:
    """sha256 of everything that decides the output: code, card, params, seed, inputs, weights, env."""
    core = [PACKAGE / name for name in CORE_FILES] + sorted((PACKAGE / "api").glob("*.yaml"))
    record = {
        "method_id": card.method_id,
        "card": _sha256(card.path, text=True),
        "core": _files_hash(PACKAGE, [file for file in core if file.exists()], text=True),
        "method_folder": folder_hash(card.path.parent, text=True),
        "params": dict(card.parameters, **(params or {})),
        "seed": seed,
        # Only fitted methods read the train bundle, so only they depend on it.
        "train": folder_hash(train_dir) if card.fit_mode == "fitted" and train_dir else None,
        "test": folder_hash(test_dir),
        "weights": (card.weights or {}).get("files", {}),
        "env_lock": _sha256(env_lock, text=True) if env_lock else None,
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def is_complete(run_dir, key) -> bool:
    """True when run_dir holds a published output made with this run key."""
    record = Path(run_dir) / "output" / "RUN.json"
    return record.exists() and json.loads(record.read_text(encoding="utf-8"))["run_key"] == key


def _refusals(card, method, inp) -> list:
    """Reasons to refuse one bundle: the card's units, modalities, value scales and patch geometry."""
    reasons = []
    if inp.unit_type not in card.units:
        reasons.append(f"unit type {inp.unit_type} is not in the card's units {card.units}")
    for name, spec in card.inputs.items():
        adata = inp.modalities.get(name)
        if spec["kind"] == "embedding":
            if not any("method_id" in a.uns for a in inp.modalities.values()):
                reasons.append(f"{name}: the bundle holds no upstream method output")
        elif adata is None:
            reasons.append(f"missing modality {name} ({name}.h5ad)")
        elif spec["kind"] == "matrix" and adata.uns["value_scale"] != spec["value_scale"]:
            reasons.append(
                f"{name}: value scale {adata.uns['value_scale']}, the card needs {spec['value_scale']}"
            )
        elif spec["kind"] == "image":
            size, um_per_px = adata.uns["patch"]["size_px"], adata.uns["patch"]["um_per_px"]
            if (size, um_per_px) != (spec["patch_px"], spec["um_per_px"]):
                reasons.append(
                    f"{name}: patches of {size} px at {um_per_px} um_per_px,"
                    f" the card needs {spec['patch_px']} px at {spec['um_per_px']} um_per_px"
                )
    # method.check may rely on a bundle that already satisfies the card.
    return reasons or list(method.check(inp))


def _seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _state_digest(method) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(method.state().items()):
        array = np.ascontiguousarray(value)
        digest.update(f"{name} {array.dtype} {array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checked_latents(card, inp, latents) -> dict:
    """The latents as float32, after checking names, row count and finiteness against the card."""
    expected, problems, arrays = sorted(output.latent for output in card.outputs), [], {}
    if sorted(latents) != expected:
        problems.append(f"latents {sorted(latents)}, the card declares {expected}")
    for name in sorted(set(latents) & set(expected)):
        array = arrays[name] = np.asarray(latents[name], dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(inp.unit_ids):
            problems.append(f"{name} has shape {array.shape}, expected ({len(inp.unit_ids)}, d)")
        elif not np.isfinite(array).all():
            problems.append(f"{name} has non-finite values")
    if problems:
        raise ValueError("output check failed: " + "; ".join(problems))
    return arrays


def _informed_by(card, inp) -> dict:
    """Modalities behind each latent; an embedding input stands for those of the upstream outputs."""
    upstream = [a.uns["informed_by"] for a in inp.modalities.values() if "method_id" in a.uns]
    upstream_modalities = [m for informed_by in upstream for ms in informed_by.values() for m in ms]
    expand = {
        name: upstream_modalities if spec["kind"] == "embedding" else [name]
        for name, spec in card.inputs.items()
    }
    return {
        o.latent: sorted({m for name in o.informed_by for m in expand[name]}) for o in card.outputs
    }


def run_method(
    method_id, train_dir, test_dir, run_dir, params=None, seed=0, model_dir=None, env_lock=None
) -> Path:
    """Run one method on a test bundle and publish <run_dir>/output/; return its output.h5ad."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output, staging = run_dir / "output", run_dir / f"output.tmp-{os.getpid()}"
    started, key = time.time(), None
    try:
        card = card_by_id(method_id)
        method = load_method(card, params, seed, model_dir)
        bundles = {"test": read_input(test_dir)}
        if card.fit_mode == "fitted" and train_dir:
            bundles["train"] = read_input(train_dir)
        reasons = [
            f"{name}: {reason}"
            for name, inp in bundles.items()
            for reason in _refusals(card, method, inp)
        ]
        if card.fit_mode == "fitted" and "train" not in bundles:
            reasons.append("a fitted method needs a train bundle")
        if reasons:
            raise InputRefused(f"{method_id} refuses its input: " + "; ".join(reasons))
        key = run_key(card, params, seed, train_dir, test_dir, env_lock)
        if is_complete(run_dir, key):
            return output / "output.h5ad"
        _seed(seed)
        method.load()
        frozen_state = _state_digest(method) if card.fit_mode == "frozen" else None
        if card.fit_mode != "frozen":
            method.fit(bundles["train" if card.fit_mode == "fitted" else "test"])
        test = bundles["test"]
        latents = _checked_latents(card, test, method.transform(test))
        if frozen_state is not None and _state_digest(method) != frozen_state:
            raise RuntimeError(f"{method_id} is frozen but its state() changed during the run")
        first = next(iter(test.modalities.values()))
        uns = {
            "dataset_id": first.uns["dataset_id"],
            "method_id": method_id,
            "normalization_id": "+".join(
                sorted({a.uns["normalization_id"] for a in test.modalities.values()})
            ),
            "unit_type": test.unit_type,
            "run_key": key,
            "informed_by": _informed_by(card, test),
        }
        record = {
            "run_key": key,
            "method_id": method_id,
            "fit_mode": card.fit_mode,
            "transductive": card.fit_mode == "transductive",
            "params": method.params,
            "seed": seed,
            "input_train": str(train_dir) if train_dir else None,
            "input_test": str(test_dir),
            "model_dir": str(model_dir) if model_dir else None,
            "env_lock": str(env_lock) if env_lock else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "wall_seconds": round(time.time() - started, 3),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        }
        staging.mkdir(exist_ok=True)
        write_output(staging / "output.h5ad", test, latents, uns)
        (staging / "RUN.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        failure = {"method_id": method_id, "run_key": key, "error": type(error).__name__}
        failure.update(message=str(error), traceback=traceback.format_exc())
        (run_dir / "FAILED.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    (run_dir / "FAILED.json").unlink(missing_ok=True)
    return output / "output.h5ad"
