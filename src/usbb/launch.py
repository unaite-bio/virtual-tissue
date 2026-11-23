"""usbb launch: run each method of an experiment file that has no complete run yet."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import tomllib
import yaml

from usbb.card import CardError, MethodCard, card_by_id
from usbb.data.process import prepare
from usbb.executors import LocalExecutor, Task
from usbb.run import folder_hash, is_complete, run_key
from usbb.weights import fetch_weights

REPO = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO / "usbb.local.toml"
GROUPS = REPO / "envs" / "groups.yaml"
REQUIRED_KEYS = ("dataset", "unit", "methods")
OPTIONAL_KEYS = ("split", "seed")
LOCAL_KEYS = ("results", "executor", "groups")
TASK_ENV = {"PYTHONPATH": str(REPO / "src"), "HF_HUB_OFFLINE": "1"}


@dataclass
class Step:
    """One method of an experiment: card, parameters, upstream method ids and environment."""

    card: MethodCard
    params: dict
    inputs: list[str]
    python: list[str]  # starts Python in the card's environment group
    lock: Path


def _steps(experiment, stores, config) -> list[Step]:
    """The methods of an experiment in file order; raise one ValueError listing every problem."""
    groups = yaml.safe_load(GROUPS.read_text(encoding="utf-8"))
    pythons = {group: command + ["python"] for group, command in config.get("groups", {}).items()}
    pythons["core"] = [sys.executable]
    problems = [f"missing key {key}" for key in REQUIRED_KEYS if key not in experiment]
    problems += [f"unknown key {k}" for k in experiment if k not in REQUIRED_KEYS + OPTIONAL_KEYS]
    problems += [f"unknown key {key} in usbb.local.toml" for key in config if key not in LOCAL_KEYS]
    if not stores:
        problems.append(f"dataset {experiment.get('dataset')} holds no .zarr store")
    steps, unconfigured = [], set()
    for entry in experiment.get("methods", []):
        method_id, params = (entry, {}) if isinstance(entry, str) else next(iter(entry.items()))
        params = dict(params or {})
        inputs = params.pop("inputs", [])
        try:
            card = card_by_id(method_id)
        except CardError as error:
            problems.append(str(error))
            continue
        except FileNotFoundError:
            problems.append(f"unknown method {method_id}")
            continue
        listed = [step.card.method_id for step in steps]
        fuser = any(spec["kind"] == "embedding" for spec in card.inputs.values())
        problems += [
            f"{method_id} has no parameter {k}" for k in params if k not in card.parameters
        ]
        if method_id in listed:
            problems.append(f"{method_id} is listed twice")
        if experiment.get("unit") not in card.units:
            problems.append(f"{method_id} does not take {experiment.get('unit')} units")
        if fuser and card.fit_mode == "fitted":
            problems.append(
                f"{method_id}: fitted fusers are not supported, only frozen or transductive"
            )
        elif card.fit_mode == "fitted" and "split" not in experiment:
            problems.append(f"{method_id} is fitted, so the experiment needs a split")
        if fuser != bool(inputs) or not set(inputs) <= set(listed):
            problems.append(
                f"{method_id}: only fusers take inputs, and they must name methods listed above"
            )
        lock = REPO / groups[card.env_group]["lock"] if card.env_group in groups else None
        if lock is None:
            problems.append(f"{method_id}: env group {card.env_group} is not in envs/groups.yaml")
        elif not lock.is_file():
            problems.append(
                f"{method_id}: env group {card.env_group} has no lock file {lock.relative_to(REPO)}"
            )
        if card.env_group not in pythons:
            unconfigured.add(card.env_group)
        steps.append(Step(card, params, inputs, pythons.get(card.env_group), lock))
    if unconfigured:
        problems.append(f"usbb.local.toml has no [groups] command for {sorted(unconfigured)}")
    if problems:
        raise ValueError("experiment refused:\n- " + "\n- ".join(problems))
    return steps


def _upstream_bundle(outputs, results) -> Path:
    """Copy upstream outputs in as <method_id>.h5ad, in a bundle folder named by its content."""
    staging = results / "bundles" / f"tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for method_id, path in outputs.items():
        shutil.copyfile(path, staging / f"{method_id}.h5ad")
    final = staging.parent / folder_hash(staging)[:16]
    if final.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(final)
    return final


def _task(step, seed, train, test, model, run_dir) -> Task:
    method_id = step.card.method_id
    command = step.python + ["-m", "usbb.cli", "run", "--method", method_id]
    command += ["--input_test", str(test), "--output", str(run_dir), "--seed", str(seed)]
    command += ["--env_lock", str(step.lock)]
    if train:
        command += ["--input_train", str(train)]
    if step.params:
        command += ["--params", json.dumps(step.params)]
    if model:
        command += ["--model", str(model)]
    return Task(method_id, command, TASK_ENV, run_dir / "log.txt", step.card.resources)


def _executor(spec):
    """LocalExecutor, or the class that usbb.local.toml names as "<file>:<class>"."""
    if spec is None:
        return LocalExecutor()
    file, class_name = spec.split(":")
    module_spec = importlib.util.spec_from_file_location(Path(file).stem, REPO / file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)()


def launch_experiment(path) -> bool:
    """Run every method of an experiment file without a complete run; True if all are complete."""
    config = {}
    if LOCAL_CONFIG.exists():
        config = tomllib.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stores = sorted((REPO / experiment.get("dataset", "")).glob("*.zarr"))
    steps = _steps(experiment, stores, config)
    results = REPO / config.get("results", "results/usbb")
    executor = _executor(config.get("executor"))
    unit, split, seed = experiment["unit"], experiment.get("split"), experiment.get("seed", 0)
    outputs, rows, pending = {}, [], steps
    while pending:
        waiting = {step.card.method_id for step in pending}
        wave = [step for step in pending if not set(step.inputs) & waiting]
        pending = [step for step in pending if step not in wave]
        tasks, started = [], []
        for step in wave:
            card = step.card
            missing = [method_id for method_id in step.inputs if method_id not in outputs]
            if missing:
                rows.append((card.method_id, "blocked", f"no output from {missing}"))
                continue
            if step.inputs:
                train = None
                test = _upstream_bundle({i: outputs[i] for i in step.inputs}, results)
            else:
                train, test = prepare(stores, card, unit, split, results)
            key = run_key(card, step.params, seed, train, test, step.lock)
            run_dir = results / "runs" / card.method_id / key[:16]
            if is_complete(run_dir, key):
                outputs[card.method_id] = run_dir / "output" / "output.h5ad"
                rows.append((card.method_id, "cached", run_dir.relative_to(results)))
                continue
            model = fetch_weights(card.weights) if card.weights else None
            run_dir.mkdir(parents=True, exist_ok=True)
            tasks.append(_task(step, seed, train, test, model, run_dir))
            started.append((card.method_id, run_dir, key))
        executor.run(tasks)
        for method_id, run_dir, key in started:
            if is_complete(run_dir, key):
                outputs[method_id] = run_dir / "output" / "output.h5ad"
                rows.append((method_id, "done", run_dir.relative_to(results)))
            else:
                rows.append((method_id, "failed", (run_dir / "log.txt").relative_to(results)))
    for method_id, status, where in rows:
        print(f"{method_id:<24} {status:<8} {where}")
    print(f"{len(outputs)} of {len(rows)} methods complete; runs in {results}")
    return len(outputs) == len(rows)
