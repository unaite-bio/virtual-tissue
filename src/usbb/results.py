"""Finished runs, read for evaluation from their files and cards; method code is never imported."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from usbb.card import MethodCard, card_by_id


@dataclass(frozen=True)
class Run:
    """One published run: its method card, its RUN.json record and the path of its output.h5ad."""

    card: MethodCard
    record: dict
    output: Path


def finished_runs(results) -> list[Run]:
    """Every published run under <results>/runs; failed or unfinished runs have no RUN.json there."""
    runs = []
    for record_path in sorted(Path(results).glob("runs/*/*/output/RUN.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        runs.append(
            Run(card_by_id(record["method_id"]), record, record_path.with_name("output.h5ad"))
        )
    return runs
