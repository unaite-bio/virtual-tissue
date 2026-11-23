"""Executors run the launcher's tasks; each has one method, run(tasks), that returns when all end."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    """One method step: its command, the variables added to its environment, its log, its resources."""

    name: str
    command: list[str]
    env: dict[str, str]
    log: Path
    resources: dict


class LocalExecutor:
    """Runs tasks one after the other as subprocesses of this machine."""

    def run(self, tasks) -> None:
        for task in tasks:
            print(f"running {task.name}, log in {task.log}", flush=True)
            with open(task.log, "w", encoding="utf-8") as log:
                subprocess.run(
                    task.command,
                    env=dict(os.environ, **task.env),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
