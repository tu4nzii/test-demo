"""Adapter for running registered chart module entry scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from reference.prediction_core.specs import ChartSpec


@dataclass(frozen=True)
class RunRequest:
    spec: ChartSpec
    chart_ids: list[str] | None = None
    batch_size: int | None = None
    dry_run: bool = False


def build_command(request: RunRequest) -> list[str]:
    command = [sys.executable, str(request.spec.relative_script)]
    if request.batch_size is not None:
        command.extend(["--batch-size", str(request.batch_size)])
    if request.chart_ids:
        command.append("--chart-ids")
        command.extend(request.chart_ids)
    return command


def prepare_env(env: dict[str, str] | None = None) -> dict[str, str]:
    prepared = dict(env or os.environ)
    prepared.setdefault("PYTHONUTF8", "1")
    prepared.setdefault("PYTHONIOENCODING", "utf-8")
    return prepared


def describe_request(request: RunRequest) -> str:
    command = build_command(request)
    lines = [
        f"[chart] {request.spec.chart_type} ({request.spec.coordinate_system})",
        f"[script] {request.spec.script}",
        f"[cwd] {request.spec.workdir}",
        f"[cmd] {' '.join(command)}",
    ]
    return "\n".join(lines)


def run_backend(request: RunRequest, env: dict[str, str] | None = None) -> int:
    print(describe_request(request))
    if request.dry_run:
        return 0
    return subprocess.call(build_command(request), cwd=request.spec.workdir, env=prepare_env(env))
