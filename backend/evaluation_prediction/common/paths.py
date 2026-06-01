"""Filesystem roots for backend-local prediction/evaluation flows."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


ASSETS_ROOT = _path_from_env(
    "EVALUATION_PREDICTION_ASSETS_ROOT",
    PROJECT_ROOT / "prediction_core" / "assets",
)
RESULTS_ROOT = _path_from_env(
    "EVALUATION_PREDICTION_RESULTS_ROOT",
    BACKEND_ROOT / "evaluation_prediction" / "results",
)
