"""Shared file and dataset helpers for chart prediction scripts."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable


PathLike = str | os.PathLike[str]
ConfigTransform = Callable[[dict[str, Any], Path, Path], dict[str, Any] | None]


def safe_filename(name: str) -> str:
    """Return a Windows-safe filename fragment."""
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def ensure_dir(path: PathLike) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_json_files(config_dir: PathLike, recursive: bool = False, exclude_emu: bool = False) -> list[Path]:
    root = Path(config_dir)
    if not root.exists():
        return []

    pattern = "**/*.json" if recursive else "*.json"
    files: list[Path] = []
    for path in root.glob(pattern):
        rel_parts = [part.lower() for part in path.relative_to(root).parts]
        if exclude_emu and any("emu" in part for part in rel_parts):
            continue
        files.append(path)
    return sorted(files)


def load_json_configs(
    config_dir: PathLike,
    *,
    recursive: bool = False,
    exclude_emu: bool = False,
    transform: ConfigTransform | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Load chart config JSON files in a deterministic order.

    ``transform`` receives ``(config, file_path, config_root)`` and can mutate
    and return the config. Returning ``None`` skips that file.
    """
    root = Path(config_dir)
    if not root.exists():
        if verbose:
            print(f"Warning: config directory does not exist: {root}")
        return []

    configs: list[dict[str, Any]] = []
    for path in iter_json_files(root, recursive=recursive, exclude_emu=exclude_emu):
        try:
            config = read_json(path)
        except json.JSONDecodeError as exc:
            if verbose:
                print(f"Warning: skip invalid JSON {path}: {exc}")
            continue
        if transform:
            config = transform(config, path, root)
            if config is None:
                continue
        configs.append(config)
        if verbose:
            print(f"Loaded config: {path.name}")
    return configs


def filter_chart_configs(configs: Iterable[dict[str, Any]], chart_ids: Iterable[str] | None) -> list[dict[str, Any]]:
    wanted = set(chart_ids or [])
    if not wanted:
        return list(configs)
    return [cfg for cfg in configs if cfg.get("chart_id") in wanted]


def image_to_base64(image_path: PathLike) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def image_to_data_url(image_path: PathLike, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{image_to_base64(image_path)}"
