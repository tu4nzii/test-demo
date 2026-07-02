"""Model-aware visual budgets for amplifier images.

The registry does not change the paper mechanism. It only tunes the visual
input budget used inside the amplifier stage so each MLLM gets enough readable
local evidence without blindly sending oversized crops.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import ceil

from .model_config import get_model_name


@dataclass(frozen=True)
class ModelVisionProfile:
    key: str
    aliases: tuple[str, ...]
    target_side: int
    point_target_side: int
    ruler_target_side: int
    sector_target_side: int
    polar_target_side: int
    max_side: int
    label_pad: int
    half_window_schedule: tuple[float, float, float]
    tick_divisors: tuple[int, int, int]
    point_grid_density: int
    polar_window_factors: tuple[float, float, float]
    polar_min_window: int
    image_token_family: str
    note: str


DEFAULT_PROFILE = ModelVisionProfile(
    key="default",
    aliases=("default",),
    target_side=768,
    point_target_side=768,
    ruler_target_side=640,
    sector_target_side=768,
    polar_target_side=768,
    max_side=1024,
    label_pad=64,
    half_window_schedule=(1.65, 1.15, 0.85),
    tick_divisors=(4, 6, 8),
    point_grid_density=2,
    polar_window_factors=(0.62, 0.48, 0.38),
    polar_min_window=160,
    image_token_family="unknown",
    note="Balanced default for unknown MLLMs.",
)


MODEL_VISION_PROFILES: tuple[ModelVisionProfile, ...] = (
    ModelVisionProfile(
        key="gpt-4o",
        aliases=("gpt-4o", "gpt4o", "openai-gpt-4o"),
        target_side=768,
        point_target_side=768,
        ruler_target_side=640,
        sector_target_side=768,
        polar_target_side=768,
        max_side=1024,
        label_pad=64,
        half_window_schedule=(1.55, 1.10, 0.82),
        tick_divisors=(4, 6, 8),
        point_grid_density=2,
        polar_window_factors=(0.58, 0.45, 0.36),
        polar_min_window=150,
        image_token_family="openai_512_tile",
        note="Strong OCR and spatial reasoning; keep crops compact after round 1.",
    ),
    ModelVisionProfile(
        key="gemini-2.5-flash",
        aliases=(
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-nothinking",
            "gemini25flash",
            "gemini25flashlite",
            "gemini25flashnothinking",
        ),
        target_side=768,
        point_target_side=768,
        ruler_target_side=640,
        sector_target_side=768,
        polar_target_side=768,
        max_side=768,
        label_pad=64,
        half_window_schedule=(1.65, 1.15, 0.85),
        tick_divisors=(4, 6, 8),
        point_grid_density=2,
        polar_window_factors=(0.62, 0.48, 0.38),
        polar_min_window=160,
        image_token_family="gemini_768_tile",
        note="Balanced speed/vision profile; preserve current validated defaults.",
    ),
    ModelVisionProfile(
        key="claude-haiku-4.5",
        aliases=("claude-haiku-4.5", "claude-haiku-45", "claude-4.5-haiku"),
        target_side=640,
        point_target_side=640,
        ruler_target_side=576,
        sector_target_side=640,
        polar_target_side=640,
        max_side=896,
        label_pad=64,
        half_window_schedule=(1.75, 1.25, 0.95),
        tick_divisors=(3, 5, 6),
        point_grid_density=1,
        polar_window_factors=(0.68, 0.52, 0.42),
        polar_min_window=170,
        image_token_family="claude_28_patch",
        note="Latency-oriented model; prefer less dense guides and fewer image tokens.",
    ),
    ModelVisionProfile(
        key="internvl3-78b",
        aliases=("internvl3-78b", "intern-vl3-78b", "intern vl3-78b", "internvl-3-78b"),
        target_side=896,
        point_target_side=896,
        ruler_target_side=672,
        sector_target_side=896,
        polar_target_side=896,
        max_side=896,
        label_pad=112,
        half_window_schedule=(1.80, 1.30, 1.00),
        tick_divisors=(3, 4, 6),
        point_grid_density=1,
        polar_window_factors=(0.72, 0.56, 0.45),
        polar_min_window=190,
        image_token_family="internvl_448_tile",
        note="Open-source large VLM; larger text/crop helps OCR, but guide density stays moderate.",
    ),
    ModelVisionProfile(
        key="pixtral-12b-2409",
        aliases=("pixtral-12b-2409", "pixtral-12b", "pixtral"),
        target_side=512,
        point_target_side=640,
        ruler_target_side=512,
        sector_target_side=640,
        polar_target_side=640,
        max_side=768,
        label_pad=64,
        half_window_schedule=(1.90, 1.35, 1.00),
        tick_divisors=(3, 4, 5),
        point_grid_density=1,
        polar_window_factors=(0.74, 0.58, 0.46),
        polar_min_window=190,
        image_token_family="pixtral_16_patch",
        note="Smaller VLM; avoid dense red guides and oversized image token budgets.",
    ),
)


def active_model_vision_profile() -> ModelVisionProfile:
    override = os.getenv("CHART_AMPLIFIER_VISION_PROFILE", "").strip()
    model_name = override or get_model_name()
    profile = profile_for_model(model_name)
    return _apply_env_overrides(profile)


def profile_for_model(model_name: str | None) -> ModelVisionProfile:
    normalized = _normalize(model_name or "")
    if not normalized:
        return DEFAULT_PROFILE
    for profile in MODEL_VISION_PROFILES:
        if normalized == _normalize(profile.key):
            return profile
        if any(normalized == _normalize(alias) for alias in profile.aliases):
            return profile
        if any(_normalize(alias) in normalized for alias in profile.aliases):
            return profile
    return DEFAULT_PROFILE


def _apply_env_overrides(profile: ModelVisionProfile) -> ModelVisionProfile:
    return ModelVisionProfile(
        key=profile.key,
        aliases=profile.aliases,
        target_side=_env_int("CHART_AMPLIFIER_TARGET_SIDE", profile.target_side),
        point_target_side=_env_int("CHART_AMPLIFIER_POINT_TARGET_SIDE", profile.point_target_side),
        ruler_target_side=_env_int("CHART_AMPLIFIER_RULER_TARGET_SIDE", profile.ruler_target_side),
        sector_target_side=_env_int("CHART_AMPLIFIER_SECTOR_TARGET_SIDE", profile.sector_target_side),
        polar_target_side=_env_int("CHART_AMPLIFIER_POLAR_TARGET_SIDE", profile.polar_target_side),
        max_side=_env_int("CHART_AMPLIFIER_MAX_SIDE", profile.max_side),
        label_pad=_env_int("CHART_AMPLIFIER_LABEL_PAD", profile.label_pad),
        half_window_schedule=_env_float_tuple(
            "CHART_AMPLIFIER_WINDOW_SCHEDULE",
            profile.half_window_schedule,
            length=3,
        ),
        tick_divisors=_env_int_tuple(
            "CHART_AMPLIFIER_TICK_DIVISORS",
            profile.tick_divisors,
            length=3,
        ),
        point_grid_density=_env_int("CHART_AMPLIFIER_POINT_GRID_DENSITY", profile.point_grid_density),
        polar_window_factors=_env_float_tuple(
            "CHART_AMPLIFIER_POLAR_WINDOW_FACTORS",
            profile.polar_window_factors,
            length=3,
        ),
        polar_min_window=_env_int("CHART_AMPLIFIER_POLAR_MIN_WINDOW", profile.polar_min_window),
        image_token_family=profile.image_token_family,
        note=profile.note,
    )


def target_side_for_family(profile: ModelVisionProfile, family: str | None) -> int:
    normalized = (family or "").strip().casefold()
    if normalized in {"point", "scatter", "bubble"}:
        return profile.point_target_side
    if normalized in {"ruler", "bar", "line", "cartesian"}:
        return profile.ruler_target_side
    if normalized in {"sector", "pie", "donut", "circular"}:
        return profile.sector_target_side
    if normalized in {"polar", "radar", "rose"}:
        return profile.polar_target_side
    return profile.target_side


def estimate_image_tokens(
    width: int,
    height: int,
    *,
    model_name: str | None = None,
    profile: ModelVisionProfile | None = None,
) -> int | None:
    """Estimate image tokens from public model image-encoding rules.

    These estimates are for experiment audit and budget shaping. Provider-side
    accounting can still differ because gateways may resize or re-encode images.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    selected = profile or profile_for_model(model_name)
    family = selected.image_token_family
    if family == "openai_512_tile":
        # High-detail GPT-4o-style accounting: fit 2048, shortest side 768,
        # then bill 512px tiles plus base.
        scale = min(2048 / w, 2048 / h, 1.0)
        sw = w * scale
        sh = h * scale
        short = max(1.0, min(sw, sh))
        scale = 768 / short
        sw *= scale
        sh *= scale
        return 85 + 170 * ceil(sw / 512) * ceil(sh / 512)
    if family == "gemini_768_tile":
        if w <= 384 and h <= 384:
            return 258
        return 258 * ceil(w / 768) * ceil(h / 768)
    if family == "claude_28_patch":
        return ceil(w / 28) * ceil(h / 28)
    if family == "internvl_448_tile":
        return 256 * ceil(w / 448) * ceil(h / 448)
    if family == "pixtral_16_patch":
        patch_w = ceil(w / 16)
        patch_h = ceil(h / 16)
        return patch_w * patch_h + patch_h + 1
    return None


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().casefold())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_int_tuple(name: str, default: tuple[int, ...], *, length: int) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(max(1, int(part.strip())) for part in raw.split(",") if part.strip())
    except ValueError:
        return default
    return values if len(values) == length else default


def _env_float_tuple(name: str, default: tuple[float, ...], *, length: int) -> tuple[float, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(max(0.1, float(part.strip())) for part in raw.split(",") if part.strip())
    except ValueError:
        return default
    return values if len(values) == length else default
