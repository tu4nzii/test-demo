"""Prompt builders for bubble charts.

Bubble and scatter prompts share the same point-chart reading contract; the
runner passes ``mark_name="bubble"`` so the prompt text remains bubble-specific.
"""

from __future__ import annotations

from ..scatter.prompts import build_point_exists_prompt, generate_prompt  # noqa: F401
