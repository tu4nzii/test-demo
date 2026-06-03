"""Visual feedback and crop generation for bubble charts.

Bubble and scatter prediction use the same point-chart visual feedback
mechanics. Keep the implementation shared so raw_crops/tempy behavior does not
drift between the two chart types.
"""

from __future__ import annotations

from ..scatter.visual import (  # noqa: F401
    chart_result_dir,
    crop_draw_ticks_resize,
    draw_crosshair,
    draw_prediction_overlay,
    generate_expanded_crop_with_grid_by_diameter,
    raw_crop_dir,
    temp_dir,
)
