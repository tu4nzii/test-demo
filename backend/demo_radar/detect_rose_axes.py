"""
Rose / polar-area chart axis label detection.

Wraps the radar detection pipeline (detect_radar_axes.py) with rose-specific
adaptations:
  - Rose charts often place numeric values INSIDE wedges; these are not axis
    labels and should be filtered.
  - Labels may be at different radial positions than radar charts.
  - Wind rose / compass charts use directional abbreviations.

Public API:
    detect_rose(image_path, center, outer_radius, use_llm=True) -> (labels, debug)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.demo_radar.detect_radar_axes import detect as detect_radar


def detect_rose(
    image_path, center, outer_radius, use_llm=True
) -> Tuple[Dict[int, str], Dict]:
    """Detect axis labels on a rose/polar-area chart.

    Currently delegates to the radar detection pipeline.  Rose-specific
    adaptations (wedge-value filtering, label-position adjustment) will be
    added as failure patterns are identified.
    """
    return detect_radar(image_path, center, outer_radius, use_llm=use_llm)
