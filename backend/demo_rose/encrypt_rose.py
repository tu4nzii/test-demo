"""
Rose chart grid encryption.

Reuses the radar chart encryption pipeline (RadarChartEncoder) directly.
Rose charts have the same concentric-circle structure as radar charts;
the circle detection and grid overlay logic works unchanged.

Public API:
    from backend.demo_rose.encrypt_rose import encrypt_rose
    from backend.demo_radar.encrypt_radar import RadarChartEncoder
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export the radar encoder for rose charts
from backend.demo_radar.encrypt_radar import RadarChartEncoder  # noqa: F401


def encrypt_rose(image_path, output_dir=None):
    """Encrypt a rose chart with grid overlay.  Wraps RadarChartEncoder."""
    encoder = RadarChartEncoder()
    return encoder.process_single_image(image_path, output_dir)
