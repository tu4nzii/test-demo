# Prediction Core Reference

This directory is archived reference code. Runtime backend/frontend code must not
import from `reference/`. The active runtime prediction package is:

```text
backend/evaluation_prediction/
```

Keep this folder only for comparing older prompt, parser, geometry, and runner
ideas.

## Current Project Boundary

- `main` is the source of truth for cartesian grid restoration and encryption.
- Dataset GT must not be used to generate runtime grids, ticks, labels, colors, or predictions.
- The current runtime chart registry does not expose `v_stacked_bar` or `h_stacked_bar`.
- Use `backend/evaluation_prediction/common/model_config.py` for active model settings.

## Historical Supported Types

The archived reference code organized:

```text
cartesian: v_bar, h_bar, line, scatter, bubble
polar:     pie, donut, rose, radar
```

## Model Note

The active project currently uses `gemini-2.5-flash-lite` as the default Gemini
profile. Do not rely on old model names in archived scripts.
