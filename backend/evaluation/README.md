# Evaluation Assets

This directory keeps evaluation code and artifacts out of the backend runtime root.

- `scripts/`: standalone evaluation scripts.
- `results/`: generated JSON metrics and sampled evaluation records.
- `reports/`: human-readable notes, reviewer responses, and summaries.
- `archive/`: older or superseded evaluation notes kept for traceability.

MLLM-based tick/axis-label evaluation is available through:

```powershell
python backend\evaluation\scripts\evaluate_cv_mllm_ticks.py --types line scatter bubble v_bar h_bar --limit 20 --cache-only
```

Drop `--cache-only` only when you intentionally want to call the MLLM and populate missing cache entries. Cache keys include the dataset id, image content hash, prompt signature, model, temperature, and cache schema version, so repeated runs reuse cached MLLM output only when the dataset image and prompt configuration are unchanged.

The broader reviewer evaluation can include this MLLM stage with:

```powershell
python backend\evaluation\scripts\evaluate_axis_prior_reviewer_questions.py --include-mllm --cache-only --mllm-limit 20
```

Runtime evaluation helpers used by the API remain in this directory root:

- `metrics.py`
- `normalizer.py`
- `service.py`
