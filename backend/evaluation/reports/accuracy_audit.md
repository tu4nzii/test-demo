# Accuracy Audit

## Key Finding

The previously reported high accuracy is not an end-to-end axis/tick extraction accuracy.

- Chart classification accuracy is high, but it only measures chart-type routing.
- `cv_success_rate` measures whether both axes were returned, not whether tick positions and tick values are correct.
- `x_pair_recall` / `y_pair_recall` are the closest available indicators for tick extraction because they require pixel position and tick value to match together.
- The original recall fields were averaged only over CV-success rows. The re-audited file adds all-sample recall fields that count CV failures as zero.

Source result:
`backend/evaluation/results/cv_mllm_tick_eval_gpt4omini_full.json`

Re-audited result:
`backend/evaluation/results/cv_mllm_tick_eval_gpt4omini_full_reaudited.json`

## Re-audited CV + MLLM Tick Metrics

| type | n | CV success | x pair recall, all | y pair recall, all | mean pair recall, all | complete pair accuracy |
|---|---:|---:|---:|---:|---:|---:|
| line | 100 | 100.00% | 15.53% | 9.87% | 12.70% | 0.00% |
| scatter | 99 | 79.80% | 21.32% | 29.85% | 25.59% | 0.00% |
| bubble | 100 | 84.00% | 32.33% | 41.00% | 36.67% | 0.00% |
| v_bar | 151 | 99.34% | 8.23% | 36.97% | 22.60% | 0.00% |
| h_bar | 92 | 90.22% | 15.87% | 5.72% | 10.80% | 0.00% |
| ALL | 542 | 91.51% | 17.71% | 26.11% | 21.91% | 0.00% |

## Interpretation

The honest statement is that the current CV module often returns axis candidates, but the full CV + MLLM tick pipeline is weak under exact or paired evaluation. The method should not claim high end-to-end accuracy from the chart-classification number or the both-axis return rate.

For the paper response, report the failure rate and fallback mechanism separately from the tick-value extraction quality. Use `mean_pair_recall_all` as a conservative headline metric unless a better endpoint metric is introduced.
