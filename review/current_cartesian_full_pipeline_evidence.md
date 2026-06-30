# Current Cartesian Full-Pipeline Evidence

This report verifies that the Cartesian evidence used in the reviewer response comes from the current full pipeline, not from the legacy axis/tick scanning path.

## Pipeline Evidence

- Latest recheck directory: `F:\program\test-demo\backend\evaluation\recheck_outputs\vishintprompt_full_grid_encryption_latest`
- Priority decision files: 650
- Decisions containing all three sources (`combined_mask`, `tick_supplement`, `semantic_guide`) for both axes: 650
- Candidate grid binding files: 1809
- Final binding files: 1176
- Final selection files: 650
- Grid status report files: 650
- Actual failure/exit reports: 14
- MLLM arbitration used after score prefill: 12

Selection policies observed:

- `score_first_mllm_when_needed`: 650

Axis source choices:

| Source | X-axis choice count | Y-axis choice count |
| --- | ---: | ---: |
| `combined_mask` | 44 | 90 |
| `semantic_guide` | 232 | 204 |
| `tick_supplement` | 374 | 356 |

## Full-Pipeline Cartesian Metrics

| Dataset | Type | Samples | Processed | Tick MAE(px) | Tick Acc@2px | Label Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Final-RealDataset | bubble | 9 | 9 | 0.482 | 99.12% | 96.49% |
| Final-RealDataset | h_bar | 12 | 11 | 1.868 | 80.88% | 75.25% |
| Final-RealDataset | line | 23 | 21 | 0.960 | 92.03% | 97.23% |
| Final-RealDataset | scatter | 9 | 7 | 0.814 | 98.04% | 100.00% |
| Final-RealDataset | v_bar | 21 | 19 | 0.806 | 94.19% | 95.86% |
| Sy.Dataset | bubble | 50 | 50 | 0.874 | 95.40% | 99.40% |
| Sy.Dataset | h_bar | 50 | 49 | 0.771 | 98.09% | 93.46% |
| Sy.Dataset | line | 50 | 50 | 0.564 | 96.76% | 98.01% |
| Sy.Dataset | scatter | 50 | 50 | 0.442 | 98.60% | 100.00% |
| Sy.Dataset | v_bar | 51 | 51 | 0.462 | 98.78% | 97.20% |

Overall Cartesian full-pipeline summary:

- Samples: 325
- Processed: 317
- Tick MAE: 0.691 px
- Tick Acc@2px: 96.37%
- Tick position MAE: 0.849 px
- Label accuracy: 96.13%

## Interpretation

The parameter sensitivity sweep in this review folder is only a legacy low-level Canny/Hough candidate-generator diagnostic. The final Cartesian results above come from the active enhanced-grid-first runtime artifacts: three candidate grids are scored, unreliable cases can produce failure/exit reports, and evaluation reads generated final bindings.
