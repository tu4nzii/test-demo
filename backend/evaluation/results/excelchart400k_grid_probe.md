# ExcelChart400k Grid Encryption Evaluation

- CSV: `F:\program\test-demo\backend\evaluation\results\excelchart400k_grid_probe.csv`
- Output root: `F:\program\test-demo\backend\evaluation\recheck_outputs\excelchart400k_grid`
- Included datasets: `bardata` and `linedata` only; `pie` has no Cartesian grid, `cls` is classification/duplicate-style metadata.

## Summary

| chart_type | total | success | effect_ok | problems | marked_recovery |
|---|---:|---:|---:|---:|---:|
| h_bar | 3 | 0 | 0 | 3 | 0 |
| line | 5 | 5 | 5 | 0 | 0 |
| v_bar | 2 | 1 | 1 | 1 | 0 |

## Problems

| chart_id | chart_type | split | problem_type | output |
|---|---|---|---|---|
| excel400k_bar_test2019_000001 | h_bar | test2019 | process_failed | `F:\program\test-demo\backend\evaluation\recheck_outputs\excelchart400k_grid\h_bar\test2019\excel400k_bar_test2019_000001_marked_ticks.png` |
| excel400k_bar_test2019_000002 | h_bar | test2019 | process_failed | `F:\program\test-demo\backend\evaluation\recheck_outputs\excelchart400k_grid\h_bar\test2019\excel400k_bar_test2019_000002_marked_ticks.png` |
| excel400k_bar_test2019_000003 | v_bar | test2019 | process_failed | `F:\program\test-demo\backend\evaluation\recheck_outputs\excelchart400k_grid\v_bar\test2019\excel400k_bar_test2019_000003_marked_ticks.png` |
| excel400k_bar_test2019_000004 | h_bar | test2019 | process_failed | `F:\program\test-demo\backend\evaluation\recheck_outputs\excelchart400k_grid\h_bar\test2019\excel400k_bar_test2019_000004_marked_ticks.png` |
