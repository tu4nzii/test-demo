# Grid generation full-process current run

- Total images: 542
- Problem images: 8
- Output dir: `backend\evaluation\recheck_outputs\current_grid_full`
- All CSV: `backend\evaluation\results\grid_full_current_all.csv`
- Problem CSV: `backend\evaluation\results\grid_full_current_problems.csv`
- Run log: `backend\evaluation\results\grid_full_current_run.log`

Problem rule: this list only contains grid-generation flow problems: process failure, missing output, unknown axis type, too few numeric ticks, or encrypted tick count not matching the model-decided axis type.
Model-vs-JSON axis differences are recorded in the CSV but are not counted as grid-generation problems.

## Problem Reason Counts

- `process_failed`: 8

## Model Axis Differences From JSON

- `x_axis_type_diff_from_json`: 16
- `y_axis_type_diff_from_json`: 1

## h_bar (6)

| file | problem | x type | y type | encrypted image | ticks json | error |
| --- | --- | --- | --- | --- | --- | --- |
| `h_bar_057.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `h_bar_117.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `h_bar_144.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `h_bar_147.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `h_bar_150.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `h_bar_174.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |

## v_bar (2)

| file | problem | x type | y type | encrypted image | ticks json | error |
| --- | --- | --- | --- | --- | --- | --- |
| `v_bar_118.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
| `v_bar_182.png` | `process_failed` | unknown | unknown | `` | `` | `not_enough_tick_lines` |
