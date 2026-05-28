# Current prompt recheck problem charts

- Input list: `backend\evaluation\results\problem_charts_current_flow_actionable.csv`
- Recheck output dir: `backend\evaluation\recheck_outputs\current_prompt_actionable`
- Total rechecked: 152
- Remaining problematic: 15

Reason counts:

- `process_failed`: 7
- `x_axis_type_wrong`: 7
- `y_axis_type_wrong`: 1

Failure note: `process_failed` means the CV stage did not detect enough tick lines, so no processed effect image was generated for that chart.

## h_bar (7)

| file | new reason | error | x pred/gt | y pred/gt | encrypted image | ticks json |
| --- | --- | --- | --- | --- | --- | --- |
| `h_bar_000.png` | `y_axis_type_wrong` | `` | numeric/numeric | numeric/text | `backend\evaluation\recheck_outputs\current_prompt_actionable\h_bar\h_bar_000_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\h_bar\h_bar_000_ticks.json` |
| `h_bar_057.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |
| `h_bar_117.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |
| `h_bar_144.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |
| `h_bar_147.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |
| `h_bar_150.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |
| `h_bar_174.png` | `process_failed` | `not_enough_tick_lines` | unknown/numeric | unknown/text | `` | `` |

## line (7)

| file | new reason | error | x pred/gt | y pred/gt | encrypted image | ticks json |
| --- | --- | --- | --- | --- | --- | --- |
| `line_008.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_008_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_008_ticks.json` |
| `line_011.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_011_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_011_ticks.json` |
| `line_013.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_013_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_013_ticks.json` |
| `line_043.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_043_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_043_ticks.json` |
| `line_065.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_065_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_065_ticks.json` |
| `line_077.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_077_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_077_ticks.json` |
| `line_111.png` | `x_axis_type_wrong` | `` | numeric/text | numeric/numeric | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_111_with_grid.png` | `backend\evaluation\recheck_outputs\current_prompt_actionable\line\line_111_ticks.json` |

## v_bar (1)

| file | new reason | error | x pred/gt | y pred/gt | encrypted image | ticks json |
| --- | --- | --- | --- | --- | --- | --- |
| `v_bar_118.png` | `process_failed` | `not_enough_tick_lines` | unknown/text | unknown/numeric | `` | `` |
