# Real Radar/Rose Evaluation Steps

## Inputs

- Radar: `D:\home work\Agent.paper\NEW\test-demo\backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final`
- Rose: `D:\home work\Agent.paper\NEW\test-demo\backend\real\RadarChart-18 & RoseChart-6\RoseChart-6`

## Commands

- Encrypt and assemble all real radar/rose charts:
  `python backend/real_polar_batch.py --chart-type all --skip-evaluation`
- Encrypt one radar chart for smoke testing:
  `python backend/real_polar_batch.py --chart-type radar --only RadarChart24 --skip-evaluation --force`
- Run the default flow, including one radar evaluator pass:
  `python backend/real_polar_batch.py --chart-type all --force`

## Outputs

- Radar encrypted images and JSON: `D:\home work\Agent.paper\NEW\test-demo\data\output\real_radar`
- Rose encrypted images and JSON: `D:\home work\Agent.paper\NEW\test-demo\data\output\real_rose`
- Evaluator input JSON files are under each output directory's `result/` folder.

## Current Run Summary

- Radar: 10/10 encrypted and assembled.
- Rose: 0/0 encrypted and assembled.

## Failed Charts

- None in the latest recorded run.

## Notes

- Polygon radar charts are intentionally excluded: 1, 5, 6, 8, 16, 17, 18, and 23.
- The batch runner uses only trusted real-chart `center/r_ticks/r_pixels` metadata for encryption and does not use Hough-circle fallback.
- All tick labels are drawn; only inserted half-interval ticks receive dashed rings.
- Original files under `backend/real` are not overwritten.
