# Cartesian Latest Outlier Charts

Source: `F:\program\test-demo\backend\evaluation\recheck_outputs\cartesian_latest_full_results_20260528`
GT: `F:\program\test-demo\backend\charts`

Excluded samples: `v_bar/v_bar_050` (problematic source data, archived under `backend/evaluation/excluded_dataset_samples/v_bar/`).

## Axis Failures

| type | chart | has_pred | tick incorrect |
| --- | --- | --- | --- |
| h_bar | h_bar_018 | False | 40 |
| h_bar | h_bar_031 | False | 32 |
| h_bar | h_bar_117 | False | 30 |

## Tick MAE Extreme Contributors

| type | chart | mae sum | share % | mae | wrong ticks |
| --- | --- | --- | --- | --- | --- |
| scatter | scatter_015 | 99.6923 | 68.41 | 5.5385 | 8 |
| scatter | scatter_048 | 42.0000 | 28.82 | 2.3333 | 8 |
| v_bar | v_bar_089 | 3.5000 | 2.40 | 0.1400 | 7 |
| scatter | scatter_034 | 0.0400 | 0.03 | 0.0022 | 8 |
| scatter | scatter_001 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_054 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_064 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_007 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_014 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_027 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_087 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_006 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_009 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_013 | 0.0200 | 0.01 | 0.0012 | 6 |
| scatter | scatter_016 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_017 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_019 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_026 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_029 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_036 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_039 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_046 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_049 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_069 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_079 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_086 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_093 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_096 | 0.0200 | 0.01 | 0.0011 | 4 |
| scatter | scatter_099 | 0.0200 | 0.01 | 0.0011 | 4 |

## Tick Accuracy Low

| type | chart | wrong/total | accuracy % | mae sum |
| --- | --- | --- | --- | --- |
| h_bar | h_bar_018 | 40/40 | 0.00 | 0.0000 |
| h_bar | h_bar_031 | 32/32 | 0.00 | 0.0000 |
| h_bar | h_bar_117 | 30/30 | 0.00 | 0.0000 |
| scatter | scatter_015 | 8/18 | 55.56 | 99.6923 |
| scatter | scatter_034 | 8/18 | 55.56 | 0.0400 |
| scatter | scatter_048 | 8/18 | 55.56 | 42.0000 |
| v_bar | v_bar_089 | 7/25 | 72.00 | 3.5000 |
| scatter | scatter_013 | 6/18 | 66.67 | 0.0200 |
| bubble | bubble_037 | 4/18 | 77.78 | 0.0000 |
| scatter | scatter_001 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_006 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_007 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_009 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_014 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_016 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_017 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_019 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_026 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_027 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_029 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_030 | 4/18 | 77.78 | 0.0000 |
| scatter | scatter_032 | 4/18 | 77.78 | 0.0000 |
| scatter | scatter_035 | 4/18 | 77.78 | 0.0000 |
| scatter | scatter_036 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_039 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_046 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_049 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_054 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_064 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_069 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_079 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_086 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_087 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_093 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_096 | 4/18 | 77.78 | 0.0200 |
| scatter | scatter_099 | 4/18 | 77.78 | 0.0200 |
| h_bar | h_bar_119 | 4/29 | 86.21 | 0.0000 |
| line | line_102 | 2/15 | 86.67 | 0.0000 |

## Legend Color Low

| type | chart | wrong/total | accuracy % |
| --- | --- | --- | --- |
| scatter | scatter_037 | 12/12 | 0.00 |
| scatter | scatter_051 | 9/9 | 0.00 |
| scatter | scatter_023 | 8/8 | 0.00 |
| scatter | scatter_006 | 6/12 | 50.00 |
| scatter | scatter_029 | 6/14 | 57.14 |
| scatter | scatter_081 | 5/10 | 50.00 |
| scatter | scatter_096 | 5/11 | 54.55 |
| scatter | scatter_065 | 5/14 | 64.29 |
| scatter | scatter_084 | 5/14 | 64.29 |
| scatter | scatter_079 | 4/8 | 50.00 |
| bubble | bubble_060 | 4/8 | 50.00 |
| scatter | scatter_043 | 4/9 | 55.56 |
| scatter | scatter_038 | 4/10 | 60.00 |
| scatter | scatter_014 | 4/11 | 63.64 |
| scatter | scatter_059 | 4/11 | 63.64 |
| scatter | scatter_026 | 4/13 | 69.23 |
| scatter | scatter_076 | 4/13 | 69.23 |
| scatter | scatter_028 | 4/14 | 71.43 |
| scatter | scatter_067 | 4/15 | 73.33 |
| scatter | scatter_068 | 4/15 | 73.33 |
| bubble | bubble_003 | 3/8 | 62.50 |
| bubble | bubble_005 | 3/8 | 62.50 |
| bubble | bubble_011 | 3/8 | 62.50 |
| bubble | bubble_018 | 3/8 | 62.50 |
| bubble | bubble_022 | 3/8 | 62.50 |
| bubble | bubble_024 | 3/8 | 62.50 |
| bubble | bubble_031 | 3/8 | 62.50 |
| bubble | bubble_032 | 3/8 | 62.50 |
| bubble | bubble_047 | 3/8 | 62.50 |
| bubble | bubble_048 | 3/8 | 62.50 |
| bubble | bubble_053 | 3/8 | 62.50 |
| bubble | bubble_055 | 3/8 | 62.50 |
| bubble | bubble_080 | 3/8 | 62.50 |
| bubble | bubble_082 | 3/8 | 62.50 |
| bubble | bubble_083 | 3/8 | 62.50 |
| bubble | bubble_090 | 3/8 | 62.50 |
| bubble | bubble_092 | 3/8 | 62.50 |
| bubble | bubble_096 | 3/8 | 62.50 |
| scatter | scatter_013 | 3/10 | 70.00 |
| scatter | scatter_053 | 3/10 | 70.00 |
| scatter | scatter_024 | 3/11 | 72.73 |
| scatter | scatter_077 | 3/11 | 72.73 |
| scatter | scatter_056 | 3/12 | 75.00 |
| scatter | scatter_032 | 3/13 | 76.92 |
| scatter | scatter_035 | 3/14 | 78.57 |
| scatter | scatter_073 | 3/14 | 78.57 |
| scatter | scatter_095 | 3/14 | 78.57 |
| scatter | scatter_098 | 3/15 | 80.00 |
| bubble | bubble_000 | 2/8 | 75.00 |
| bubble | bubble_007 | 2/8 | 75.00 |
| bubble | bubble_008 | 2/8 | 75.00 |
| bubble | bubble_015 | 2/8 | 75.00 |
| bubble | bubble_017 | 2/8 | 75.00 |
| bubble | bubble_021 | 2/8 | 75.00 |
| bubble | bubble_023 | 2/8 | 75.00 |
| bubble | bubble_026 | 2/8 | 75.00 |
| bubble | bubble_028 | 2/8 | 75.00 |
| bubble | bubble_030 | 2/8 | 75.00 |
| bubble | bubble_038 | 2/8 | 75.00 |
| bubble | bubble_040 | 2/8 | 75.00 |

## Point Label Name Low

| type | chart | wrong/total | accuracy % |
| --- | --- | --- | --- |
| scatter | scatter_037 | 12/12 | 0.00 |
| scatter | scatter_051 | 9/9 | 0.00 |
| scatter | scatter_023 | 8/8 | 0.00 |
| scatter | scatter_043 | 2/9 | 77.78 |
| scatter | scatter_052 | 1/9 | 88.89 |
| scatter | scatter_091 | 1/10 | 90.00 |
| scatter | scatter_059 | 1/11 | 90.91 |
| scatter | scatter_061 | 1/12 | 91.67 |
| scatter | scatter_068 | 1/15 | 93.33 |

## Text Axis Low

| type | chart | wrong/total | accuracy % |
| --- | --- | --- | --- |
| h_bar | h_bar_018 | 23/23 | 0.00 |
| h_bar | h_bar_031 | 19/19 | 0.00 |
| h_bar | h_bar_117 | 19/19 | 0.00 |
| h_bar | h_bar_119 | 4/12 | 66.67 |
| line | line_102 | 2/6 | 66.67 |
| line | line_014 | 1/1 | 0.00 |
| h_bar | h_bar_029 | 1/4 | 75.00 |
| h_bar | h_bar_137 | 1/14 | 92.86 |
| h_bar | h_bar_101 | 1/20 | 95.00 |
