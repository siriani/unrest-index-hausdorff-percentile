# Spearman rho with the four optical-flow statistics (per recording)

|                                                                           |   mean |   var |   skew |   kurt |
|:--------------------------------------------------------------------------|-------:|------:|-------:|-------:|
| ('bbox', 'Travel distance', 'DVR-Ch1-CAM2-2026-06-18')                    |  0.629 | 0.641 |  0.346 |  0.331 |
| ('bbox', 'Travel distance', 'DVR-Ch2-CAM1-2026-06-18')                    |  0.645 | 0.656 |  0.376 |  0.355 |
| ('bbox', 'Travel distance', 'DVR-Ch3-CAM3-2026-06-18')                    |  0.712 | 0.698 |  0.27  |  0.236 |
| ('bbox', 'Travel distance', 'RecordPlan_20260224100150_1007_2')           |  0.323 | 0.257 |  0.064 |  0.09  |
| ('bbox', 'Travel distance', 'RecordPlan_20260224140202_1007_3')           |  0.574 | 0.505 |  0.133 |  0.133 |
| ('bbox', 'Travel distance', 'RecordPlan_20260409084432_1007_3')           |  0.429 | 0.531 |  0.494 |  0.497 |
| ('bbox', 'Travel distance', 'RecordPlan_20260423103037_1001_4')           |  0.667 | 0.655 |  0.251 |  0.245 |
| ('bbox', 'Travel distance', 'SimPlay-LCT2-CAM2-2026-03-16')               |  0.442 | 0.382 |  0.033 |  0.069 |
| ('bbox', 'Unrest Index (classic)', 'DVR-Ch1-CAM2-2026-06-18')             |  0.498 | 0.541 |  0.338 |  0.321 |
| ('bbox', 'Unrest Index (classic)', 'DVR-Ch2-CAM1-2026-06-18')             |  0.481 | 0.512 |  0.313 |  0.289 |
| ('bbox', 'Unrest Index (classic)', 'DVR-Ch3-CAM3-2026-06-18')             |  0.502 | 0.52  |  0.26  |  0.235 |
| ('bbox', 'Unrest Index (classic)', 'RecordPlan_20260224100150_1007_2')    |  0.253 | 0.212 |  0.071 |  0.094 |
| ('bbox', 'Unrest Index (classic)', 'RecordPlan_20260224140202_1007_3')    |  0.405 | 0.373 |  0.13  |  0.131 |
| ('bbox', 'Unrest Index (classic)', 'RecordPlan_20260409084432_1007_3')    |  0.327 | 0.406 |  0.343 |  0.343 |
| ('bbox', 'Unrest Index (classic)', 'RecordPlan_20260423103037_1001_4')    |  0.459 | 0.467 |  0.198 |  0.191 |
| ('bbox', 'Unrest Index (classic)', 'SimPlay-LCT2-CAM2-2026-03-16')        |  0.344 | 0.309 |  0.026 |  0.052 |
| ('mask_v3', 'Travel distance', 'DVR-Ch1-CAM2-2026-06-18')                 |  0.677 | 0.68  |  0.328 |  0.306 |
| ('mask_v3', 'Travel distance', 'DVR-Ch2-CAM1-2026-06-18')                 |  0.669 | 0.66  |  0.313 |  0.287 |
| ('mask_v3', 'Travel distance', 'DVR-Ch3-CAM3-2026-06-18')                 |  0.701 | 0.668 |  0.216 |  0.184 |
| ('mask_v3', 'Travel distance', 'RecordPlan_20260224100150_1007_2')        |  0.348 | 0.275 |  0.055 |  0.079 |
| ('mask_v3', 'Travel distance', 'RecordPlan_20260224140202_1007_3')        |  0.557 | 0.477 |  0.124 |  0.126 |
| ('mask_v3', 'Travel distance', 'RecordPlan_20260409084432_1007_3')        |  0.466 | 0.524 |  0.411 |  0.416 |
| ('mask_v3', 'Travel distance', 'RecordPlan_20260423103037_1001_4')        |  0.688 | 0.668 |  0.243 |  0.231 |
| ('mask_v3', 'Travel distance', 'SimPlay-LCT2-CAM2-2026-03-16')            |  0.538 | 0.476 |  0.042 |  0.072 |
| ('mask_v3', 'Unrest Index (classic)', 'DVR-Ch1-CAM2-2026-06-18')          |  0.576 | 0.626 |  0.374 |  0.349 |
| ('mask_v3', 'Unrest Index (classic)', 'DVR-Ch2-CAM1-2026-06-18')          |  0.563 | 0.595 |  0.349 |  0.32  |
| ('mask_v3', 'Unrest Index (classic)', 'DVR-Ch3-CAM3-2026-06-18')          |  0.587 | 0.598 |  0.255 |  0.216 |
| ('mask_v3', 'Unrest Index (classic)', 'RecordPlan_20260224100150_1007_2') |  0.284 | 0.24  |  0.071 |  0.091 |
| ('mask_v3', 'Unrest Index (classic)', 'RecordPlan_20260224140202_1007_3') |  0.434 | 0.397 |  0.138 |  0.139 |
| ('mask_v3', 'Unrest Index (classic)', 'RecordPlan_20260409084432_1007_3') |  0.364 | 0.45  |  0.368 |  0.365 |
| ('mask_v3', 'Unrest Index (classic)', 'RecordPlan_20260423103037_1001_4') |  0.563 | 0.586 |  0.269 |  0.248 |
| ('mask_v3', 'Unrest Index (classic)', 'SimPlay-LCT2-CAM2-2026-03-16')     |  0.469 | 0.436 |  0.055 |  0.08  |

## Across recordings (median, range)

|                                               |   median |   min |   max |
|:----------------------------------------------|---------:|------:|------:|
| ('bbox', 'Travel distance', 'kurt')           |    0.24  | 0.069 | 0.497 |
| ('bbox', 'Travel distance', 'mean')           |    0.601 | 0.323 | 0.712 |
| ('bbox', 'Travel distance', 'skew')           |    0.26  | 0.033 | 0.494 |
| ('bbox', 'Travel distance', 'var')            |    0.586 | 0.257 | 0.698 |
| ('bbox', 'Unrest Index (classic)', 'kurt')    |    0.213 | 0.052 | 0.343 |
| ('bbox', 'Unrest Index (classic)', 'mean')    |    0.432 | 0.253 | 0.502 |
| ('bbox', 'Unrest Index (classic)', 'skew')    |    0.229 | 0.026 | 0.343 |
| ('bbox', 'Unrest Index (classic)', 'var')     |    0.436 | 0.212 | 0.541 |
| ('mask_v3', 'Travel distance', 'kurt')        |    0.208 | 0.072 | 0.416 |
| ('mask_v3', 'Travel distance', 'mean')        |    0.613 | 0.348 | 0.701 |
| ('mask_v3', 'Travel distance', 'skew')        |    0.229 | 0.042 | 0.411 |
| ('mask_v3', 'Travel distance', 'var')         |    0.592 | 0.275 | 0.68  |
| ('mask_v3', 'Unrest Index (classic)', 'kurt') |    0.232 | 0.08  | 0.365 |
| ('mask_v3', 'Unrest Index (classic)', 'mean') |    0.516 | 0.284 | 0.587 |
| ('mask_v3', 'Unrest Index (classic)', 'skew') |    0.262 | 0.055 | 0.374 |
| ('mask_v3', 'Unrest Index (classic)', 'var')  |    0.518 | 0.24  | 0.626 |
