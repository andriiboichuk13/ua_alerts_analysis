# Raw data snapshot

Files from [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(MIT License, see `LICENSE` in this folder), downloaded on **2026-10-10 19:58 UTC**.

| file | coverage in this snapshot |
|---|---|
| `official_data_en.csv` | 2022-03-15 .. 2026-09-07 05:28 UTC (upstream stopped updating) |
| `volunteer_data_en.csv` | 2022-02-25 .. 2026-10-10 04:47 UTC |

The snapshot is committed so that results are reproducible: the volunteer file changes daily upstream.
`ua_alerts.data.download()` keeps existing files; call `download(force=True)` to refresh them.
