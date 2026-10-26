# Ceilings file format

A ceilings file states what one device reached when it was measured: how fast it computes in each dtype it supports, how fast it moves memory, and how long an empty kernel takes. `device_ceilings` writes it and `read_ceilings` in the `profiling` module reads it. The `roofline` analysis of SushiProfiler reads the same numbers from a record's facts, so a record is judged against the ceilings it was made with, not against a file that may have moved since.

## File

`<device>.ceilings.json`, one file per device. `<device>` is the device's name in lower case, every run of characters other than letters and digits replaced by one underscore: `nvidia_geforce_rtx_3080_ti.ceilings.json`. The file is JSON, UTF-8, one object. The reader refuses an unknown member and a missing one.

| Member | Kind | Holds |
| --- | --- | --- |
| `format` | string | `sushiblas.ceilings` |
| `version` | whole | `1` |
| `device` | object | `name`, `backend` and `driver`, strings; `name` is not empty |
| `smoke_only` | bool | True when the numbers only show that the program runs, as on a CPU device |
| `method` | object | `warmup_runs` and `runs` (whole, `runs` at least 1), `min_span_ms` and `spread_limit` (real): how the numbers were taken |
| `compute` | array | One object per dtype the device supports: `dtype` (`f16`, `f32` or `f64`, each once), `flops_per_s` (real, above 0), `spread` (real) and `runs` (whole) |
| `bandwidth` | object | `bytes_per_s` (real, above 0), `spread` (real), `runs` (whole) |
| `launch_floor_ns` | real | The device time of an empty kernel, zero or more |

```json
{
  "format": "sushiblas.ceilings",
  "version": 1,
  "device": { "name": "Test Device", "backend": "Level-Zero", "driver": "1.2.3" },
  "smoke_only": false,
  "method": { "warmup_runs": 2, "runs": 5, "min_span_ms": 10.0, "spread_limit": 0.05 },
  "compute": [ { "dtype": "f32", "flops_per_s": 1.5e13, "spread": 0.01, "runs": 5 } ],
  "bandwidth": { "bytes_per_s": 7.5e11, "spread": 0.015, "runs": 5 },
  "launch_floor_ns": 5200.0
}
```

Source: not compiled

## Validation

`read_ceilings` returns a status and a message that names the path of the first refusal.

| Status | Meaning |
| --- | --- |
| `Ok` | A version 1 file; every value is valid |
| `NotJson` | The text is not JSON; the message gives the line and column |
| `WrongFormat` | The document has no `format` member equal to `sushiblas.ceilings` |
| `UnknownVersion` | A ceilings file of a version other than 1 |
| `Invalid` | A member is missing, unexpected, of the wrong kind or out of range, such as `compute[0].flops_per_s: not a positive number` |

## Method

Each number is the median of the timed runs, taken from the device clock through a graph run with a `RunObserver`, the path the profiler's spans come from. A run lasts at least `min_span_ms`: the program scales the work until it does, then discards `warmup_runs` runs and times `runs`. `spread` is the range of the timed runs over their median. A measurement whose spread is above `spread_limit` is repeated up to three times and the steadiest one is kept, with its real spread in the file. The streaming copy counts the bytes it reads plus the bytes it writes.

A CPU device's numbers are `smoke_only`: they show that the program runs and are not a ceiling to judge a kernel against. A ceiling is untested until the owner runs `device_ceilings` on the GPU.

## Version 1

The version is the number the writer writes and the only one the reader accepts. A change that adds, renames or removes a member is a new version; the reader of the old one refuses the new file by its `version` and not by a missing member.
