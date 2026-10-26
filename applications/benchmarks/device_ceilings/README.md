# device_ceilings

Measures what one device of the runtime can do, and writes it as `<device>.ceilings.json`. A SushiProfiler record then states the numbers as facts, and the `roofline` analysis judges each kernel's achieved rate against them. The file's format is [`docs/reference/CEILINGS_FORMAT.md`](../../../docs/reference/CEILINGS_FORMAT.md).

```
sb bench device_ceilings -- --device 0 --out build/ceilings
```

Source: not compiled

| Option | Default | Holds |
| --- | --- | --- |
| `--device` | 0 | Index of the runtime device |
| `--out` | `.` | Folder the file is written into |
| `--runs` | 5 | Timed runs per ceiling; the ceiling is their median |
| `--warmup` | 2 | Runs made and discarded before the timed ones |
| `--min-span-ms` | 10 | The shortest device span a timed run may have; the work is scaled until it lasts this long |
| `--spread-limit` | 0.05 | The spread above which a measurement is repeated, up to three times |
| `--stream-mib` | 256 | Size of each buffer of the bandwidth kernel |

## What it measures

| Ceiling | Kernel | Work counted |
| --- | --- | --- |
| Compute, per dtype the device supports (`f32`, `f64`, `f16`) | Eight independent multiply-add chains per work-item, `4096` work-items per compute unit | 16 operations per work-item per iteration |
| Bandwidth | A copy of one buffer to another, as many passes as the span needs | The bytes read plus the bytes written |
| Launch floor | An empty kernel | The device time of one |

Each kernel runs as one node of a runtime graph on the chosen device, and the span comes from a `RunObserver` on the device clock, the path SushiProfiler's spans come from. A bandwidth span covers several kernels, handed to the runtime with `SushiRuntime::note_dispatch_event`. The program prints the device, the driver, the build and the protocol, then one line `<name> <value> <unit>` per result.

A CPU device's file is marked `smoke_only`: its numbers show that the program runs and are not a ceiling. The numbers that count come from the GPU.

## Files

| File | Holds |
| --- | --- |
| `main.cpp` | The options, the orchestration and the file |
| `ceilings_options.hpp` | `Options` and `parse_options` |
| `ceilings_statistics.hpp` | `summarize`, the median and the spread |
| `span_observer.hpp` | `SpanObserver`, the span of the last run |
| `device_meter.hpp` | `DeviceMeter`, the kernels and the measurement protocol |
