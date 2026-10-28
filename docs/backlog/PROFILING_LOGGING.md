# Profiling logging

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-117 | P2 | unmeasured | Build and run the profiling and logging work | `sb build --benchmarks` configures and builds with the profiler on and off, and every suite of `profile`, `cost`, `profiling`, `launch`, `engine`… |  | `docs/design/PROFILING/README.md` | 2026-10-26 |
| RW-118 | P2 | unmeasured | Run `device_ceilings` on the RTX 3080 Ti and keep its file | `sb bench device_ceilings` has written the ceilings of the GPU, each span at least 10 ms and each spread within the limit, and a SushiAI record's `roofline`… |  | `docs/design/PROFILING/README.md` | 2026-10-26 |
| RW-119 | P2 | unmeasured | Measure what the profiling zones, counters and instants cost | the per-record and per-launch cost of `blas.record`, the lock wait, the scratch counters and the GEMM instants is recorded with the profiler off, on with no… |  | `docs/design/PROFILING/README.md` | 2026-10-26 |
| RW-121 | P2 | open | Write a cost rule for every `Unaudited` row of the cost table | no row of `cost_table.cpp` is `Unaudited`: each is costed from its body, or declared `DataDependent` with its reason in the cost README |  |  | 2026-10-26 |
| RW-122 | P2 | blocked | State the variant a kernel ran as, and its shape, on the span of its node | a GEMM span names its variant and a kernel span its launch shape as attributes, and the per-launch instants of `gemm.hpp` are gone | external: SushiProfiler has no way to write a fact from a task body and join it to its node | `docs/design/PROFILING/README.md` | 2026-10-26 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
