# Continuous integration

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-85 | P2 | open | Fix the failing jobs and build the benchmarks | the Windows job finds a SYCL device past its filter, and the nightly Benchmarks and Quarantined jobs configure (`cmake/SushiBLASSushiLog.cmake:23`) and build… |  |  | 2026-10-19 |
| RW-86 | P2 | open | Bring the Linux job of the small and medium tests under 45 minutes | the job runs within the pre-merge budget and its timeout drops from 90 minutes |  |  | 2026-10-19 |
| RW-87 | P2 | open | Size the medium suites that reach nothing | each `Medium_` suite whose tests reach no runtime, queue, tensor or file is `Small_` |  |  | 2026-10-19 |
| RW-88 | P2 | unmeasured | Run the release workflow once | `.github/workflows/release.yml` has run before the tag depends on it |  |  | 2026-10-19 |
| RW-123 | P1 | open | Link TSan-instrumented library packages into a ThreadSanitizer tree | a --sanitize thread tree links SushiRuntime, SushiLog and SushiProfiler packages built with TSan, and the ThreadSanitizer job passes |  |  | 2026-10-29 |
| RW-124 | P1 | open | Check out SushiLog and SushiProfiler in the nightly benchmark and quarantined jobs | Both nightly jobs check out the pinned SushiLog and SushiProfiler and sb finds them; the nightly run is green |  | `.github/workflows/nightly.yml` | 2026-10-30 |
| RW-125 | P0 | open | Make the Windows pre-merge job finish inside its 150-minute timeout | The Windows (small, medium) job of ci.yml completes sb build and both test labels on main, with the gate green |  | `.github/workflows/ci.yml` | 2026-10-30 |
| RW-126 | P0 | decision | Choose how the pre-merge builds fit their time budget: raise the timeout or cut the build | The owner rules whether timeouts rise above the 20/45-minute budget or the build cost falls |  | `.github/workflows/ci.yml` | 2026-10-30 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-26 | own | The build pins no SushiRuntime version | check out a runtime commit that carries them | RW-89 |  | 2026-10-19 |
| KI-29 | own | The benchmark programs are not built on the module tree | build them locally with `sb build --benchmarks` | RW-85 |  | 2026-10-19 |
