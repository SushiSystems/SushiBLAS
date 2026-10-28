# Continuous integration

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-85 | P2 | open | Fix the failing jobs and build the benchmarks | the Windows job finds a SYCL device past its filter, and the nightly Benchmarks and Quarantined jobs configure (`cmake/SushiBLASSushiLog.cmake:23`) and build… |  |  | 2026-10-19 |
| RW-86 | P2 | open | Bring the Linux job of the small and medium tests under 45 minutes | the job runs within the pre-merge budget and its timeout drops from 90 minutes |  |  | 2026-10-19 |
| RW-87 | P2 | open | Size the medium suites that reach nothing | each `Medium_` suite whose tests reach no runtime, queue, tensor or file is `Small_` |  |  | 2026-10-19 |
| RW-88 | P2 | unmeasured | Run the release workflow once | `.github/workflows/release.yml` has run before the tag depends on it |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-26 | own | The build pins no SushiRuntime version | check out a runtime commit that carries them | RW-89 |  | 2026-10-19 |
| KI-29 | own | The benchmark programs are not built on the module tree | build them locally with `sb build --benchmarks` | RW-85 |  | 2026-10-19 |
