# General

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-1 | P2 | decision | Decide the names of `spmv` and of "packed" | the owner has ruled whether the CSR `spmv` or the packed `spmv` is renamed, and what "packed" names besides BLAS packed storage |  |  | 2026-10-19 |
| RW-2 | P2 | decision | Delete `blas/sparse.hpp` or move it under `experimental/` | `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp` is gone or under `experimental/` |  |  | 2026-10-19 |
| RW-3 | P2 | decision | Remove `core/export.hpp` and the `<numbers>` include | neither is in the tree (C13) |  |  | 2026-10-19 |
| RW-4 | P2 | decision | Split `IO` into one reader and writer per format | the owner has approved the boundary, and each format has a reader and a writer over a shared host-copy helper, with a load that returns a status (C4) |  |  | 2026-10-19 |
| RW-5 | P2 | decision | Decide whether ELU accepts complex input | ELU takes complex input, or the owner has ruled it real-only |  |  | 2026-10-19 |
| RW-7 | P2 | decision | Decide whether a green CI on the tagged commit is a release condition | the owner has ruled, and the build jobs build against the tagged runtime |  |  | 2026-10-19 |
| RW-9 | P2 | decision | Rule how a function-local `constexpr` is spelt | the about 124 local values not spelt `K_UPPER_SNAKE` are renamed, or the owner has ruled that a local keeps `snake_case` |  |  | 2026-10-19 |
| RW-143 | P2 | open | Replace the test_runner LogCapture stand-in with SushiLog::LogCapture | test_runner/log_capture.hpp and test_log_capture.cpp are gone and every caller uses SushiLog::LogCapture |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |
| RW-144 | P2 | decision | Give the operations refusal checks host-callable seams so their tests run small | The overlap, dtype, count and shape refusals of blend, convolve, linalg, transforms and dense_operand run Small_ |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |
| RW-145 | P3 | open | Move the forced-mode and refusal test helpers into tests/support | in_each_forced_mode, trace_of, device_computes and expect_refusal live once in tests/support |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
