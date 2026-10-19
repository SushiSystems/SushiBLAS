# Logging standard

**Status:** Shipped — 2026-10-17

SushiBLAS moves its logging to the SushiSkills `logging` standard and to SushiLog, the one logger
every Sushi library links (task 9 of SushiSkills `docs/agent/2026_10_16_SUSHILOG_REPOSITORY`).
SushiBLAS becomes a front on SushiLog's `SL_LOG_FRONT_AT`, links SushiLog directly at the pinned
v0.1.2, and keeps no logger of its own.

## Today

- `modules/foundation/core/include/SushiBLAS/core/logger.hpp` holds a logger of its own: an
  `AsyncLogger` with a queue and a thread, a `SyncLogger`, four levels and `SB_LOG_*` macros
  that take no category.
- 119 calls in 19 folders. Most are `SB_LOG_INFO` in a kernel's launch path, once per call.
- `check_logging.py` from SushiSkills reports 64 uncategorized calls, 13 modules without a
  category header and console writes in `benchmark_harness.hpp` and `io.cpp`.
- `tools/` has no `logs/` checker; `tools/settings.py` has no `K_LOG_PREFIX` or
  `K_CONSOLE_PATHS`.

## Target

| Part | SushiBLAS |
| --- | --- |
| Project | `K_SUSHIBLAS_LOG_PROJECT{"sushiblas", "SB_LOG_LEVEL"}` in `core/sushiblas_log_project.hpp` |
| Front | `core/logger.hpp` defines `SB_LOG_TRACE` through `SB_LOG_FATAL` on `SL_LOG_FRONT_AT`, with a per-translation-unit ceiling from `NDEBUG` and `SB_ENABLE_LOGGING` |
| Dependency | `[sushi.sushilog]` in `cli/sushistack.deps.toml`; the CLI passes `SUSHILOG_DIR`; `cmake/SushiBLASSushiLog.cmake` adds it before SushiRuntime; `sushilog.dll` beside every program |
| Categories | One `K_<MODULE>_LOG_CATEGORY` per module that logs, in `<module>/<module>_log_category.hpp` |
| Queue, thread, sink | None; SushiLog's |
| Synchronous mode | Callers write `SushiLog::set_log_synchronous`; the front re-exports no SushiLog name |
| `SB_ASSERT`, `SB_THROW_IF` | Stay in `logger.hpp`; `SB_ASSERT` writes a `FATAL` record before it aborts |

## Levels of the existing calls

| Call | Level |
| --- | --- |
| A kernel's launch shape and chosen variant, once per call | `DEBUG` |
| An element-wise, reduction, random or logic op's size, once per call | `TRACE` |
| Engine created, recorded tasks discarded, file saved or loaded | `INFO` |
| A fallback, such as the GEMM local tile the device cannot hold | `WARN` |
| A failure status created at a boundary | `ERROR` |

A format spec such as `{:.4f}` is formatted into a string before the call.

## Out of scope

- `core/format.hpp` stays; `io` and `operand_check` format error text with it.
- The benchmark harness's table on stdout is output, not a log. Its folder goes into
  `K_CONSOLE_PATHS`.

## Acceptance

`python tools/check_all.py .` is clean with the re-copied `check_logging.py`, and
`sb test --label small` and `--label medium` pass with 3 build jobs.
