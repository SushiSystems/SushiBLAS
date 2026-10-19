# Plan

**Status:** Shipped — 2026-10-17

Closes the three items the logging standard work left open
(`docs/agent/2026_10_16_LOGGING_STANDARD`). Runs inline on `fix/logging_followups`; the owner
asked for it to land on main without waiting for main's CI.

| # | Task | Done when |
| --- | --- | --- |
| 1 | `SB_THROW_IF` throws and logs nothing; check every `catch` that could have relied on its record | `test_logger.cpp` shows no record; each `catch` in the library returns the failure or logs it |
| 2 | `SB_ASSERT(category, condition, msg)` logs in the caller's category; `K_CORE_LOG_CATEGORY` goes, `K_OPERANDS_LOG_CATEGORY` is added | Every call site names its module's category |
| 3 | Delete `core/format.hpp`; format with SushiRuntime's `sr_format` and link `SushiRuntime::core` | No source includes `core/format.hpp` |
| 4 | Build and run `sb test` | small and medium pass |

## The catches

| Catch | Why it needs no record of its own |
| --- | --- |
| `io.cpp` `status_of` | Catches `IOFailure`, which `fail_io` logs where it creates it, and returns the status |
| `io.cpp` `catch (...)` | Removes the partial file and rethrows |
| `benchmark_harness.hpp` `run_guarded` | A program's entry point; it prints the failure as its output and returns `EXIT_FAILURE` |
| `adapter/status_mapping.hpp`, `errors_demo.cpp` | Examples that return the failure as a status or print it |

No catch in the library relied on `SB_THROW_IF`'s record: its exceptions reach the caller.
