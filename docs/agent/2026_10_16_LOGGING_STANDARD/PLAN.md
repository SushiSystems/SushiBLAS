# Plan

Runs inline, on branch `feature/logging_standard`. Builds use at most 3 jobs
(`CMAKE_BUILD_PARALLEL_LEVEL=3`), and only after the owner allows them.

| # | Task | Done when |
| --- | --- | --- |
| 1 | Re-copy the shared tools from SushiSkills `16ff88b`; add `K_LOG_PREFIX = "SB"` and `K_CONSOLE_PATHS` to `tools/settings.py` | `check_all.py` runs the logging checker and lists the debt |
| 2 | Add `core/sushiblas_log_project.hpp`; rewrite `core/logger.hpp` as a front on `SL_LOG_FRONT_AT`; pin SushiLog and pass it from the CLI; remove `AsyncLogger` and `SyncLogger` | The core module compiles; `test_logger.cpp` tests the front |
| 3 | Add one `<module>_log_category.hpp` per logging module and move every call to its category, at the level of SPEC's table | The checker reports no uncategorized call and no module without a category |
| 4 | Replace the console write in `io.cpp` with the logger | The checker reports no console write |
| 5 | Update the core README, the manual pages that name the old logger, and the changelog | `check_docs_truth.py` is clean |
| 6 | Build and run `sb test --label small` and `--label medium` | Both pass |
