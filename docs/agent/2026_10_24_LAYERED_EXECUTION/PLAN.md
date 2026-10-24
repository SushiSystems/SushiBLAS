# Layered execution in SushiBLAS: plan of the run

**Status:** Open — B1 to B12 landed
**Implements:** LAYERED_EXECUTION#B1, LAYERED_EXECUTION#B2, LAYERED_EXECUTION#B3, LAYERED_EXECUTION#B4, LAYERED_EXECUTION#B5, LAYERED_EXECUTION#B6, LAYERED_EXECUTION#B7, LAYERED_EXECUTION#B8, LAYERED_EXECUTION#B9, LAYERED_EXECUTION#B10, LAYERED_EXECUTION#B11, LAYERED_EXECUTION#B12, LAYERED_EXECUTION#B13

## How it runs

One orchestrator session runs the four repositories of the stack, from SushiRuntime's work folder
`docs/agent/2026_10_24_LAYERED_EXECUTION/` there. Up to four workers run at once, on tasks whose
files are disjoint. A worker checks its C++ with `clang -fsyntax-only` and builds nothing; the
orchestrator runs the checkers under `tools/` and commits each task. The owner ruled on
2026-10-24 that each repository is built and tested once, at the end of the programme, and that
AdaptiveCpp is checked then too. Work lands on `feature/layered_execution`.

## Tasks

| Id | Waits on | State |
| --- | --- | --- |
| B1 | nothing | built |
| B2 | B1 | built |
| B3 | nothing | built |
| B4 | B1 | built |
| B5 | B2, B3, B4, `sushiai` A0 | built |
| B6 | nothing | built |
| B7 | B6 | built |
| B8 | nothing | built |
| B9 | nothing | built |
| B10 | `sushiruntime` R2 | built |
| B11 | nothing | built |
| B12 | B11 | built |
| B13 | `sushiruntime` R10 | planned |
