# SushiBLAS adopts SushiProfiler, and logs where it is silent: plan

**Status:** Open — written 2026-10-26; nothing is built, and no task starts before the owner has answered the decisions marked in SPEC.md, section 5.

**Spec:** `docs/agent/2026_10_26_PROFILER_AND_LOG_ADOPTION/SPEC.md`

## How to read it

- Ids: `PB-n` pass one (what SushiAI's inference profile needs), `PB-2n` pass two profiling, `LB-n` pass two logging. `BDn` is a decision of SPEC.md section 5; `D1` to `D24` and `PF-n` are SushiRuntime's; `SP-0` is the SushiProfiler v0.2.0 release; `SP-R1` is the generic `roofline` analysis in SushiProfiler with the options keys of G-B2 (BD3); `PF-A1` is the amendment BD1 asks of the runtime: `NodeCost`, carried by `NodeOptions`, `NodeTableRow` and `PlanNode`, written as the attributes `flops`, `bytes` and `work_class` by the device source (extends its PF-3, PF-4, PF-6).
- Order is hierarchy first: a brick comes before what uses it, and a module boundary with its build edges is committed before either side is written.
- A step marked **orchestrator** edits a build file. The implementer does not. Build edits wait until the SYCL compile-time work (`docs/agent/2026_10_26_SYCL_COMPILE_TIME`) has closed its build; it holds uncommitted edits to the same CMake files.
- One acceptance criterion and one build per task. The build is run by the orchestrator through `sb build`, then `sb run sushiblas_<module>_tests -- --gtest_filter='<pattern>'`. No implementer runs a build.
- Tests: a name that starts with the unit, a size prefix, a fixed seed, a regression test seen failing first (`testing`). `Medium_` uses the CPU SYCL device; `Large_` needs the GPU and is skipped, never passed, without it. A test that reads a log uses SushiLog's synchronous mode.
- Each task's module READMEs change in its commit. Manual pages, the design topic, the changelog and the backlog are PB-11.
- New test files go in their `tests/**/CMakeLists.txt` (orchestrator).

## First pass

| Id | Task | Waits on | Build files (orchestrator) |
| --- | --- | --- | --- |
| PB-0 | Pin SushiProfiler, the options, the package file | SP-0, PF-0, BD11 | yes |
| PB-1 | The `profile` front | PB-0 | yes |
| PB-2 | The `cost` module: types, the table, the GEMM rule | BD2, BD6 | yes |
| PB-3 | Stamp the cost in `TaskRecorder::insert`; the join test | PB-2, PF-A1 | tests list |
| PB-4 | Elementwise, nonlinear, logic and copy rules and their seams | PB-3 | tests list |
| PB-5 | Spatial rules and their seams | PB-3 | tests list |
| PB-6 | Every other operation: costed or declared uncosted; the checker | PB-4, PB-5 | tests list |
| PB-7 | A test observer for the four launch-width tests | PF-3, PF-5 (D24) | tests list |
| PB-8 | The `profiling` module: contributor and `register_profiler_parts` | PB-1, PF-7, SP-0, BD9, BD10 | yes |
| PB-9 | `device_ceilings` and the ceilings file | PB-8, PF-5, BD4, BD5, BD12 | yes |
| PB-10 | End to end: a session records a GEMM with cost and ceilings | PB-3, PB-9, PF-6, SP-R1 | tests list |
| PB-11 | Documentation | every PB task | no |

PB-7 can run as soon as the runtime's PF-5 lands and does not wait on PB-2 to PB-6.

### PB-0 Pin, options, package (orchestrator)

- **Files:** `cli/sushistack.deps.toml` (`depends_on` gains `sushiprofiler`; a `[sushisystems.sushiprofiler]` pin at v0.2.0, with the key prefix the stack settled, `[sushisystems.<name>]`); `cmake/` dependency block taken only when `SB_ENABLE_PROFILER` is on; the options `SB_ENABLE_PROFILER` (ON) and `SB_PROFILE_LEVEL` (2); the package config's `find_dependency`; the CI clone step if it does not read the toml generically.
- **Acceptance:** `sb setup` fetches the pinned SushiProfiler and `sb build` configures with the option on and finds `SushiProfiler::SushiProfiler`.
- **Build:** one `sb build`. **Tests:** none; the configure is the test. The off configuration is part of PB-10.

### PB-1 The `profile` front (foundation)

- **Files:** `modules/foundation/profile/include/SushiBLAS/profile/profile.hpp` (macros `SB_ZONE`, `SB_VALUE`, `SB_COUNTER`, each on SushiProfiler's `_AT` form with this library's level), `profile_gate.hpp` (the level of a translation unit from `SB_PROFILE_LEVEL`), `profile_keys.hpp` (attribute names `flops`, `bytes`, `work_class` and the site names, one place), `README.md`. **Orchestrator:** the module's `CMakeLists.txt`, the tier table `cmake/SushiBLASLayers.cmake` and `tools/settings.py`.
- **Acceptance:** a zone compiled at level 1 writes one event into an open collector; at level 0 it writes nothing and registers no site; with `SB_ENABLE_PROFILER` off the macros expand to nothing.
- **Build:** one `sb build`, then `sb run sushiblas_profile_tests`.
- **Tests:** `Small_ProfileGate.AZoneAtTheCompiledLevelWritesOneEvent`, `Small_ProfileGate.AZoneAboveTheCompiledLevelWritesNothing`, `Small_ProfileKeys.EveryKeyIsListedOnce`; two translation units of different levels.

### PB-2 The `cost` module

- **Files:** `modules/foundation/cost/include/SushiBLAS/cost/`: `cost.hpp` (`struct Cost { std::uint64_t flops; std::uint64_t bytes; std::uint8_t work_class; }`, `Cost::uncosted(UncostedReason)`), `cost_shape.hpp` (`struct CostShape { std::int64_t m, n, k, batch, count; Core::DataType dtype; std::uint32_t extra_flops_per_element; }`), `cost_table.hpp` (`[[nodiscard]] Cost cost_of(SushiRuntime::OperationID, const CostShape&) noexcept` and a row type `CostRow{OperationID, RuleKind, weight}`), `source/cost_table.cpp` (the sorted rows and the rules), `README.md` (the conventions of BD6). The module includes no SYCL, no engine and no SushiProfiler. **Orchestrator:** `CMakeLists.txt` (tier `foundation`, depends on `core`), tier table, tests list.
- **Acceptance:** the GEMM row gives `2*m*n*k*batch` FLOPs for a real dtype and `8*m*n*k*batch` for a complex one, the class of the dtype, and a lookup of an id with no row returns `Cost::uncosted(NoRule)`.
- **Build:** one `sb build`, then `sb run sushiblas_cost_tests`.
- **Tests:** `Small_CostTable.GemmCostsTwoMNKPerBatch`, `Small_CostTable.AComplexGemmCostsEightMNK`, `Small_CostTable.AnUnknownIdIsUncostedNoRule`, `Small_CostTable.RowsAreSortedAndUnique`, `Small_CostShape.TheClassFollowsTheDtype`.

### PB-3 Stamp the cost; the join

- **Files:** `modules/engine/engine/include/SushiBLAS/engine/task_description.hpp` (member `CostShape shape`, default `count = 0`), `task_recorder.hpp` (`insert` computes `cost_of(description.operation, description.shape)`, adds the footprint of `OperandSet::read_regions()` and `write_regions()` to `bytes`, and sets `SushiRuntime::NodeOptions::cost`), `modules/operations/blas/include/SushiBLAS/blas/dispatch.hpp` (`execute_gemm` fills `shape` from `m`, `n`, `k`, `batch_size`, the dtype and the epilogue's per-element count), `modules/engine/graph_probe/include/SushiBLAS/graph_probe/graph_probe.hpp` (`RecordedTask::cost`, read from `PlanNode`), the module READMEs. **Orchestrator:** `engine` gains `DEPENDS cost`.
- **Acceptance:** after recording one GEMM through `BLASOps`, the plan's node holds `flops = 2*m*n*k`, a `bytes` equal to the footprint of A, B and C, and the `work_class` of the dtype; and every record call adds one rule at the position `node_count()` had before it.
- **Build:** one `sb build`, then `sb run sushiblas_engine_tests` and `sushiblas_blas_tests`.
- **Tests:** `Medium_TaskRecorderCost.GemmNodeStatesFlopsBytesAndClass`, `Medium_TaskRecorderCost.AnUpdatedOperandCountsTwiceInTheBytes`, `Medium_TaskRecorderCost.ABetaZeroGemmReadsNoC`, `Medium_TaskRecorderJoin.EachRecordCallAddsOneRuleAtTheNodeCountItSaw`, `Medium_TaskRecorderJoin.ALateCountPairIsOneRule`, `Medium_TaskRecorderJoin.RuleRegistrationMapsEveryRuleOfAFusedGemm`.

### PB-4 Elementwise, nonlinear, logic and copy

- **Files:** rows and rules in `cost_table.cpp`; the shape facts in `elementwise/dispatch.hpp`, `nonlinear/dispatch.hpp`, `logic/dispatch.hpp`, `elementwise/source/convert.cpp`, `blas` level-1 `copy`, `scal`, `axpy`; READMEs.
- **Acceptance:** each costed operation of these families records the FLOPs its row states for a given count, and the epilogue of a fused GEMM adds its per-element count.
- **Build:** one `sb build`, then the `cost`, `elementwise`, `nonlinear`, `logic` and `blas` suites.
- **Tests:** `Small_CostTable.AnElementwiseRowScalesWithTheCount`, `Small_CostTable.ACopyCostsNoFlops`, `Medium_ElementwiseCost.AddRecordsOneFlopPerElement`, `Medium_NonlinearCost.EveryForwardOperationHasARow`, `Medium_FusedGemmCost.TheEpilogueAddsItsPerElementCount`.

### PB-5 Spatial

- **Files:** rows and rules for `im2col`, `col2im`, `window_dot` (and its two backward forms), `pool_max`, `pool_avg`, `upsample_nearest`, `channel_affine`, `broadcast_spatial`, concat, slice and layout conversion; shape facts in `modules/operations/spatial/source/*.cpp`; READMEs.
- **Acceptance:** `window_dot` over a known window geometry records `2 * outputs * window * channels` FLOPs; operations that only move data record 0 FLOPs and a footprint.
- **Build:** one `sb build`, then `sushiblas_cost_tests` and `sushiblas_spatial_tests`.
- **Tests:** `Medium_SpatialCost.WindowDotCostsTwoPerWindowElement`, `Medium_SpatialCost.Im2colMovesDataAndCostsNoFlops`, `Medium_SpatialCost.PoolMaxCountsOneComparePerWindowElement`.

### PB-6 Every other operation; the checker

- **Files:** a row for each remaining `_op` literal, costed or `uncosted(reason)` with the reasons `SeveralSubmits` (BD8), `DataDependent`, `NoWork`; `tools/profiling/check_operation_costs.py` (the repository's own checker: every `"..."_op` literal under `modules/` has a row; run by `check_all.py`) and its test in `tools/tests/`; READMEs.
- **Acceptance:** the checker reports no `_op` literal without a row; each operation marked costed has had its body read and submits one kernel, which the task records in the module README as a table of operation, rule and submit count.
- **Build:** one `sb build`, then `sushiblas_cost_tests`; the checker runs without a build.
- **Tests:** `Small_CostTable.EveryRowNamesAKnownRule`, `Small_CostTable.AMultiStageOperationIsUncostedSeveralSubmits`, `tools/tests/test_check_operation_costs.py`.

### PB-7 A test observer for the four launch-width tests

- **Files:** `modules/foundation/test_runner/include/SushiBLAS/test_runner/run_observer_capture.hpp` (`class RunObserverCapture`, a `SushiRuntime::RunObserver` that keeps the dispatch records of a run: `node_index`, `launched_width`, `live_width`; `attach(graph)`, `records()`); `tests/foundation/graph_primitives/test_counting_sort.cpp` (`:370`, `:623`, `:666`), `test_prefix_scan.cpp` (`:246`, `:283`), `test_reduce/test_reduce.cpp` (`:465`, `:490`), `test_segmented_reduce.cpp` (`:306`, `:338`): each `graph.set_profiling(true)` becomes `capture.attach(graph)` and each `report.node_events` becomes `capture.records()`; `modules/foundation/test_runner/README.md`.
- **Acceptance:** the four files assert the same widths and counts as now and no longer name a `RunReport` timing field or `set_profiling`.
- **Build:** one `sb build`, then `sb run sushiblas_graph_primitives_tests`.
- **Tests:** the existing tests, unchanged in meaning; plus `Small_RunObserverCapture.KeepsOneRecordPerDispatch`.
- **Note:** `git grep -n "node_events\|set_profiling" tests modules applications docs` must find nothing outside `docs/agent` and `docs/archive` afterwards.

### PB-8 The `profiling` module

- **Files:** `modules/engine/profiling/include/SushiBLAS/profiling/`: `profiler_parts.hpp` (`class SushiBLASProfilerParts` owning the contributor; `[[nodiscard]] SushiBLASProfilerParts register_profiler_parts(SushiProfiler::Registry&, SushiRuntime::Runtime&, std::string_view ceilings_folder)`), `blas_environment.hpp` (`EnvironmentContributor`: `sushiblas.version`, build facts compiled inside the archive, default execution mode, `SB_GEMM_TILE` as read, layout, per-device `Device::Profile` facts, and the ceilings facts `ceiling.<track>.<class>`, `ceiling.<track>.bandwidth`, `ceiling.<track>.launch_floor_ns`; `unmeasured` when no file), `ceilings_file.hpp` (read and validate), `README.md`. It includes SushiProfiler only through `profile`. **Orchestrator:** `CMakeLists.txt` (tier `engine`, `DEPENDS core launch profile`), tier table, tests list.
- **Acceptance:** a session started with the registry the function filled records the sushiblas facts, the build facts and the ceilings of every device that has a file, and refuses a second registration.
- **Build:** one `sb build`, then `sb run sushiblas_profiling_tests`.
- **Tests:** `Medium_BlasEnvironment.StatesTheTilePinAndTheDefaultMode`, `Medium_BlasEnvironment.StatesTheCeilingsOfADeviceWithAFile`, `Medium_BlasEnvironment.StatesUnmeasuredWithoutAFile`, `Small_CeilingsFile.RefusesAnUnknownVersion`, `Small_CeilingsFile.RefusesAMissingBandwidth`, `Small_RegisterProfilerParts.RefusesASecondRegistration`, `Small_RegisterProfilerParts.TheHandleOutlivesTheSession`.

### PB-9 `device_ceilings` and the ceilings file

- **Files:** `applications/benchmarks/device_ceilings/` (`main.cpp`, `README.md`; options `--device`, `--out`, `--runs`; one graph run per kernel with a `RunObserver` attached, median of at least three runs, the machine and build printed, the file written as `<device>.ceilings.json`), `applications/benchmarks/README.md` (new, lists it), `docs/reference/CEILINGS_FORMAT.md` (the format, version 1, with the validator the reader of PB-8 implements). **Orchestrator:** `applications/benchmarks/CMakeLists.txt`, the option wiring under `SB_BUILD_BENCHMARKS`.
- **Acceptance:** `sb run device_ceilings -- --out <folder>` on the CPU SYCL device writes a file the PB-8 reader accepts, with a finite compute ceiling for every precision the device supports, a bandwidth and a launch floor.
- **Build:** one `sb build --benchmarks`, then the run above (on the owner's GPU for the numbers that count).
- **Tests:** `Small_CeilingsFormat.TheWriterOutputPassesTheReader` in `sushiblas_profiling_tests`; the program itself never passes or fails (`performance`).

### PB-10 End to end

- **Files:** `tests/engine/profiling/test_profiling_end_to_end.cpp`.
- **Acceptance:** a session over a graph that runs two GEMMs and an elementwise task produces a record whose device spans carry `flops`, `bytes` and `work_class`, whose environment holds the ceilings, and whose `roofline` table has a row per operation with its achieved rate and the fraction of the ceiling; the same session built with `SB_ENABLE_PROFILER` off compiles and runs the graph.
- **Build:** one `sb build`, then `sb run sushiblas_profiling_tests`.
- **Tests:** `Medium_ProfilingEndToEnd.AGemmSpanCarriesFlopsBytesAndAClass`, `Medium_ProfilingEndToEnd.TheRooflineRowNamesTheCeiling`, `Medium_ProfilingEndToEnd.AnUncostedOperationHasNoFraction`, `Medium_ProfilingEndToEnd.TheProfilerOffBuildStillRuns`.

### PB-11 Documentation (orchestrator)

- **Files:** `docs/design/PROFILING/README.md` (BD13) and its row in `docs/design/README.md`; `docs/architecture/` page for cost and the profiler; the guide of registering the parts; the module READMEs index in `modules/README.md`; `docs/reference/CHANGELOG.md` lines; backlog entries for every pass-two task in `docs/design/REMAINING_WORK.md` (after the SYCL compile-time work commits its own edit to that file); the glossary entries; the `Implements` line of this work.
- **Acceptance:** `python tools/check_all.py .` is clean.

## Second pass (listed, not first pass)

Profiling:

| Id | Task | Waits on |
| --- | --- | --- |
| PB-20 | Zones: `TaskRecorder::insert`, the recording lock, scratch lease | PB-1, runtime PF-8 |
| PB-21 | The benchmark harness marks `bench.iteration`; every micro benchmark writes a record | PB-8, SP-0 |
| PB-22 | `sb bench <name> -- <options>` (CLI; owner decides, BD12) | PB-9 |
| PB-23 | The kernel variant as a span attribute or instant, then delete rows K2 and K3 | a SushiProfiler design for a fact from a task body |
| PB-24 | Name and identify the nodes of `graph_primitives` (row P1) | api-stability line |
| PB-25 | Cost the multi-stage operations once the runtime times a whole body | runtime design, BD8 |
| PB-26 | Pool hit and miss counters for scratch | SushiLog RW-4 |

Logging (all wait on runtime D18 and on a first-pass build closing):

| Id | Task | Rows | Waits on |
| --- | --- | --- | --- |
| LB-1 | Category `launch`, with its build edge and header | D1 to D3 | none |
| LB-2 | The device probe states its fallbacks and the profile of each device | D1, D2, D3 | LB-1, SushiLog RW-3 |
| LB-3 | The tile pin accepted is stated | D4 | none |
| LB-4 | The no-local fallback warns once per device and tile | K1 | SushiLog RW-3 |
| LB-5 | A failed partial-file cleanup warns | I1 | none |
| LB-6 | Remove the per-launch `DEBUG` and the task-body `TRACE` records | K3, K4 | PB-23 |
| LB-7 | Categories for `blas`, `spatial`, `nonlinear`, `sparse`, `graph_primitives` where a task above adds a call | | LB-1 |

Each logging task's test captures records through SushiLog's synchronous mode (RW-8 when shipped): `Medium_DeviceProbeLog.AFallbackWarnsOncePerDevice`, `Medium_GemmFallbackLog.WarnsOncePerDeviceAndTile`, `Medium_IoLog.AFailedCleanupWarns`.

## Proposed order

Three waves, a proposal. Wave 1 (no runtime dependency): PB-2, then PB-3 once PF-A1 is in the runtime's plan, PB-1 once PB-0 is built. Wave 2: PB-4, PB-5, PB-6 in one repository sequence (they edit the same table file, so they run one after another, not in parallel). Wave 3: PB-7, PB-8, PB-9, PB-10 as the runtime's PF-3 to PF-7 and SushiProfiler's SP-0 and SP-R1 release. The owner chooses inline or waves.
