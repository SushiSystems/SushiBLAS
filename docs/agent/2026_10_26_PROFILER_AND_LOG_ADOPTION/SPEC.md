# SushiBLAS adopts SushiProfiler, and logs where it is silent: analysis

**Status:** Open — written 2026-10-26; analysis done, section 8 holds the rulings; implementation runs with every build at the end.

No `**Implements:**` line yet: the repository has no profiling design topic. Decision BD13 opens
`docs/design/PROFILING/README.md`, as SushiRuntime's D22 does; the line follows it.

Read on 2026-10-26: this repository on `feature/layered_execution` (the uncommitted files of
`docs/agent/2026_10_26_SYCL_COMPILE_TIME` were not read and are not touched), SushiProfiler at
`8caa01b` (v0.1.0 plus the shared-library commit; `docs/agent/2026_10_26_CLIENT_CONTRACT/SPEC.md`
describes v0.2.0), SushiLog's `docs/design/REMAINING_WORK.md` and SushiRuntime's
`docs/agent/2026_10_26_PROFILER_AND_LOG_ADOPTION/SPEC.md` with its decisions D1 to D24. A claim that
was not verified says so. Nothing was built or run.

Priority: SushiAI's inference profile. Pass one is only what that profile needs from SushiBLAS: what
each kernel did, how close it ran to the device's ceilings, and whether the join from a span to an IR
node works. The rest is listed in section 6 and in `PLAN.md` as pass two.

## 1. Profiling

### 1.1 What SushiBLAS measures today

Nothing. No file under `modules/` or `applications/` reads a clock, an event or a `RunReport` timing
field (grep for `profiling`, `node_events`, `node_timings`, `steady_clock`, 2026-10-26: the only
hits are `Device::Profile`, the test files of 1.7 and the benchmark harness). The harness times with
`std::chrono::steady_clock` around record, plan compile and run together
(`modules/operations/benchmark_harness/README.md`, paragraph on `time_iterations`), so it measures no
device time. No operation knows its FLOP count; `Bench::K_FLOPS_PER_GIGAFLOP`
(`benchmark_harness.hpp`) is a unit, not a rule.

### 1.2 Where a kernel is submitted and how a node is named

| Step | Where |
| --- | --- |
| Every tensor operation ends in one `TaskRecorder::record_pair` or `record_late_pair` | `task_recorder.hpp:201-249` |
| Both call `insert`, the one caller of `Recorder::task` (and `dynamic_kernel`) in the library | `:284-330`, `:237-246` |
| `insert` copies name, operation, determinism and the 8 parameter slots into `NodeOptions` | `:310-315` |
| The runtime later calls the stored body, which calls the reference or the SYCL kernel | `RecordedPairTask::operator()`, `:84-87`; `KernelPairDispatcher::run`, `kernel_pair_dispatcher.hpp:45-57` |
| The SYCL kernel returns the event of its last `queue.submit` | e.g. `gemm.hpp:448`, `:590`, `:761` (one submit per variant) |

Names and operation ids come from the call site: `Level3::gemm` passes `"blas.lvl3.gemm"` and
`"blas.lvl3.gemm"_op` (`modules/operations/blas/source/level3/gemm.cpp:45`); `BLASMetadata` stores
them (`blas_metadata.hpp:52-56`). There are 178 distinct `_op` literals (grep over `modules/`),
spread over about 25 construction sites of `TaskDescription` (`task_description.hpp:45`). The name is
the operation, not the IR node: every GEMM of a network is named `blas.lvl3.gemm`.

Nodes recorded outside `TaskRecorder` carry no operation id. `GraphPrimitives::Detail::NodeStamper`
sets width, and name and determinism only when the call stated options (`node_stamper.hpp:89-106`);
an unnamed call yields a node with no name and a default `OperationID`.

Findings:

- **One recording point.** Every operation reaches the runtime through `insert` (`task_recorder.hpp:304`),
  so anything a node should carry is stamped in one place.
- **A body may submit several kernels and return one event.** `reduce_indexed` submits a staged kernel
  and a final one and returns the final event (`reduce_1d.hpp:779-817`). The runtime times the returned
  event only, so a span covers the last stage, and FLOPs divided by that span overstate the rate.
  Other files with several `.submit(` calls (`fft.hpp`, `lu.hpp`, `level1.hpp`, `triangular_solve.hpp`)
  are unverified: a branch per variant or a chain, not read.
- **An empty task returns `sycl::event()`** (`window_dot.hpp:58`, `:145`), which has no profiling info;
  the runtime swallows the failed read (its SPEC 1.2, item 4). Such a task states cost 0 and may have no span.
- **`TaskHandle` is not a join key.** Its `sequence_` is "how many tasks the recorder had recorded before
  this one" (`task_handle.hpp:77`); a recorder shares its graph with other writers, so it differs from the rule
  position the runtime indexes by.

### 1.3 The join from a span to an IR node

CLIENTS.md has SushiAI record, per IR node, the range of runtime registration indices it emitted, and
join on `NodeEventRecord::node_index`. SushiProfiler's open question is whether that index counts every
task, including scratch and late pairs. From SushiBLAS's side:

- A record call adds exactly one runtime rule: `insert` checks `node_count()` before and after and throws
  if it did not grow (`task_recorder.hpp:319-327`); the comment on the callable says "adds exactly one node"
  (`:299`). A late pair is one `dynamic_kernel` (`:241`). Scratch is a leased operand, not a node
  (`dispatch.hpp:252`, `OperandSet::scratch`).
- An operation that records several tasks (fused GEMM with spill, composite operations) records several
  rules, each at its own `node_count()`. SushiAI reads `graph.node_count()` around a lowering
  (`sushiai/.../compiled_step.cpp:122`, `differentiator.cpp:126-147` use the same call) and maps the range through
  `Graph::rule_registration` (`graph.hpp:148`). The range then holds every task of the IR node.
- So SushiBLAS needs no change for the join, provided the range is taken from the graph and not from `TaskHandle`
  (BD7). A SushiBLAS test pins the property (PB-3, `Medium_TaskRecorderJoin.*`). Whether the runtime's registration
  index follows rule order one to one is the runtime's to confirm (its PF-4 test `TheTableAndTheRecordsJoinByNodeIndex`).

### 1.4 The cost data each operation knows

At record time each seam holds the extents, strides, dtype, flags and operand tensors. `execute_gemm` has
`m`, `n`, `k`, `batch_size`, the dtype and `beta` in hand (`blas/dispatch.hpp:197-250`). Elementwise and
logic seams hold the element count (`elementwise/dispatch.hpp:85-86`). The data needed for FLOPs is therefore known at
record time and static for a recorded task, except for a late-width task, whose count is read from the device at
run time (`task_recorder.hpp:221-249`; its record-time capacity is an upper bound).

Bytes need no per-operation rule. `OperandSet` already lists the byte intervals a task reads and writes
(`operand_set.hpp:174-181`), and `Tensor::region()` is the element span of the view in bytes
(`tensor.hpp:313-322`). The sum of read and write lengths, an updated operand counted twice, is the footprint the task
declared. It is exact for a dense operand and an overestimate for a strided or sliced view; a `K_WHOLE`
length has no size and is skipped.

| Cost part | Source | Rule |
| --- | --- | --- |
| `flops` | A per-operation rule over the extents and the dtype | BD2, BD6 |
| `bytes` | Declared footprint of the operand set | automatic in `insert` |
| `work_class` | The dtype's ceiling class: half, float, double, complex counted in its real class, integer | one per dtype |

### 1.5 The device facts for ceilings

`Device::Profile` holds `compute_units`, `sub_group_size`, `max_work_group_size`, `local_mem_bytes`,
`cache_bytes`, `supports_fp64`, `supports_fp16`, vendor and architecture (`device_policy.hpp:211-262`,
probed at `:367-413`). It holds no clock, no peak rate and no memory bandwidth, and SYCL gives no bandwidth.
A theoretical peak (units times lanes times clock) needs the lane count of an architecture the code does not
list for every part (`Architecture` has nine values). The ceilings are therefore measured, as CLIENTS.md says.
`cache_bytes` falls back to an assumed 256 KiB when the device reports none (`:72`, `:408-409`), and that
fallback is silent (section 2, row D1).

### 1.6 The ceilings and the roofline

- `device_ceilings` measures, per device, an FMA loop per precision the device supports, a streaming copy and an
  empty kernel (CLIENTS.md). It must time on the device clock. With D24 (timing leaves `RunReport`) the way to do
  that is a graph run with a `RunObserver` attached, which is the same path the spans come from (BD4).
- The result is a file, `<device>.ceilings.json`: a persistent format, so it carries a version key, the device name,
  backend and driver, the compute ceilings per class in FLOP/s, the bandwidth in bytes/s, the launch floor in ns,
  the median and spread of the runs, and a validator (`file-formats`; the format page is part of PB-9).
- The analysis `roofline` reads a record, never the live program (RECORD_AND_ANALYSES.md, "Analyses"). So the ceilings
  must be in the record, and the analysis must be one `sp record show` can run (section 4, G-B1 and G-B3; BD3, BD5).
- A fraction of the ceiling is `min(compute ceiling, bandwidth times arithmetic intensity)` against the achieved
  rate. With footprint bytes the intensity is an upper bound, so the fraction is a lower bound of the true cache-aware
  fraction. The table's header says so.

### 1.7 Where the runtime's change reaches SushiBLAS

SushiRuntime D24 removes every timing field from `RunReport`, `Graph::set_profiling` and `RuntimeConfig::profiling`.
Four test files read them:

| File | Uses | Count |
| --- | --- | --- |
| `tests/foundation/graph_primitives/test_counting_sort.cpp` | `set_profiling` (`:370`, `:623`, `:666`), `node_events` sizes, `node_index`, `launched_width`, `live_width` | 8 reads |
| `tests/foundation/graph_primitives/test_prefix_scan.cpp` | `:246`, `:283`; `node_index`, widths | 6 reads |
| `tests/foundation/graph_primitives/test_reduce/test_reduce.cpp` | `:465`, `:490`; widths | 6 reads |
| `tests/foundation/graph_primitives/test_segmented_reduce.cpp` | `:306`, `:338`; widths | 6 reads |

The other `RunReport` mentions (`tests/engine/engine/test_task_recorder.cpp:184-220`,
`tests/operations/index/test_index_ops.cpp:722`, `recording_graph.hpp:130`, three examples) pass a report to
`run` and read no timing field (grep, 2026-10-26). The manual pages that name `RunReport`
(`FIRST_PROGRAM.md:61`, `INTEGRATION.md:313`, `EXCEPTIONS.md`, `OVERVIEW.md`, `RUNTIME_DEPENDENCY.md`) must be
rechecked by `git grep` when the runtime releases. All three timing-field readers are the four test files, which
install a test observer instead (PB-7). The launch widths they assert are a property of the primitives, so the
observer must deliver them; the runtime's `RunObserver` record keeps `launched_width` and `live_width` (its D5).

### 1.8 What pass one gives SushiAI

A SushiAI inference run with a session attached produces, for every dispatch of a BLAS node: the span from the runtime
(time, device track, node, invocation), plus `flops`, `bytes` and `work_class` attributes, plus environment facts with
the device's ceilings, plus the `roofline` table. The IR node comes from SushiAI's own range. Not in pass one: which
kernel variant ran (section 2, row K1), record-time cost of recording, the benchmark harness on a timeline.

## 2. Logging

Existing log calls: 61 outside the macro definitions, of which 28 are per-launch `DEBUG` records in `modules/kernels`
and 15 are `TRACE` records in task bodies; the rest are `ERROR` before a throw, `INFO` at start and `WARN` twice. Modules
that own a log category: tensor, engine, functor_map, operands, blas_kernels, blas_seam_kernels, sparse_kernels, elementwise,
io, logic, random, reductions, test_runner. Modules that log or should and own none: `launch` (the probe), `blas`, `spatial`,
`nonlinear`, `sparse`, `graph_primitives`, `cluster`, `benchmark_harness`.

A failure thrown to the caller is logged once where it is made (`operand_check.hpp:59`, `:74`, `:92`, `io.cpp:66`), and
`TaskRecorder` turns a returned runtime `Error` into an exception without logging (`task_recorder.hpp:323-326`). That is
the "log once, by whoever holds it last" rule and stays.

| # | Place | What happens | What to log | Level, category |
| --- | --- | --- | --- | --- |
| D1 | `device_policy.hpp:356-358` | A throwing architecture query returns `Unknown`; tile limits then use the generic path | The query threw, device name, falling back to Unknown | WARN once per device, `launch` (new) |
| D2 | `device_policy.hpp:394-396`, `:408-409` | A device with no sub-group size gets 1; one with no cache size gets 256 KiB | Which field was assumed and the value | WARN once per device, `launch` |
| D3 | `device_policy.hpp:419-440` (`profile_for`, first sight) | The probed profile is never stated; a tuning report cannot say which device it describes | One line per device: units, sub-group, cache reported or assumed, architecture | INFO once per device, `launch` |
| D4 | `gemm_tile_pin.hpp:62-70` | `SB_GEMM_TILE` changes which kernel runs; an accepted pin is silent (a rejected one warns once, `gemm.hpp:957-974`) | The pin read and the shape it selected | INFO once per process, `blas_seam_kernels` |
| K1 | `gemm.hpp:1048-1051` | `WARN` on every call when the local tile does not fit, so a training or inference loop writes one record per GEMM per step | Same text, once per device and shape class | WARN once per (device, tile) through SushiLog RW-3 |
| K2 | `gemm.hpp:887`, `:1003`, `:1016`, `:1054`, `gemm_packed.hpp:130` | The variant chosen (local tiled, depth split, joint matrix, no local, packed) is a `DEBUG` record per launch, and the only place it is stated | Move the variant into the profile as a span attribute or an instant (pass two); then delete the record | DEBUG now; none after |
| K3 | 28 `SB_LOG_DEBUG` in `modules/kernels` (list: grep `SB_LOG_DEBUG`) | One record per launch, compiled into release because the `NDEBUG` ceiling is `DEBUG` (`logger.hpp:36-38`) | Nothing: shape and variant are the profiler's data; remove after K2 | DEBUG, `blas_kernels`, `sparse_kernels` |
| K4 | 15 `SB_LOG_TRACE` in task bodies (`map_task.hpp:154`, `index_task.hpp:127`, `reductions/dispatch.hpp:218`, ...) | One record per run of a node; compiled out of release, present in debug | Nothing: entry of a function | TRACE; remove |
| I1 | `io.cpp:500-505` | `catch (...)` removes the partial file with `std::remove`, ignores its result, rethrows | A failed cleanup: path and that the partial file stays | WARN, `io` |
| S1 | `scratch_shelf.hpp:142-167` | A scratch lease above 4096 B is allocated and freed every time (`K_SCRATCH_UNPOOLED`); a lease that finds its class empty allocates; neither is counted | Counters: pool hits, misses and unpooled leases per interval (SushiLog RW-4); one WARN if a device allocation fails is already a throw | counter, `engine` |
| E1 | `engine.cpp:38` | `INFO` per `Engine` constructed; a process that builds one per test or per graph writes many | Keep; one record per Engine is a lifecycle event | INFO, `engine` |
| P1 | `node_stamper.hpp:89-106` | A primitive called with no options records an unnamed node with no operation | Not a log: name the nodes (pass two) so reports and spans have identity | none |

Silent behaviour that is correct and needs no record: `KernelPairDispatcher` picks the SYCL kernel under `Auto` by
definition (`execution_mode.hpp`), and `ForceReference` is a mode the caller chose.

The logging tasks wait for a second pass (runtime D18). Rows D1 to D4, I1 and K1 need no SushiLog change except K1,
which waits on SushiLog RW-3 or ships a local `std::once_flag` as `gemm.hpp:957` does.

## 3. What SushiLog lacks

SushiBLAS's needs that SushiLog's `REMAINING_WORK.md` already holds: log once per key (RW-3: K1, D1 to D4), a counter
per interval (RW-4: S1), a shipped capture sink for the tests that assert a record (RW-8), the dropped and per-level
counts (RW-11), the thread name on a record (RW-7, for a record made on a runtime worker).

Gaps not in RW-3 to RW-12:

| Gap | Evidence | Why it matters |
| --- | --- | --- |
| SG-1 The compile-time ceiling is one switch per project | `logger.hpp:36-38`: `SB_ENABLE_LOGGING` 0 or 1, `DEBUG` under `NDEBUG` | A project cannot keep its 28 per-launch `DEBUG` records out of release while keeping its other `DEBUG` records. Pass two removes those records, so this is a note, not a request |

Nothing else. In particular SushiBLAS needs no new record path for a real-time thread: its task bodies run on the runtime's
workers and log at most once per key after K1 and K3 are fixed.

## 4. What SushiProfiler v0.2.0 lacks for SushiBLAS

v0.2.0 closes the shared library, the source contract, the registration pattern and the zone runs
(`CLIENT_CONTRACT/SPEC.md`, "Decisions"). Its `REMAINING_WORK.md` holds RW-3, RW-6 to RW-22. The following are in neither.

| Gap | Evidence | What SushiBLAS needs |
| --- | --- | --- |
| G-B1 A library cannot add an analysis | `Registry` has factories for sources, stream writers, record writers and contributors only (`registry.hpp:56-75`); `Session::report` and `Session::open` build `AnalysisCatalog` and call `register_builtin_analyses` (`session.cpp:131-134`, `:159-160`); `sp record show` does the same (`show_command.cpp:90-91`). `CAPTURE_AND_SESSION.md:30` says the registry holds analyses | `roofline` cannot be registered by SushiBLAS, and `sp` on a disk record cannot know it. BD3 chooses a generic analysis in SushiProfiler, which closes the need without closing the gap; the gap stays open for SushiAI's own analyses |
| G-B2 A session runs analyses with default options | `AnalysisRunner(catalog).run(record, config_.analyses, AnalysisOptions{})` (`session.cpp:134`) | `launch_floor_ns` (the ceilings' empty-kernel time) and `group_by` (`operation`, `node`) cannot reach `short_spans` and `top_spans` in a session. They need keys in `ProfilerConfig` |
| G-B3 No analysis reads a span's cost attributes or a per-device ceiling | `analysis/README.md`, the table of analyses; facts are strings (`record_facts.hpp:48`) | A generic `roofline`: attribute names for work, bytes and class as options, ceilings from record facts named by the track, a column for "no ceiling known" |
| G-B4 A source cannot read or annotate another source's spans | `period_assembler.cpp:287-290` builds one `SpanSink` per source over the same table; the sink only appends (`span_sink.hpp`, `append`, `set_*`) | Not needed by BD1 (a). It blocks the alternative BD1 (b) and SushiAI's `sushiai.attribution` as CLIENTS.md describes it ("adds `ir_node` and `op` to each span"). Raised for the SushiAI analysis; no SushiBLAS task waits on it |
| G-B5 No way to carry a ceilings file into the record except a string fact | `record_facts.hpp:41-52` | A record fact table with numbers would avoid parsing. A string with a fixed number format is enough for pass one |

## 5. Decisions

Taken from SushiRuntime's section 8 and not reopened, with SushiBLAS's form:

| Runtime decision | SushiBLAS form |
| --- | --- |
| D1 foundation front and one includer | A foundation module `profile` holds `SB_PROFILE_LEVEL`, the `SB_ZONE` family and the attribute keys; it is the only includer of SushiProfiler below the module that registers the parts |
| D2, D23 shared link, pin v0.2.0 | `sushiblas` is a static archive (`modules/library/sushiblas/CMakeLists.txt:9`), so its code lands in each consumer's binary and links the one shared SushiProfiler |
| D8 level 2 in every build type | `SB_PROFILE_LEVEL` default 2, per-element sites 3 (none exist in pass one) |
| D12 registration | `register_profiler_parts` returning an owning handle |
| D11, D14, D18 | Device timing from `sushiruntime.device`; a failure status logged once by whoever holds it last; logging waits for pass two |
| D22 | BD13 below |
| D24 | The four tests move to a test observer (PB-7) |

New questions. A mark in the third column means the decision changes a public interface or a module boundary, so it goes
to the owner. The recommendation is mine and is the one that holds as the stack grows.

| # | Question, options | Recommendation and why | Owner |
| --- | --- | --- | --- |
| BD1 | How cost reaches a span. (a) SushiRuntime carries an opaque `NodeCost{flops, bytes, work_class}` on `NodeOptions`, the node table and the plan view; its device source writes the three attributes. (b) A SushiBLAS source annotates the runtime's spans. (c) The cost goes in the 8 parameter slots | (a). The cost then travels with the node through plan rebuilds, `clear()` and replays, and there is no second table to keep in step with registration indices. The runtime stays ignorant of BLAS: three numbers it copies. (b) needs G-B4 and a map from rule to cost; (c) has no free slots in GEMM (6 of 8 used, `blas_metadata.hpp`) and is not general | yes: `NodeOptions`, `NodeTableRow`, `PlanNode` of SushiRuntime change; additive |
| BD2 | Where cost rules live. (a) Each seam computes `Cost` and sets it. (b) Each seam states shape facts in a `CostShape` on `TaskDescription`; a new foundation module `cost` holds one rule per operation id; `TaskRecorder::insert` computes the cost and the footprint bytes once. (c) A virtual rule object per operation | (b). `insert` is the one recording point (1.2), so the rule is applied once, a seam states facts it already holds, and the knowledge "what an `exp` costs" sits in one module that has no SYCL include and builds without device passes (SYCL compile-time work). (a) spreads rules over 25 sites; (c) is an interface per kind for a closed table | yes: new module `cost`, new member `TaskDescription::shape` |
| BD3 | Where `roofline` lives. (a) In SushiBLAS, registered by a library analysis factory. (b) Generic in SushiProfiler: attribute names and the ceiling facts are data. (c) A SushiBLAS tool that reads records | (b). An analysis in SushiBLAS cannot reach `sp record show` (G-B1) without `sp` linking every client, which inverts the dependency. A generic analysis also serves SushiDSP's budget work and SushiAI. It knows no kernel: it is told which attributes mean work and bytes. CLIENTS.md says "in SushiBLAS" and must change | yes: it changes the approved design and SushiProfiler's analysis module |
| BD4 | How `device_ceilings` times. (a) SYCL events on a queue it builds. (b) A graph run with a `RunObserver`, reading device spans | (b). It measures through the path the spans come from, uses only the runtime's public API and needs no queue of its own. It waits on the runtime's PF-5 | no |
| BD5 | How the ceilings reach the record. (a) The environment contributor writes them as facts from the file. (b) The analysis reads the file path from options. (c) The cost attribute carries the ceiling | (a). Analyses read records only; (b) breaks that and makes an old record unreadable once the file moves. Fact names embed the exact track name the runtime source uses, `ceiling.<track>.<class>` | no, once the track name is fixed by the runtime's PF-6 |
| BD6 | FLOP and byte conventions (mine). FLOPs: GEMM 2mnk per batch, complex 8mnk; an elementwise function counts the operations of its formula, a transcendental as 1; an epilogue adds its per-element count; a reduction n-1 per output. Bytes: declared footprint. A task that returns an empty event states 0 | Fixed in the `cost` README and pinned by test values, so a change is a visible diff | no |
| BD7 | The IR join. (a) SushiAI takes `[first,count)` from `graph.node_count()` and maps through `rule_registration`. (b) `TaskHandle` carries the rule position | (a). It needs no change and `TaskHandle` is a per-recorder counter (1.2). (b) would add a public member for a join SushiAI can make itself | no |
| BD8 | Multi-stage bodies (1.2). (a) Cost them with the last stage's span. (b) Declare them `uncosted(several_submits)` until a timing path covers the whole body. (c) Make the kernels return a joined event | (b). (a) prints a wrong fraction with a straight face; (c) has no SYCL way to give a joined event with trustworthy profiling times. The inference path (GEMM, im2col, pool, upsample, channel affine, elementwise) is single-submit; PB-6 audits each costed operation | no |
| BD9 | Tier and shape of the module that registers the parts. (a) A new tier `surface`. (b) A module `profiling` in tier `engine`, depending on `launch`, `core`, `profile`, not on `engine`. (c) In `library` | (b). The tier list has no surface tier and `library` holds only composition (`README` of the module, `modules/README.md`). `engine` is the first tier that uses `SushiRuntime::Runtime`, which the contributor needs for the device list | yes: new module and its tier edge |
| BD10 | The registration signature. `register_profiler_parts(SushiProfiler::Registry&, SushiRuntime::Runtime&, std::string_view ceilings_folder)` returns `[[nodiscard]] SushiBLASProfilerParts`, an owning handle; the contributor states `sushiblas.version`, build facts, default execution mode, tile pin, layout and the ceilings. SushiAI calls it beside the runtime's | The same shape as the runtime's D12; a folder, not a file, so several devices find their own file | yes: a public function |
| BD11 | `SB_ENABLE_PROFILER` default ON, `SB_PROFILE_LEVEL` default 2, as the runtime's | Same options in `cmake/`; off, the front's macros expand to nothing and `profiling` is not built | no |
| BD12 | The benchmark program's home and door. (a) `tests/.../benchmark_*.cpp` like the eleven today. (b) `applications/benchmarks/device_ceilings/`, run by `sb run device_ceilings` now and by `sb bench <name>` later | (b). `performance` makes a scenario benchmark an application with its own `main()` and options; `check_probes.py` expects that folder. `sb bench` takes no name today (`cli/README.md`, "sb bench"); adding one is a CLI change and goes to pass two | the later `sb bench <name>`: yes, a CLI option |
| BD13 | The design topic. (a) Open `docs/design/PROFILING/README.md` with the phases of this plan and the second-pass items as backlog entries. (b) None | (a), as the runtime's D22: the work folder then carries `Implements` | no |

## 6. What this analysis did not cover

- Zones inside SushiBLAS (record time: `TaskRecorder::insert`, the recording lock, scratch leases) and the benchmark harness
  on the timeline `bench.iteration`. Both are pass two. Record time is paid once per plan in SushiAI's compiled step, so
  it is not part of the inference profile.
- The cost of operations other than those SushiAI's inference lowering uses; they are listed and declared in PB-6 but
  their rules are written from the operation's definition, not measured.
- The multi-submit audit of 1.2 beyond `reduce_1d.hpp` and `gemm.hpp`; PB-6 does it per costed operation.
- Which kernel variant ran (K2). It is decided inside the task body at run time, so it cannot be stamped at record time.
  A profiler instant or a variant attribute from the body needs a design on both sides.
- The distributed `cluster` module, the sparse and signal families' cost rules beyond a declaration, and `IO`.
- Overhead: no number here is measured. The cost of stamping three integers per `NodeOptions` is one `insert` call per
  record; the runtime's PF-13 measures the whole path.
- GPU behaviour. Everything about a device ceiling is untested until the owner runs `device_ceilings` on the 3080 Ti.

## 7. SOLID breaches and dependencies seen

| Where | Breach | Cost |
| --- | --- | --- |
| `modules/foundation/launch/include/SushiBLAS/launch/device_policy.hpp` | One header holds the probe, a process-wide cache with a `thread_local` pointer, the cache-blocking maths and the launch geometry, and it logs nothing for the fallbacks it takes (rows D1 to D3). Seen, not changed | A device that falls back is invisible; a split into probe, policy and geometry would let each be tested alone |
| `modules/kernels/*` (28 `SB_LOG_DEBUG`) | Kernels log per launch, which `logging` forbids per frame and which the profiler's data replaces | A relaxed load and a branch per launch in release, and a record in the log of every step when `debug` is on |
| `modules/foundation/graph_primitives/.../node_stamper.hpp:89-106` | A node with no name and no operation id when a call states no options | A span with no identity; no cost rule can apply |
| `engine/task_handle.hpp:77` | A name (`TaskHandle`) that suggests identity but carries a per-recorder counter that is not the rule position | A reader may join on it and get a wrong node |
| `modules/operations/benchmark_harness` in tier `operations` | A benchmark-only module in the tier of the operations, with dependencies on `engine`, `io` and `recording_graph`; its timer measures record plus run on the host | It cannot serve a device measurement and sits above what it times. Seen, not changed |
| `engine/engine/include/.../scratch_shelf.hpp:142-167` | Allocation and pooling in a header template with no counter | A pool miss or an unpooled lease is invisible |
| `cli/sushistack.deps.toml:25-27` | Pins SushiLog `v0.2.2`; the runtime's SPEC records `v0.2.3` as the last tag | A drift between two repositories of one process |

Dependencies this work adds: SushiProfiler, through `profile` only; the `cost` module has none. Dependencies seen that the
code may not need: none confirmed. `benchmark_harness` depends on `io`; whether any benchmark program uses it through the
harness was not checked.

SOLID breaches this analysis made: none (it wrote two documents). Breaches pass one would make if the recommendations are
refused: BD2 (a) would place a rule in every seam.

## 8. Rulings

The owner delegated the decisions of section 5 to the orchestrator on 2026-10-26, with every
library's own timing leaving it for SushiProfiler and all code written before any build.

| # | Ruling |
| --- | --- |
| BD1 | SushiRuntime carries a generic `NodeCost{flops, bytes}` on `NodeOptions`, the node table and `PlanNode`, and its device source writes them as the span attributes `flops` and `bytes`. No `work_class`: a cost is two numbers any library can state, and a library-specific class stays in the node's name. Folded into the runtime's PF-3, PF-4 and PF-6 |
| BD2 | As recommended: a `cost` module, shape facts through `TaskDescription::shape`, one rule applied in `insert` |
| BD3 | As recommended: `roofline` is a generic SushiProfiler analysis reading `flops` and `bytes` attributes and ceilings stated as record facts; it lands in SushiProfiler v0.2.0 |
| BD4 to BD8 | As recommended |
| BD9 | As recommended: `profiling` in tier `engine` |
| BD10 | As recommended: `register_profiler_parts(Registry&, Runtime&, std::string_view ceilings_folder)` returning an owning handle |
| BD12 | Later; the ceilings program runs through `sb run` |
| Multi-kernel bodies | A body that submits several kernels returns an event that completes with all of them, so the span covers the whole body; PB-6 fixes each one it finds |
| Scope | Every PB task of `PLAN.md`, then the second pass PB-20 to PB-26 and LB-1 to LB-7, written before any build |
