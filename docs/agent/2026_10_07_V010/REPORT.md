# Report: the v0.1.0 interface break

Written on 2026-10-07 against HEAD `82504db` and the working tree of that evening.

## 1. Nothing here has been built or run

Nothing written since the interface break began has been built or run. The last build and test
run on record is the CUDA run logged at 14:53 on 2026-10-07, before the first package was
committed (`0b9ec90`, 16:26). The owner forbade builds after that and builds at the end. Every
commit of waves 0 to 11 is source that was written and, at most, parsed by the compiler front
end. Whether it compiles as a whole, links, passes its tests or runs faster than what it replaced
is not known.

Three consequences for a reader of this folder:

- A sentence in `SPEC.md` or `PLAN.md` that says a test passes or a wave "closes with a build"
  states what was planned. No wave was closed by a build.
- The CMake lines of the new sources and tests are in the tree and were never configured.
- The changes ride on `RuntimeContext` entry points that sit uncommitted in SushiRuntime's
  working tree (SR-1), so a build needs that checkout.

## 2. Verification that exists

| What | When | Result | Source |
| --- | --- | --- | --- |
| CUDA build and test run on an RTX 3080 Ti, before the first package | 2026-10-07 14:15, repeated 14:53 | 792 tests: 790 passed, `Regression_LoggerDeathTest.AssertWritesItsMessageBeforeAborting` skipped (it needs a debug build), `Unit_Level2SyclParityTest.SymvMatchesReferenceLarge<float>` failed | the two test logs in the session scratchpad; `PLAN`, section 0; `DESIGN_REVIEW`, section 7.3 |
| Front-end sweep (`clang++ -fsyntax-only`) of every translation unit | at commit `1746b03`, the end of wave 2 | all 250 units of the library, tests, examples and benchmarks parsed | the message of `1746b03` |
| File-by-file front-end check by each package's author and its reviewer | waves 3 to 11 | reported clean per file; the tree as a whole was not swept | the commit messages from `9723bdc` on; `OWNER_DECISIONS_LATE`, first point |
| Front-end check of twelve files after the `UnsupportedExecution` change | working tree, uncommitted | each exited 0 | `EXCEPTIONS`, "Syntax check" |
| `python tools/layering/check_layering.py .` | 2026-10-07, working tree, run for this report | 0 findings, exit 0; output below | this section |
| Python tests of the version checker and of the `sb` device gate | before wave 0 | 90 and 110 tests passed | `PKG_RELEASE_MECHANICS`, `PKG_DEVICE_GATE` |
| The two plan checkers: disjoint file sets per wave, ordered dependencies | when each plan was written | 0 problems of 470 and of 160 file claims | `PLAN`, section 7; `PLAN_EXT`, section 6, both in `PLAN.md` |
| Design code parsed by its author | per design | which blocks were parsed and which were not is in each design's last section | `SPEC.md` |

The front-end sweep proves that 250 units parse at `1746b03`. It says nothing about the 16
commits of this work that follow it, and nothing about linking or behaviour at any commit.

```
$ python tools/layering/check_layering.py . --report
rule_declared_tiers: 0
rule_upward_includes: 0
rule_private_reach: 0
rule_support_independence: 0
rule_kernel_includes: 0
rule_engine_header_includes: 0
rule_seam_engine_includes: 0
rule_operation_engine_calls: 0
rule_view_engine_calls: 0
rule_graph_insertions: 0
rule_task_dependencies: 0
total: 0
```

The failing SYMV parity test of the GPU run is what `DESIGN_BLAS`, sections 4.8 and 7.10, and
commits `f52e10d` and `63a0c90` were written to fix. The fix has not been run on any device.

## 3. Commits by wave

`git log 535e83c..HEAD` holds 41 commits. Thirty-one belong to this work, five came before the
plan, and five are the documentation and CI work of another session in the same tree. The
package column was read from each commit's file list against the package rows of `PLAN.md`; the
folder holds no dispatch log to check it against.

| Wave | Commit | Subject | Packages |
| --- | --- | --- | --- |
| before the plan | `5993a1e` | build(tools): check every copy of the version against project() | `PKG_RELEASE_MECHANICS` |
| before the plan | `9a32760` | feat(cli): pick the SYCL device with --device on sb test and sb run | `PKG_DEVICE_GATE` |
| before the plan | `358e746` | Revert of `9a32760` | the owner's removal of `--device` |
| before the plan | `813676e` | docs(changelog): record the version checker and the notice | none |
| before the plan | `dd5cdfc` | docs(cli): say that an empty SB_SYCL_TARGETS follows SushiRuntime | none; the tree the designs were read against |
| 0 | `0b9ec90` | fix(cli): write the API site under the build tree in sb docs | SM-15 |
| 0 | `e729c55` | build(tools): hold the changelog release heading to the version | SM-12 |
| 0 | `b7468cd` | feat(support)!: take a launch shape as an argument and drop the unread profile field | SM-4 |
| 0 | `e782864` | feat(engine): add rank, extent, scalar and device dtype checks | SM-5 |
| 0 | `1cb6e83` | feat(blas): add the flag enums, the scalar type and the kernel bricks | BLAS-1 |
| 0 | `b455e63` | feat(engine): add a scratch shelf and pool for device blocks | SM-2 |
| 0 | none here | SR-1, in SushiRuntime's working tree, uncommitted there | SR-1 |
| 1 | `b42c79b` | build(layering): check views, seams and task dependencies | R7 |
| 1 | `703606d` | feat(engine)!: add the task recorder, the task handle and the tensor factory | R1 |
| 2 | `1e4fa63` | refactor(engine)!: make Engine a composition root over the recorder | R2 |
| 2 | `2176a1f` | refactor(math)!: record elementwise, nonlinear and logic through the recorder | R3 |
| 2 | `df6a333` | refactor(math)!: record reductions and random through the recorder | R4 |
| 2 | `1e26e17` | refactor(spatial)!: record the spatial family through the recorder | R5 |
| 2 | `cc30ffa` | refactor(blas)!: record BLAS through the recorder and drop the sparse base | R6 |
| 2 | `b0a2bef` | feat(blas): add the vector operand, argument and metadata bricks | BLAS-2 |
| 2 | `1746b03` | refactor(tests)!: move every call site onto the views and the graph probe | R8 |
| 3 | `9723bdc` | feat(storage): align every tensor allocation to 128 bytes | SM-1 |
| 3 | `08a0355` | perf(io): print and load without a full-size staging copy | SM-9 |
| 3 | `1b0b0aa` | fix(spatial)!: throw std::invalid_argument for a refused operand | SM-8 |
| 3 | `ad1e825` | feat(reductions)!: fold var, std, logsumexp and the arg reductions in parallel | SM-7 |
| 3 | `2e1ed7e` | test(engine): cover the recorder, the handle, the factory and the scratch lease | R9, SM-3 |
| 3 | `5fa67f9` | fix(math)!: throw std::invalid_argument for every record-time refusal | SM-6 |
| 4 | `f52e10d` | feat(blas)!: give the kernels enum flags, a conjugation policy and wide reference sums | BLAS-3, BLAS-4, BLAS-5 |
| 5 and 6 | `968d7b9` | feat(blas)!: take enum flags and BlasScalar, and add the complex and Hermitian routines | BLAS-6, BLAS-7, BLAS-8, BLAS-9 |
| 6 | `63a0c90` | test(blas)!: move the BLAS tests onto the enums and measure parity against a truth | BLAS-10, BLAS-11 |
| YOLO26, its wave 7 | `6082663` | fix(spatial): sum HALF in FLOAT32 in the window kernels and refuse HALF without fp16 | YL-1, YL-2 |
| YOLO26, its wave 7 | `5f403bd` | feat(engine): add the host loops and the declared-reads brick | YL-3, YL-4 |
| YOLO26, its waves 8 and 9 | `1707272` | feat(spatial): add the windowed product and its two gradients | YL-5, YL-6, YL-7 |
| YOLO26, its wave 9 | `52189a2` | feat(math)!: give the five device-only families a host reference path and declared reads | YL-8, YL-9, YL-10 |
| 7 to 10 | `2c75a0c` | feat(sparse): add SparseOps with CSR products, a triangular solve and the Level 1 routines | SPX-1 to SPX-12, without SPX-7's line in `SushiBLAS.h` |
| 7 to 11 | `9a25527` | perf(blas): add a cache-blocked, panel-packed GEMM path for CPU devices | GMX-1 to GMX-7, the new files and `device_policy.hpp` |
| 7 to 11 | `82504db` | feat(blas): complete Level 3, add band, packed and rotation routines and parallel triangular solves | L3X-1 to L3X-9, BPX-1 to BPX-13; the edits of GMX-3, GMX-5 and GMX-7 to `utils.hpp`, `dispatch.hpp`, `syrk.hpp`, `syrk.cpp` and `symm.cpp`; SPX-7's line in `SushiBLAS.h`; the CMake lines of the second round |

The other session's five commits, listed so the count closes: `2beddef` (the FAQ page),
`4319ec8` (`sb docs bundle`), `d452821` (the release workflow), `5ca5294` and `e6e837d` (two
review findings of that rollout).

Uncommitted in the working tree when this was written: `include/SushiBLAS/core/errors.hpp`,
`tests/functional/unit/test_unsupported_execution.cpp` and edits to tracked files under
`include/`, `src/` and `tests/`. They are the third late decision, the refusal type derived from `std::invalid_argument`. `EXCEPTIONS`
describes the tree with that change applied.

## 4. Packages and their review records

A reviewer's verdict is given here only where this folder or a commit message records one. For
most packages the only record is one sentence of the commit message, and the reviewer's own
report was returned to the orchestrator and is not in the folder.

| Packages | Review record |
| --- | --- |
| The three first-round designs | `DESIGN_REVIEW`, section 1: they "can be dispatched in the order of section 4 once the owner has answered the points of section 5". The owner answered the same day (section 5) |
| The four second-round designs | `PLAN_EXT`, section 4: seven conflicts between the designs, five settled by an edit or by wave order and two left to the owner; no numerical defect found in a kernel, on hand-worked examples; nothing compiled by the review |
| `DESIGN_YOLO26` | none in the folder |
| `PKG_RELEASE_MECHANICS` (`5993a1e`) | reviewed; the reviewer re-ran the commands, got the same results and changed four things (`PKG_RELEASE_MECHANICS`, "Review, 2026-10-07") |
| `PKG_DEVICE_GATE` (`9a32760`) | reviewed; the reviewer re-ran the commands, the counts matched, and it changed three things (`PKG_DEVICE_GATE`, "Review"). The owner then had the feature reverted |
| Wave 0: SM-2, SM-4, SM-5, SM-12, SM-15, BLAS-1; SR-1 | none in the folder |
| Wave 1: R1, R7 | none in the folder |
| Wave 2: R2, R3, R4, R5, R6, R8, BLAS-2 | none per package. The wave as a whole is covered by the 250-unit front-end sweep at `1746b03` |
| Wave 3: R9, SM-1, SM-3, SM-6, SM-7, SM-8, SM-9 | each commit message: "Checked file by file with the compiler front end by its author and its reviewer". No verdict beyond that |
| Waves 4 to 6: BLAS-3 to BLAS-11 | the same sentence in `f52e10d`, `968d7b9` and `63a0c90`. No verdict beyond that |
| YL-1 to YL-10 | the same sentence in `6082663`, `5f403bd`, `1707272` and `52189a2`. No verdict beyond that |
| SPX-1 to SPX-12 | the same sentence in `2c75a0c`. No verdict beyond that |
| GMX-1 to GMX-7 | the same sentence in `9a25527`. No verdict beyond that |
| L3X-1 to L3X-9, BPX-1 to BPX-13 | the same sentence in `82504db`. No verdict beyond that |
| The `UnsupportedExecution` change | none in the folder; uncommitted |

`PLAN_EXT`, section 5, point 6, asked for a `medium` review of six numeric packages with the
design open: L3X-2, L3X-5, L3X-6, GMX-2, SPX-6 and BPX-1. The folder does not record whether
those reviews were of that kind. The same holds for the open question on `drotmg.f` (point 9)
and for the two `Core::IndexBase` values 221 and 222, which SPX-1's implementer was to check
(point 11).

## 5. Defects reviewers found and fixed

Recorded in this folder:

| Found by | Where it is written | What |
| --- | --- | --- |
| The first design review | `DESIGN_REVIEW`, section 2 (a) and (c) | Two SOLID findings fixed in the designs (A1: a second operand-set brick; A2: a protected recorder member) and twelve C++ mismatches between the three designs (c1 to c12), each fixed in the design text. Four SOLID findings were left or passed to the owner (A3 to A6) |
| The second design review | `PLAN_EXT`, section 4 | The band and packed sibling tests compared results bit for bit with dense routines the Level 3 design had made parallel; the triangular cases were rewritten on exact integer data. Two designs each wanted a timing helper; one owner was set. Three designs anchored CMake lines on the same line. The sparse SYCL kernel's log line was renamed. Four sentences that stated an unmeasured cost as fact were reworded |
| The reviewer of `PKG_DEVICE_GATE` | `PKG_DEVICE_GATE`, "Review" | `device_selector` stripped blanks its documentation said it passed through; one line over the column limit; an unstated rule on the case of the `selector:` prefix |
| The reviewer of `PKG_RELEASE_MECHANICS` | `PKG_RELEASE_MECHANICS`, "Review, 2026-10-07" | Four parsing cases of `check_version.py` (a commented `project()`, a four-number version, a single-quoted TOML value, a table header that did not close the previous table), with six tests written after the fixes; comment placement and line length; `NOTICE.md` said three third-party works ship where two do |

Not recorded in this folder: what the reviewers of waves 0 to 11 found in the source packages.
Those findings were in the reports returned to the orchestrator.

Found by this work and left open, in `EXCEPTIONS`: four `@throws` lines that name
`std::runtime_error` where the code throws `std::invalid_argument`, and three READMEs that still
describe a refusal by `ExecutionMode::VERIFY`, which no longer exists.

## 6. Deviations from the designs and the plans

Each row is something the tree or the history shows that differs from what `SPEC.md` or
`PLAN.md` says. Where an implementer departed from a design's code is not recorded in this
folder and is not listed.

| Planned or designed | What happened | Evidence |
| --- | --- | --- |
| Waves 0, 2, 3, 6, 8, 10 and 11 close with `sb build`, `sb test` and the five checks; waves 6, 8, 10 and 11 again on the CUDA build | no build and no test run; one front-end sweep at the end of wave 2, then file-by-file checks | `OWNER_DECISIONS_LATE`, first point; the commit messages |
| `sb --device` (`PKG_DEVICE_GATE`; SM-10 and SM-11 of the support design) | landed in `9a32760`, reverted in `358e746`; SM-10 and SM-11 deleted, SM-15 keeps one line | `PLAN`, section 0 |
| A family without a reference kernel refuses `FORCE_REFERENCE` at record time (answer E; G4 of `YOLO26_COMPAT`) | written by `5fa67f9`, then replaced by a host reference path in `52189a2`, which removes `TaskRecorder::record` and `require_sycl_path` | `OWNER_DECISIONS_LATE`, fifth point; `DESIGN_YOLO26`, Y4 |
| The remaining mode and device refusals throw `std::invalid_argument` | they throw `SushiBLAS::UnsupportedExecution`, derived from it; uncommitted | `OWNER_DECISIONS_LATE`, third point; `EXCEPTIONS` |
| `symm`, `hemm`, `syr2k`, `her2k`, CPU GEMM packing, parallel TRMV, TRSV and TRSM, and a record-time HALF check for a device without fp16 wait until after 0.1.0 | all are in the tree | `PLAN`, section 5; `OWNER_DECISIONS_LATE`, sixth and seventh points; `82504db`, `9a25527`, `6082663` |
| The second round lands in five waves that mix the four designs; the YOLO26 packages in three waves of their own | one commit per design, the YOLO26 packages in four; the shared-file edits of GMX-3, GMX-5 and GMX-7 are in the Level 3 commit | section 3 above |
| The orchestrator adds the CMake lines wave by wave, each proven by that wave's build | the lines are in `b455e63`, `ad1e825`, `2e1ed7e`, `5f403bd` and `82504db`; none was configured | the file lists of those commits |
| SR-1 lands in SushiRuntime in wave 0 and is in the runtime tag the release pins | committed in SushiRuntime as `2728826`; no runtime tag exists yet | the message of `9723bdc` |
| The device refusals of SYRK and SYR2K | still `std::runtime_error`, thrown at execute time from a kernel, which may not include the engine header that declares the new type | `EXCEPTIONS`, "Execute time" |
| Enumerators in PascalCase and constants with the `K_` prefix across the library | done in `67ded36`, by script; fifteen translation units parsed afterwards | `OWNER_DECISIONS_LATE`, second point; section 8 below |

## 7. Not done

- The build, the test run on the CPU build and on the CUDA build, and every benchmark. No
  performance statement of the second-round designs has a measurement behind it.
- R11 and YL-11, the SushiAI migration, which belongs to the sushiai session.
- SM-13, the build configuration of the release: `src/CMakeLists.txt` has no `experimental`
  exclusion in its install rule and `cmake/` has no `SushiRuntime 0.4` pin.
- Answers to the open points of `PLAN`, section 6, and `PLAN_EXT`, section 5.

## 8. Naming table

Under the heading `NAMING_TABLE` below, as the naming sweep of `67ded36` wrote it. The same rows
are in `tools/naming/sushiblas_0_1_0.tsv` for a consumer's rename tool.

## 9. Exception table

Under the heading `EXCEPTIONS` below: the types, the four helpers that throw, the sites of
`UnsupportedExecution`, and every layer's exceptions at record time, at execute time and outside
both.

## 10. Records kept with this report

The records below are as their authors wrote them, apart from the heading level and the heading
that carries each old file name. `SPEC.md`, under "Names used in this folder", maps the names.

| Heading | What it is |
| --- | --- |
| `scout/blas-flags`, `scout/engine-srp`, `scout/facade`, `scout/gpu-perf`, `scout/memory`, `scout/numerics-left`, `scout/release` | the seven read-only surveys of the tree at `dd5cdfc` that the first-round designs were read against |
| `PKG_DEVICE_GATE` | the record of the reverted `--device` package, with its review |
| `PKG_RELEASE_MECHANICS` | the record of the version checker and `NOTICE.md`, with its review |
| `EXCEPTIONS` | exceptions by layer |
| `NAMING_TABLE` | the renames of the naming sweep |

## 11. The merge of this folder

On 2026-10-07 the folder held 25 Markdown files, seven of them in a `scout/` subfolder.
`check_docs_layout.py` reported 18 `rule_work_folders` findings and seven `rule_document_names`
findings for it. The documentation skill allows a work folder `SPEC.md`, `PLAN.md` and
`REPORT.md` and has no rule for an appendix, a subfolder or an archive of live work, so the
files were merged into those three. A script rebuilt each old file from the merged text and
compared it byte for byte with the file on disk before the old file was deleted.

Paths that cited an old file and now need the new one, outside what this merge could write:

| Citing file | Cites | Now |
| --- | --- | --- |
| `tests/functional/unit/test_blas_band.cpp:343`, `test_blas_packed.cpp:336` | `docs/agent/2026_10_07_V010/DESIGN_BLAS_BAND_PACKED.md` | `SPEC.md`, heading `DESIGN_BLAS_BAND_PACKED` |
| `tests/functional/unit/test_packed_product.cpp:169`, `test_packed_symm.cpp:181`, `test_packed_syrk.cpp:124` | `docs/agent/2026_10_07_V010/DESIGN_GEMM_CPU.md` | `SPEC.md`, heading `DESIGN_GEMM_CPU` |
| `tests/functional/unit/test_packed_product_plan.cpp:54` | `DESIGN_GEMM_CPU.md, section 3.2`, by bare name | the same heading; the section number holds |
| `tests/functional/unit/test_triangular_parallel.cpp:874`, `:1430`, `test_triangular_plan.cpp:245` | `DESIGN_BLAS_L3.md`, sections 7.6, 7.2 and 7.4, by bare name | `SPEC.md`, heading `DESIGN_BLAS_L3`; the section numbers hold |
| `docs/reference/KNOWN_ISSUES.md:13`, `:14` | `PLAN_EXT.md` and `OWNER_DECISIONS_LATE.md`, by bare name | `PLAN.md`, heading `PLAN_EXT`; `SPEC.md`, heading `OWNER_DECISIONS_LATE` |
| `docs/reference/KNOWN_ISSUES.md:27` | eight `DESIGN_*.md` files with a section number each, "all under `docs/agent/2026_10_07_V010/`" | `SPEC.md`, the heading of each name; the section numbers hold |
| `docs/reference/KNOWN_ISSUES.md:66` | the row that reports this folder's layout findings | the row is no longer true and goes |
| `docs/design/REMAINING_WORK.md:6` | "Design names refer to the files under `docs/agent/2026_10_07_V010/`" | the names are headings of `SPEC.md` and `PLAN.md` |
| `docs/design/REMAINING_WORK.md:33`, `:36`, `:89`, `:108`, `:124`, `:129`, `:157` | `DESIGN_GEMM_CPU.md`, `DESIGN_BLAS_L3.md`, `DESIGN_YOLO26.md`, `PLAN_EXT.md`, `OWNER_DECISIONS_LATE.md`, `DESIGN_RECORDER.md`, `DESIGN_SUPPORT_MEMORY.md`, by bare name | covered once line 6 is reworded |
| `docs/design/REMAINING_WORK.md:163` | the backlog item that asks for this merge | done; the item goes |
| `sushiai/docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md:953` | `DESIGN_RECORDER.md:540-546` and `:701-765` | `SPEC.md`, lines 1067 to 1073 and 1228 to 1292 |
| the same file, line 954 | `DESIGN_RECORDER.md:864`, `:886`, `:926` | `SPEC.md`, lines 1391, 1413 and 1453 |
| the same file, line 955 | `PLAN.md:350` | `PLAN.md`, line 371 |
| the same file, line 1031 | `DESIGN_RECORDER.md:665-673` | `SPEC.md`, lines 1192 to 1200 |
| `sushiai/docs/agent/2026_10_07_YOLO26/SPEC.md:493`, `sushiai/docs/agent/2026_10_07_RELEASE_READINESS/REPORT.md:78` | `SUSHIAI_CONSTRAINTS.md:47-51` | `SPEC.md`, lines 153 to 157 |
| `sushiai/docs/agent/2026_10_07_RELEASE_READINESS/REPORT.md:166` | `DESIGN_BLAS.md` section 9.4 | `SPEC.md`, heading `DESIGN_BLAS`; the section number holds |
| `sushiai/docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:197` | `SUSHIAI_CONSTRAINTS.md`, by bare name | `SPEC.md`, heading `SUSHIAI_CONSTRAINTS` |

The line numbers in the right column hold while the head of `SPEC.md` and of `PLAN.md` keeps its
length; a line added above a heading moves every one of them. The line numbers of the citing
files were read on 2026-10-07 and move as those files are edited.

Four citations name the folder itself and need no change: `docs/reference/KNOWN_ISSUES.md:6`,
`sushiai/docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:24`, `sushiai/docs/design/PROGRAMME.md:45`
and `sushiai/docs/design/REMAINING_WORK.md:29`.

## scout/blas-flags (Scout: blas-flags)

Read-only survey of the BLAS public surface of D:\Projects\sushiblas. Nothing was written, built or run through `sb`; items 6 (syntax check) and 7 (scratchpad) are vacuous: no file was created, scratchpad size is unchanged (0 bytes from me). The humanizer skill was loaded before writing this report.

PUBLIC SIGNATURES, VERBATIM (19 methods + 1 seam). [B] = positional bool flag, [S] = scalar parameter.

Level1 (D:\Projects\sushiblas\include\SushiBLAS\engine\blas\level1.hpp)
- :53  `sycl::event axpy(double alpha, const Tensor& x, Tensor& y);`  [S alpha double, FIRST parameter]
- :59  `sycl::event dot(const Tensor& x, const Tensor& y, Tensor& result);`
- :65  `sycl::event scal(double alpha, Tensor& x);`  [S alpha double, first]
- :70  `sycl::event copy(const Tensor& x, Tensor& y);`
- :75  `sycl::event swap(Tensor& x, Tensor& y);`
- :82  `sycl::event nrm2(const Tensor& x, Tensor& result);`
- :88  `sycl::event asum(const Tensor& x, Tensor& result);`
- :94  `sycl::event iamax(const Tensor& x, Tensor& result);`
- :100 `sycl::event rot(Tensor& x, Tensor& y, double c, double s);`  [S c, s double, LAST, no defaults]

Level2 (level2.hpp)
- :52  `sycl::event gemv(const Tensor& A, const Tensor& x, Tensor& y, bool transA = false, double alpha = 1.0, double beta = 0.0);`  [B transA] [S alpha, beta]
- :58  `sycl::event ger(const Tensor& x, const Tensor& y, Tensor& A, double alpha = 1.0);`  [S alpha]
- :64  `sycl::event symv(const Tensor& A, const Tensor& x, Tensor& y, bool upper = false, double alpha = 1.0, double beta = 0.0);`  [B upper] [S alpha, beta]
- :72  `sycl::event trmv(const Tensor& A, Tensor& x, bool upper = false, bool transA = false, bool unit_diag = false);`  [B x3]
- :80  `sycl::event trsv(const Tensor& A, Tensor& b, bool upper = false, bool transA = false, bool unit_diag = false);`  [B x3]
- :87  `sycl::event syr(const Tensor& x, Tensor& A, bool upper = false, double alpha = 1.0);`  [B upper] [S alpha]
- :93  `sycl::event syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper = false, double alpha = 1.0);`  [B upper] [S alpha]

Level3 (level3.hpp)
- :51  `sycl::event gemm(const Tensor& A, const Tensor& B, Tensor& C, bool transA = false, bool transB = false, double alpha = 1.0, double beta = 0.0);`  [B x2] [S x2]
- :60  `sycl::event trsm(const Tensor& A, Tensor& B, bool left_side = true, bool upper = false, bool transA = false, bool unit_diag = false, double alpha = 1.0);`  [B x4] [S alpha]
- :70  `sycl::event syrk(const Tensor& A, Tensor& C, bool upper = false, bool transA = false, double alpha = 1.0, double beta = 0.0);`  [B x2] [S x2]

Seam (engine/blas/dispatch.hpp:205-209)
- `template<Domain Dom = Domain::REAL, typename Epilogue> sycl::event execute_gemm(Engine& engine, const Tensor& A, const Tensor& B, Tensor& C, bool transA, bool transB, double alpha, double beta, const char* name, SushiRuntime::Graph::OpID op_id, Epilogue epilogue)`  [B x2] [S x2]
- plus `GemmEpilogue<E>::reads/writes/functor/read_regions/write_regions` (:76-103) and `gemm_epilogue` (:135); no flags.

Others: `BLASOps(Engine&)` (blas.hpp:53); `SparseBLAS` declares no operation (sparse.hpp:44-52). matrix_operand.hpp and utils.hpp are installed headers in `Dispatch`/`Internal` with no flag or scalar parameter of interest.

Totals: 18 positional bools across 10 entry points; every scalar is `double`; no public BLAS signature uses float, ElementScalar or std::complex. No int flags.

CALL SITES (regex `blas()` / `Level<N>::` + name on one line; a call broken across lines before the name is missed, so these are lower bounds; `src` holds only the definition of each method, no internal caller):
tests / examples / benchmarks / docs+README, then sushiai (D:\Projects\sushiai, the only sibling that calls BLAS):
axpy 11/0/1/2 +14 | dot 9/0/1/0 | scal 12/0/0/0 +16 | copy 12/0/0/0 +19 | swap 2/0/0/0 | nrm2 9/0/1/2 | asum 5/0/1/2 | iamax 8/0/1/0 | rot 2/0/0/0 | gemv 10/0/0/0 +2 | ger 7/0/0/0 +4 | symv 7 | trmv 9 | trsv 9 | syr 4 | syr2 4 | gemm 27/1/1/5 +3 | trsm 7 | syrk 8 | execute_gemm 21 in tests (test_gemm.cpp 11, test_channel_bias_epilogue.cpp 9, tests/package/main.cpp 1), 1 in docs/api/mainpage.md, 2 code samples in engine/blas/README.md, +5 in sushiai.
Methods that carry a flag or scalar and so change shape: about 150 call lines in this repo plus about 44 in sushiai (axpy, scal, gemm, gemv, ger, execute_gemm).

EXISTING CORE ENUMS (include/SushiBLAS/core/common.hpp): only `Layout {ROW_MAJOR, COLUMN_MAJOR}` (:38) and `DataType` (:49). No Transpose, Uplo, Diag or Side anywhere in include/, src/ or kernels/. `Dispatch::ElementScalar` (engine/math/elementwise/scalar.hpp:44) holds one double as half/float/double/int32/int64 and its static_assert (:79) excludes complex.

Backlog already names both jobs: docs/design/REMAINING_WORK.md:60-61 (complex scalars, hemv/her/herk, conjugate transpose) and :81-82 (enum classes for the bool flags).

### Findings

#### Public docs and kernels disagree on conjugation in dot, ger, syr, syr2

Size: S, 4 headers/sources plus tests (test_dot complex cases) (breaking)

Evidence: level1.hpp:56 documents dot as 'the unconjugated dot product x^T * y', but src/engine/blas/level1/dot.cpp routes every complex dtype to Kernels::*::dotc (the `if constexpr (Core::is_complex_v<T>)` branch; src/README.md:44 confirms 'computes the conjugated dot product (DOTC)'). level2.hpp:56 documents ger as 'A = alpha * x * y^H + A', :84 syr as 'x * x^H', :90 syr2 as 'x * y^H + y * x^H', while kernels/reference/blas/level2/ger.hpp:41,57, syr.hpp:41,60, syr2.hpp:41,61 and kernels/sycl/blas/level2/ger.hpp compute the unconjugated y^T / x^T forms with no std::conj. Grep for `conj` under kernels/ and src/engine/blas finds it only in the two level1.hpp files.

Proposal: Fix the semantics before freezing v0.1.0 by naming them the CBLAS way: `dot` real-only or unconjugated for all dtypes, with `dotu` and `dotc` as explicit complex entries; `ger` real or unconjugated, `geru` / `gerc` explicit; `syr`/`syr2` stay unconjugated (complex symmetric), `her`/`her2` carry the conjugate. Correct the three @brief lines in level2.hpp and the one in level1.hpp to match. Changing what `dot` does on a complex tensor is a behaviour break and needs the owner's choice (see decisions).

#### 18 positional bool flags across 10 entry points; no flag enums exist

Size: L. 4 public headers + 1 new, 10 src/engine/blas/*.cpp, 9 reference + 9 SYCL kernel headers, about 110 call lines in tests/benchmarks/examples/docs, 10 call lines in sushiai (gemm 3, gemv 2, execute_gemm 5) (breaking)

Evidence: gemv(transA) level2.hpp:52; symv(upper) :64; trmv(upper, transA, unit_diag) :72; trsv(same) :80; syr(upper) :87; syr2(upper) :93; gemm(transA, transB) level3.hpp:51; trsm(left_side, upper, transA, unit_diag) :60; syrk(upper, transA) :70; execute_gemm(transA, transB) dispatch.hpp:207. The same bools run through every kernel signature: kernels/reference/blas/level3/trsm.hpp:45, level2/trmv.hpp:45, symv.hpp:45, syr.hpp:45, syr2.hpp:45, gemv.hpp:45, level3/syrk.hpp:46, include/SushiBLAS/kernels/reference/blas/level3/gemm.hpp:49,98 and their SYCL twins (kernels/sycl/blas/level3/trsm.hpp:49, syrk.hpp:86, level2/*.hpp, include/.../sycl/blas/level3/gemm.hpp:479). A call such as `trsm(A, B, true, false, false, true)` compiles with any two flags swapped. Flag order also differs between siblings: trmv is (upper, trans, unit), trsm is (side, upper, trans, unit), syrk is (upper, trans), gemv puts trans first.

Proposal: Add one header, `include/SushiBLAS/core/blas_types.hpp` (or extend core/common.hpp), holding CBLAS-named enums in `SushiBLAS::Core`, all `enum class : uint8_t` so they fit `TaskMetadata::set_param` (sushiruntime task_types.hpp:155-158 requires trivially copyable and <= 8 bytes): `Transpose { NO_TRANS, TRANS, CONJ_TRANS }`, `Uplo { UPPER, LOWER }`, `Diag { NON_UNIT, UNIT }`, `Side { LEFT, RIGHT }`. Use them in the public methods, in execute_gemm and in every kernel signature, so the bool never reappears one layer down. Fix one argument order for all siblings, the CBLAS one: Side, Uplo, Transpose, Diag, then scalars.

Proposed public set (S = scalar type from the next finding; operand order kept as it is today, flags and scalars trailing):
  axpy(S alpha, const Tensor& x, Tensor& y)
  scal(S alpha, Tensor& x)
  rot(Tensor& x, Tensor& y, double c, double s)   // c, s real in CBLAS csrot/zdrot; kernels already take RealOf<T> (reference level1.hpp:77)
  dot / dotu / dotc(const Tensor& x, const Tensor& y, Tensor& result)
  copy, swap, nrm2, asum, iamax unchanged
  gemv(A, x, y, Transpose trans = NO_TRANS, S alpha = 1, S beta = 0)
  ger / geru / gerc(x, y, A, S alpha = 1)
  symv / hemv(A, x, y, Uplo uplo, S alpha = 1, S beta = 0)
  syr(x, A, Uplo uplo, S alpha = 1);  her(x, A, Uplo uplo, double alpha = 1)   // her's alpha is real in CBLAS
  syr2 / her2(x, y, A, Uplo uplo, S alpha = 1)
  trmv / trsv(A, x, Uplo uplo, Transpose trans = NO_TRANS, Diag diag = NON_UNIT)
  gemm(A, B, C, Transpose transA = NO_TRANS, Transpose transB = NO_TRANS, S alpha = 1, S beta = 0)
  symm / hemm(A, B, C, Side side, Uplo uplo, S alpha = 1, S beta = 0)
  syrk(A, C, Uplo uplo, Transpose trans = NO_TRANS, S alpha = 1, S beta = 0)
  herk(A, C, Uplo uplo, Transpose trans = NO_TRANS, double alpha = 1, double beta = 0)   // both real in CBLAS
  syr2k(A, B, C, Uplo, Transpose, S alpha, S beta);  her2k(..., S alpha, double beta)
  trsm / trmm(A, B, Side side, Uplo uplo, Transpose trans = NO_TRANS, Diag diag = NON_UNIT, S alpha = 1)
  execute_gemm(engine, A, B, C, Transpose transA, Transpose transB, S alpha, S beta, name, op_id, epilogue)

I recommend dropping the default on Uplo and Side: today `upper = false` silently means LOWER and `left_side = true` LEFT, and a triangle is not something a caller should get by omission. This is an owner decision.

#### Scalars are `double` everywhere, so no complex alpha/beta can be passed

Size: M. 1 new header, 4 public headers, about 12 src files, 1 test file for the scalar (breaking)

Evidence: All 14 scalar parameters above are `double`. Each operation narrows with `static_cast<T>(alpha)` inside the dtype-generic closure (axpy.cpp, scal.cpp:65, gemv.cpp:98-99, symv.cpp:95-96, syr.cpp:86, syr2.cpp:93, ger.cpp, syrk.cpp:98-99, trsm.cpp, dispatch.hpp:288-289); for T = std::complex<R> that yields a zero imaginary part. The operations do accept complex tensors: execute_blas_task defaults to Domain::REAL_AND_COMPLEX (utils.hpp:152) and Level3::gemm passes it (src/engine/blas/level3/gemm.cpp:43). The engine/blas/README.md section 'Scalar parameters of Level1' records why a templated signature was rejected (dtype is a runtime value). `Dispatch::ElementScalar` cannot serve: it has no complex member (scalar.hpp:79). `TaskMetadata::set_param` static-asserts sizeof(T) <= 8 (sushiruntime task_types.hpp:158), so a complex<double> does not fit one slot; K_MAX_PARAMS is 8 (:137). gemm already uses 4 slots, trsm 5.

Proposal: One value type, sibling of ElementScalar and shaped the same: `Dispatch::BlasScalar` (name is the owner's call) in `include/SushiBLAS/engine/blas/scalar.hpp`. It stores `std::complex<double>`, has implicit constructors from `double` and from `std::complex<double>` (implicit so `gemm(A, B, C, NO_TRANS, NO_TRANS, 2.0)` and existing `axpy(0.5, x, y)` keep compiling), and `template<typename T> T as() const` returning float/double/half for real T and complex<R> for complex T. `as<real T>()` on a value with a non-zero imaginary part must throw at record time, in line with 'an unsupported dtype throws'; that check belongs in one helper called by each operation with the tensor dtype, not in the closure. Metadata takes two slots per scalar (re, im): gemm then needs 6 of 8 slots, trsm 6, which fits. This replaces the 13 `static_cast<T>(alpha)` sites with `alpha.as<T>()`. Because the constructors are implicit, existing real call sites (axpy 11+14, scal 12+16, ger, syr...) compile unchanged; only the flag change forces edits. Whether ElementScalar and this type should be one class is a data-ownership question for the owner; they differ in domain (int vs complex) so I would keep two.

#### Conjugate transpose: what each existing kernel needs

Size: M. 1 new kernel header, 4 reference + 4 SYCL kernels, 5 src files, tests per routine

Evidence: Every kernel reads op(A) through one accessor, and none conjugates: reference gemm A_at/B_at (include/.../reference/blas/level3/gemm.hpp:60-67); SYCL gemm through `Detail::strides_for(layout, bool transposed, ld)` and `Detail::load_operand` (sycl gemm.hpp:69-90, 6 uses of load_operand, strides chosen at :487-488); gemv `transA ? A_at(k,o) : A_at(o,k)` (reference gemv.hpp:64, sycl gemv.hpp:98); trmv A_eff (reference trmv.hpp:52-55, sycl trmv.hpp:64-66); trsm A_eff in both side branches (reference trsm.hpp:63,91; sycl trsm.hpp:83,109); trsv has no kernel of its own and calls trsm with n x 1 (src/engine/blas/level2/trsv.cpp).

Proposal: Add one shared brick, e.g. `include/SushiBLAS/kernels/blas/conjugate.hpp`, with `template<typename T> T conj_if(bool conjugate, T v)` that is the identity for real T at compile time (`if constexpr (Core::is_complex_v<T>)`), usable on host and device. Then per kernel:
- gemm: split Transpose into (transposed, conjugated) once at entry; strides_for keeps its bool; wrap the load sites. Reference: 2 accessor lambdas. SYCL: load_operand gains a conjugate argument or the 6 call sites wrap it; that header is 537 lines with three variants, so this is the largest single edit. Reuse of existing code: all of it, the tiling does not change.
- gemv: 1 line in each of the two kernels.
- trmv: 1 accessor line each (A_eff). Diagonal read goes through the same accessor, so it is conjugated correctly.
- trsm: 2 accessor lines each (left and right branch); the divide by A_eff(i,i) then uses the conjugated diagonal, which is what CBLAS specifies. trsv inherits it for free.
- syrk: stays transpose-only (CBLAS rejects ConjTrans for syrk; herk is the conjugating sibling). The record-time check should throw on CONJ_TRANS for syrk and on TRANS for herk with complex dtype.
For real dtypes CONJ_TRANS must behave as TRANS, as in CBLAS.

#### Missing complex/Hermitian routines and how much of the symmetric sibling each reuses

Size: L. Cheap group (dotu, dotc, geru, gerc, hemv, her, her2, herk): about 8 public methods, 8 src files, edits to 5 kernel pairs. Expensive group (symm/hemm, syr2k/her2k): 2 new kernel pairs = 4 new kernel headers, 4 src files, new tests

Evidence: Existing kernel pairs: level1 (axpy, copy, scal, swap, rot, asum, nrm2, dot, dotc, iamax), gemv, ger, symv, syr, syr2, trmv, gemm, syrk, trsm (kernels/blas.hpp:33-56). dotc exists in both back ends (reference level1.hpp:181-186, sycl level1.hpp:210-224) but has no public method of its own; the reference dotc sums in plain std::complex<T> while dot uses Accumulator<T> (:171), so the two siblings are not shaped the same. No symm, syr2k, trmm kernel exists, hence no symmetric sibling for hemm or her2k to reuse.

Proposal: Per routine, with reuse stated:
- dotu: zero new kernel code; the existing `dot<T>` kernel already is dotu for complex T. Needs only a public method and removal of the complex branch in dot.cpp.
- dotc: kernels exist. Bring the reference one to Accumulator form so it matches dot; public method is new.
- geru: zero new kernel code; today's ger kernel is geru.
- gerc: ger kernel plus one `conj_if` on y[j]. Best expressed as one kernel `ger(..., bool conj_y)` with ger/geru/gerc as three thin operation classes; reuse about 100%.
- hemv: symv with two changes in `sym_at`: conjugate the mirrored read, and take only the real part of the diagonal (CBLAS ignores the imaginary diagonal of a Hermitian matrix). One kernel `symv(..., bool hermitian)` serves both; the SYCL row-reduction path (symv.hpp:68-88) is untouched. Reuse about 95%.
- her: syr with conj on x[j], alpha real (RealOf<T>), and the diagonal's imaginary part forced to zero after the update. Reuse about 90%; the real alpha means a different scalar type in the signature, so a flag on syr is not enough on its own, it needs a small wrapper.
- her2: syr2 with conj(alpha) on the second term and conj on the column factors, diagonal imaginary part zeroed. Reuse about 85%.
- herk: syrk with the second factor conjugated, alpha and beta real, diagonal imaginary zeroed, trans in {NO_TRANS, CONJ_TRANS}. Reference: one line at syrk.hpp:70 plus the diagonal rule. SYCL: conjugate when loading tileB (syrk.hpp, the second load_operand) plus the diagonal rule at the store; tiling reused whole. Reuse about 90%.
- her2k: NO sibling to reuse; syr2k does not exist. Building syr2k first and deriving her2k keeps the family shaped the same; a correct but slower route is two gemm-shaped products restricted to a triangle. New kernel pair.
- hemm: NO sibling; symm does not exist. symm + hemm as one kernel pair with a hermitian flag, modelled on symv's mirrored accessor applied to a gemm loop. New kernel pair.
- trmm: not requested but is the missing sibling of trsm (trmv has one, trsm's multiply partner does not).
- conjugate transpose in gemm/gemv/trsm/trmv/trsv: covered by the previous finding.
Note that docs/CONTRIBUTING.md section 5 (per AGENTS.md) requires a reference and a SYCL kernel for each, and every real dtype must either work or throw: hemv/her/her2/herk/hemm on a real tensor should either forward to the symmetric routine or throw; that is a choice for the owner.

#### Level 2/3 siblings are not shaped the same below the public surface

Size: M. 7 level2 src files, 3 level3 src files, dispatch.hpp

Evidence: Level 3 operations validate through matrix_operand.hpp (trsm.cpp uses describe_matrix_operand, require_same_layout, require_matrix_extent, batch_stride), while Level 2 operations hand-roll rank/lda/shape checks with SB_THROW_IF (ger.cpp, trsv.cpp: `int64_t lda = (A.layout == ROW_MAJOR) ? A.shape[rA-1] : A.shape[rA-2]` repeated per file). The README records the two exception types that result (engine/blas/README.md:292-294; REMAINING_WORK.md:78). The metadata slot order also varies: gemv stores transA at slot 0 and alpha at 1 (gemv.cpp:83-85), gemm stores alpha at 0 and transA at 2 (dispatch.hpp:273-276), trsm alpha at 0 then flags (trsm.cpp), syr upper at 0 then alpha (syr.cpp:73-74). Each Level1/2/3 class has a public data-less interface but a protected `Engine& engine_` each, so BLASOps holds three copies of the reference plus a fourth in SparseBLAS (blas.hpp:46,53).

Proposal: When the signatures are rewritten anyway, route Level 2 matrices through describe_matrix_operand (it already yields rows, cols, leading_dimension, layout) and fix one metadata slot convention (scalars first as re/im pairs, then Side, Uplo, Transpose, Diag). These are internal and do not change the public interface, but doing them in the same pass avoids touching the 19 src files twice. Whether SparseBLAS, an empty class, ships in the v0.1.0 public interface is an owner decision.

#### Answer to the owner's question: where kernel fusion happens

Size: S, 0 files

Evidence: Not in Level1/2/3. The only fusion on the BLAS side is the GEMM epilogue: `Dispatch::execute_gemm` (engine/blas/dispatch.hpp:205) passes a per-element functor `(value, row, col)` into Kernels::Reference::gemm / Kernels::SYCL::gemm, which apply it just before the store (reference gemm.hpp:85). engine/blas/README.md:21-25 states that the five math/logic families fuse a chain of elementwise operations into one launch through their own dispatch seams (include/SushiBLAS/engine/dispatch.hpp and engine/math/**/dispatch.hpp), and this one fuses the elementwise work into the GEMM. `Level3::gemm` is execute_gemm with GemmIdentityEpilogue (src/engine/blas/level3/gemm.cpp:43). I did not survey the math seams or SushiRuntime's graph for automatic fusion; that is outside this area and I make no claim about it.

Proposal: No change proposed here. Note for the signature work: execute_gemm is the one BLAS entry point with outside callers that fuse (sushiai, 5 sites), so its Transpose/scalar change is the part of this break that reaches another repository.

### Decisions raised

- What should `dot` do on a complex tensor in v0.1.0? Today the header says unconjugated (level1.hpp:56) and the code conjugates (dot.cpp).
  - Options: CBLAS split: `dot` real-only (throws on complex), `dotu` and `dotc` for complex | `dot` unconjugated for every dtype (matches the header), plus `dotc`; `dotu` as an alias or omitted | Keep `dot` = conjugated for complex (matches current behaviour, NumPy vdot style), add `dotu`
  - Recommended: CBLAS split: `dot` real-only, `dotu` and `dotc` explicit. Nothing is then decided by dtype behind the caller's back.

- Same question for `ger`: the header says y^H (level2.hpp:56), the kernels compute y^T.
  - Options: `ger` real-only, `geru` / `gerc` for complex (CBLAS) | `ger` = unconjugated for all dtypes (current behaviour), add `gerc` only
  - Recommended: `ger` real-only with `geru` / `gerc`, the same rule as dot so the siblings match. sushiai's 4 ger calls are unaffected if its tensors are real; I did not check their dtypes.

- Name and home of the complex-capable scalar type, and whether it merges with Dispatch::ElementScalar.
  - Options: New `Dispatch::BlasScalar` in engine/blas/scalar.hpp, separate from ElementScalar (real+complex vs real+int domains) | Extend ElementScalar to carry complex and use it everywhere | Overloads: every method twice, once with double and once with std::complex<double>
  - Recommended: A separate type beside ElementScalar, implicit from double and std::complex<double>. Overloads double the surface; merging forces complex into operations (clamp, pow, elu) that reject it.

- Should Uplo and Side keep default values?
  - Options: No defaults for Uplo and Side; Transpose and Diag default to NO_TRANS / NON_UNIT | Keep today's defaults (LOWER, LEFT)
  - Recommended: No defaults for Uplo and Side. It is a second source break on symv, syr, syr2, trmv, trsv, trsm, syrk, taken once with the enum change.

- Enum spelling and home.
  - Options: `Core::Transpose { NO_TRANS, TRANS, CONJ_TRANS }`, `Core::Uplo { UPPER, LOWER }`, `Core::Diag { NON_UNIT, UNIT }`, `Core::Side { LEFT, RIGHT }` in a new core/blas_types.hpp | Same enums appended to core/common.hpp | Longer names: `Triangle`, `Diagonal`, `Transpose { NONE, TRANSPOSE, CONJUGATE }`
  - Recommended: CBLAS names in a new core/blas_types.hpp: common.hpp stays about layout and dtype, and the kernels can include the enum header alone.

- How far does v0.1.0 go on the missing routines?
  - Options: Signatures + conjugate transpose + the cheap group (dotu, dotc, geru, gerc, hemv, her, her2, herk); symm/hemm, syr2k/her2k, trmm to the backlog | Everything, including new symm/hemm and syr2k/her2k kernel pairs | Signatures and enums only; all Hermitian routines after v0.1.0
  - Recommended: The first: enums that include CONJ_TRANS and a scalar that carries complex are what protect the interface from a second break; hemm and her2k are additions and can land later without breaking anyone. The user's request says 'all of it', so if that is meant literally the second option applies and adds four kernel headers with no sibling to copy.

- Hermitian routine called on a real tensor: forward to the symmetric routine or throw?
  - Options: Forward (hemv on FLOAT32 behaves as symv), as CBLAS users expect from generic code | Throw, as an unsupported dtype
  - Recommended: Forward. With one kernel carrying a hermitian flag it costs nothing, since conj is the identity on real T.

- Does the empty `SparseBLAS` base stay in BLASOps for v0.1.0?
  - Options: Remove it from the public interface until it declares an operation | Keep the placeholder
  - Recommended: Remove it: adding a base class later is not a break, removing one after release is. It is the owner's file, so I only raise it.

## scout/engine-srp (Scout: engine-srp)

Read-only survey of Engine and its direct neighbours. Nothing was written, built or run through `sb`; items 6 and 7 of the brief are vacuous (no source written, scratchpad untouched, 0 bytes). Evidence is from reading files and `rg -c` counts; no test was executed, so behavioural claims below are read from code, not observed.

WHAT ENGINE OWNS (D:\Projects\sushiblas\include\SushiBLAS\engine.hpp:217-222, D:\Projects\sushiblas\src\engine.cpp): `context_` (RuntimeContext&), `graph_` (TaskGraph by value), `default_layout_`, `mode_`, `rng_`, plus eight facade factories (blas, io, logic, random, nonlinear, reductions, spatial, elementwise), four `create_tensor` overloads and an inline `execute()`. That is five responsibilities in one class: composition, recording target, execution, allocation, and policy state.

WHO USES EACH MEMBER (code hits; README/doc mentions excluded where noted)
- get_graph(): include/SushiBLAS/engine 2 (task.hpp:257, :286); engine/logic/dispatch.hpp 3; engine/math/*/dispatch.hpp 12 (reductions 6, elementwise 3, nonlinear 2, random 1); src/engine/logic 2 (all.cpp:102, any.cpp:102); src/engine/math/elementwise 1 (convert.cpp:64); src/engine/math/random 9 (constant.cpp 7, orthogonal.cpp:80, shuffle.cpp:96); tests/functional 20 hits in 9 files, all `take_nodes()`. Total 29 direct `add_task` sites outside any shared recorder (task.hpp's own 2 included).
- get_context(): library code 2 only: task.hpp:283 (host-reachability check) and scratch.hpp:62 (scratch allocation). engine.cpp:103 uses `context_` directly. The 74 tests/functional hits and 9 tests/common hits are `TestEnvironment::get_context()`, not Engine's.
- get_mode(): library code 1 (task.hpp:282); benchmarks 1 (bench_common.hpp:114); tests 1 (test_execution_mode.cpp:65).
- set_mode(): tests/functional 56 hits in 20 files; no library caller.
- get_layout(): no library caller; tests/functional 1.
- rng(): 2, both in engine/math/random/dispatch.hpp:71 and :77.
- create_tensor(): tests/functional 766 hits in 121 files; benchmarks 13 in 5; tests/package 10; examples 6 in 2; tests/common 1. No library caller: operations never allocate tensors.
- execute(): src/io.cpp 9 (`engine_.execute().wait()`); benchmarks 2; tests/package 4; tests/functional about 403 pattern hits in 120 files.
- Engine& as a parameter or member: 71 hits in 33 files under include/ (every facade, every Dispatch seam, ScratchBuffer, allocate_level1_scratch).

DOWNSTREAM: D:\Projects\sushiai consumes this surface. Engine& appears about 73 times in its src/ and include/; `get_context()` 13 times there (mostly `.wait_all()`, also allocation.cpp:93/:159 and trainer.cpp:289 `get_execution_engine()`); `get_graph().compile()` at src/graph/compiled_step.cpp:50 with its own execute at :69; `get_mode()` at src/graph/fusion_selector.cpp:39; `get_layout()` twice; `Dispatch::execute_*`/`execute_typed_task`/`execute_gemm` 10 times. Any change to these names breaks sushiai in the same release.

KERNEL FUSION (the owner's side question): SushiBLAS does not decide fusion. It supplies the seam: the public `SushiBLAS::Dispatch` executors take a caller-written functor and emit one kernel (include/SushiBLAS/engine/README.md:27-33; docs/api/mainpage.md:23), and GEMM takes an epilogue (include/SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp, channel_bias_epilogue.hpp). The choice of what to fuse lives in SushiAI (sushiai/src/graph/fusion_selector.cpp, compiled_step.cpp). SushiRuntime's chain plan (include/SushiRuntime/execution/chain_plan.hpp) orders submissions; I did not find kernel merging there, and did not read it fully.

PROPOSED BRICKS, Engine as a thin composition root: (1) ExecutionMode + KernelPairDispatcher + require_host_reachable in their own header below Engine; (2) a TaskRecorder that is the only thing an operation or Dispatch seam sees; (3) a TensorFactory/allocator owning context + default layout + device; (4) RngStream passed directly; (5) execute() as a small GraphRunner. Findings below give evidence, shape and breakage for each.

### Findings

#### Operations and Dispatch seams take the whole Engine when they need a recorder

Size: L, about 45 files in sushiblas (20 headers, 14 sources with direct add_task, facades, tests using take_nodes) plus sushiai (breaking)

Evidence: Every public seam is `execute_*(Engine& engine, ...)`: include/SushiBLAS/engine/task.hpp:250, :274; engine/math/elementwise/dispatch.hpp:65, :118, :178; engine/math/nonlinear/dispatch.hpp:62, :114; engine/math/reductions/dispatch.hpp:101, :179, :230, :293, :398, :453; engine/logic/dispatch.hpp:57, :112, :171; engine/math/random/dispatch.hpp:223; engine/blas/dispatch.hpp:206; engine/blas/utils.hpp:129, :153. What they actually call: `get_graph().add_task` (29 sites), `get_mode()` once (task.hpp:282), `get_context()` twice (task.hpp:283, scratch.hpp:62), `rng()` twice (random/dispatch.hpp:71, :77). task.hpp:40 includes engine.hpp and engine.hpp:33-44 includes every family, so each seam header pulls in all eight families (include/SushiBLAS/engine/README.md:382-384 states the include is deliberate). tools/layering/check_layering.py declares `K_ENGINE_ALLOWED_MEMBERS = frozenset({"get_graph"})` for `src/engine`, so the rule 'an operation knows the engine only through get_graph()' already exists but is enforced on sources, while the headers reach get_mode, get_context and rng. Backlog items C7 and C8 (docs/design/REMAINING_WORK.md:87-89) name the same gap.

Proposal: Introduce `SushiBLAS::TaskRecorder` in its own header (no include of engine.hpp): holds `TaskGraph&`, the execution policy and what host-reachability needs, and exposes one `record(meta, reads, writes, body)` plus the typed variant that is today `execute_typed_task`. Engine owns one and returns it from a single accessor; facades (`Level1`, `ElementwiseOps`, ...) and every `Dispatch::execute_*` take `TaskRecorder&`. The 29 direct `add_task` sites collapse onto it, which is C7. Breaks: the first parameter type of every public Dispatch seam and of every facade constructor; sushiai's 10 seam calls and its `engine.get_graph()` uses (compiled_step.cpp:50, two tests). Callers of `engine.blas().axpy(...)` do not change. Decide whether `get_graph()` stays public as an escape hatch for SushiAI's own compile step.

#### Execution mode is Engine state read at record time by one overload, and one enumerator exists only to be refused

Size: M, about 8 files (engine.hpp, engine.cpp, task.hpp, new header, README, ARCHITECTURE, 2 tests) (breaking)

Evidence: `mode_` lives in Engine (engine.hpp:221). Only `execute_typed_task`'s second overload reads it (task.hpp:282); the first overload hard-codes `KernelPairDispatcher(ExecutionMode::AUTO)` (task.hpp:260). docs/reference/KNOWN_ISSUES.md:11 records that elementwise, nonlinear, logic, reduction and random never read the mode. `ExecutionMode::VERIFY` is declared (engine.hpp:58) and `set_mode` throws `std::logic_error` on it (src/engine.cpp:70-76); `dispatch_with_mode` carries a third branch that throws for it (task.hpp:73-76). `ExecutionMode`, `dispatch_with_mode`, `KernelPairDispatcher` and `require_host_reachable` are in namespace `Internal` inside a public installed header (task.hpp:48-155) and SushiAI reads `engine.get_mode()` (sushiai/src/graph/fusion_selector.cpp:39).

Proposal: Move `ExecutionMode` into `engine/execution_mode.hpp` beside a small `ExecutionPolicy` value object (holds the mode, validates on set, builds the `KernelPairDispatcher`, runs the host-reachability check). The recorder asks the policy; no seam calls `engine.get_mode()`. Remove `VERIFY` before 0.1.0: an enumerator whose only behaviour is a throw is surface that must be broken later to implement or delete. Whether the five families honour the mode is the owner decision already on the backlog (C2); the policy object is the place either answer lands. Breaks: `ExecutionMode::VERIFY` users (tests that assert the throw), the include path for `ExecutionMode` if callers relied on engine.hpp transitively (engine.hpp can keep including it, so source-compatible), and `Engine::get_mode/set_mode` only if they move to `engine.policy()`.

#### Tensor creation is welded to Engine, cannot pick a device, and refuses an empty shape

Size: M, about 7 files with forwarders kept; L (130+ files, mostly tests) without

Evidence: `create_tensor` has four overloads (engine.hpp:179-209); the two initializer_list ones copy into a `std::vector` first (src/engine.cpp:81, :88). `default_layout_` is used only there (engine.cpp:105-106) and `get_layout()` has no library caller. `Storage` accepts a `device_index` (storage.hpp:63-66) and `ScratchBuffer` passes one (scratch.hpp:62-64), but `create_tensor` never does (engine.cpp:103), so every tensor Engine creates lands on device queue 0. `Storage` throws on `n_bytes == 0` (storage.hpp:69), so by reading the code `create_tensor({0, 3})` throws 'Requested size cannot be zero'; tests build empty tensors by hand with a null storage instead (tests/functional/unit/test_empty_tensors.cpp:44). `ScratchBuffer` needs an `Engine&` only to reach the context (scratch.hpp:56-65). No operation in src/engine calls create_tensor; its callers are tests (766 hits/121 files), benchmarks (13), examples (6), tests/package (10), and sushiai builds `Storage` itself (src/core/allocation.cpp:93, :159).

Proposal: Extract `TensorFactory` (or `TensorAllocator`): owns `RuntimeContext&`, default layout and default device index; one primary `create(span<const int64_t> dims, DataType, Residency, device_index)` with the byte-count check that is now an anonymous function in engine.cpp:36-57, and thin overloads forwarding to it without a vector copy. It returns a storage-less tensor for a zero-element shape, which settles the empty-tensor question in one place. `ScratchBuffer` takes the context or the factory, not Engine. Engine keeps `create_tensor` as a one-line forward so the 800 call sites survive, or exposes `engine.tensors()` only. Breaks: nothing if the forwarders stay; `ScratchBuffer`'s constructor and `allocate_level1_scratch` change their first parameter (both internal to the library, 6 call sites in reductions/dispatch.hpp and blas/utils.hpp). If the forwarders go, every test, benchmark and example changes.

#### Engine::execute() and every operation return a sycl::event that is always empty

Size: L, signature change on roughly 110 facade methods and their 112 sources; mechanical (breaking)

Evidence: `execute()` returns `sycl::event()` on both paths and its own comment says the call returns only after the graph finished (engine.hpp:159-170). src/io.cpp calls `engine_.execute().wait()` nine times (:525-:696), a wait on a default event. `execute_typed_task` returns `sycl::event()` at record time (task.hpp:261, :289) and so does every facade method (for example include/SushiBLAS/engine/blas/level1.hpp, `sycl::event axpy(...)`). The estate audit already lists SYCL types in the public surface as an owner decision (docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md:1011-1015). SushiAI bypasses `execute()` and drives `get_graph().compile()` plus `get_execution_engine().execute(plan)` itself (sushiai/src/graph/compiled_step.cpp:50, :69).

Proposal: Give execution its own brick, `GraphRunner` (compile, empty-graph early return, execute), held by Engine; `Engine::execute()` forwards to it and returns `void`. Record calls return `void` too, or a small `RecordedTask` handle if a per-task identity is wanted later; picking that now avoids a second break. SushiAI's compiled step can then use the same runner instead of reaching through `get_context()`. Breaks: every caller that stores or waits on the returned event (io.cpp's nine `.wait()`, any `auto ev = engine.blas()...` in tests and sushiai); callers that ignore the return value compile unchanged.

#### Facades are shaped unevenly and expose the Engine reference

Size: M, about 12 headers (breaking)

Evidence: `Level1`, `Level2`, `Level3` declare `Engine& engine_` as a protected data member listed before the public section (level1.hpp:42-45, level2.hpp:44-45, level3.hpp:44-45); `ElementwiseOps`, `NonLinearOps`, `ReductionOps`, `RandomOps`, `LogicOps`, `SpatialOps`, `IO` keep it private at the end (elementwise.hpp:330, nonlinear.hpp:192, reductions.hpp:198, random.hpp:194, logic.hpp:159, spatial.hpp:178, io.hpp:258). `BLASOps` inherits from four bases, each with its own reference, one of them `[[maybe_unused]]` (blas.hpp:47-53, sparse.hpp:51). Engine is edited for every new family (engine.hpp:103-145), which the audit records as C8.

Proposal: One facade shape: a private `TaskRecorder&` (and `RngStream&` for RandomOps, the runner for IO), public methods only. `BLASOps` composes the three levels through accessors (`blas().level1()`) or stays a flat class with one reference; either removes the four-way inheritance. Whether `engine.hpp` keeps including the families is already an open owner decision (REMAINING_WORK.md:74-76); with the recorder header in place the dependency becomes one-way (families include recorder, Engine includes families) and the include cycle through task.hpp disappears even if the facade stays. Breaks: nothing for `engine.blas().axpy()` callers if BLASOps stays flat; code that subclassed LevelN to reach `engine_` breaks (none found in sushiblas; not checked in sushiai).

#### RngStream is already a brick but is reached through Engine

Size: S, 3 to 5 files (breaking)

Evidence: `RngStream` is self-contained and includes only <cstdint> and <mutex> (engine/rng_stream.hpp:28-92). Its two users call `engine.rng()` (engine/math/random/dispatch.hpp:71, :77). Engine exposes it as a mutable public accessor (engine.hpp:215).

Proposal: Hand `RngStream&` to `RandomOps` and to `Dispatch::reseed`/`reserve_offsets`/`execute_random` directly; keep `Engine::rng()` only if a consumer needs to reseed outside `random().set_seed`. Breaks: the signatures of the three random Dispatch functions; `engine.random().set_seed()` callers unchanged.

#### Recording thread-safety is stated for the RNG only

Size: S, documentation plus possibly rng_stream.hpp

Evidence: `RngStream` locks every call and its README says tasks may be recorded from different threads (engine/README.md:209-211). `mode_` is a plain member with unsynchronised get and set (engine.hpp:91, :221). SushiRuntime documents `TaskGraph::compile()` as 'not thread-safe against add_node() or add_task()' (sushiruntime/include/SushiRuntime/graph/task_graph.hpp:227-228); I did not read `add_task`'s implementation, so whether two concurrent `add_task` calls are safe is unverified.

Proposal: State one contract on the recorder ('one recording thread per Engine' or 'record calls are serialised') and make the bricks agree with it. If the answer is single-threaded recording, the mutex in RngStream is dead weight taken once per random op; if multi-threaded, the mode and the graph need the same guarantee. Resolve by reading src/graph/task_graph.cpp in sushiruntime.

#### Small memory and lookup costs next to Engine

Size: S, 4 files plus field users (breaking)

Evidence: `Tensor::get_device()` looks the pointer up with `context_index_for_ptr` (tensor.hpp:227-230) although `Storage` stores `device_index_` (storage.hpp:109, :156); src/io.cpp:67 already uses the stored index. `Storage` exposes `size_bytes` and `requested_bytes` as public mutable fields (storage.hpp:50-53) and `Tensor`'s constructor reads `storage->size_bytes` for its bounds check (tensor.hpp:206). `ReferenceOperands` builds a `std::vector<const void*>` per recorded BLAS or spatial task (task.hpp:87, :94), and `require_host_reachable` copies a `sycl::context` per pointer per device (task.hpp:118-121) but only under FORCE_REFERENCE. `allocate_level1_scratch` calls `x.get_device()` per record (blas/utils.hpp:137).

Proposal: Have `get_device()` use `storage->get_device_index()`; make the two Storage size fields private with const accessors (a field a caller can overwrite defeats the Tensor bounds check); give `ReferenceOperands` a fixed-capacity array, or skip building it unless the mode is FORCE_REFERENCE. Breaks: direct reads of `storage->size_bytes` (tensor.hpp and any test or sushiai code touching the field).

### Decisions raised

- What do operations and the public Dispatch seams take as their first parameter?
  - Options: Keep `Engine&` (no break, C7 and C8 stay open, every seam header keeps including all families) | A narrow `TaskRecorder&` owned by Engine (breaks every Dispatch seam signature and sushiai's 10 seam calls once) | An abstract interface `IRecordContext` Engine implements (same break, adds a virtual call per record)
  - Recommended: A concrete `TaskRecorder&`; recording happens once per operation, yet nothing needs runtime polymorphism here, so the virtual layer buys nothing.

- Does `Engine::get_graph()` stay public?
  - Options: Yes, as the documented escape hatch SushiAI uses for its own compile step | No; SushiAI goes through a `GraphRunner`/recorder API and tests get a `pending_task_count()` instead of `take_nodes()`
  - Recommended: Keep it for 0.1.0 but route every library call through the recorder, so removing it later touches only SushiAI and 9 test files.

- Is `ExecutionMode::VERIFY` removed before 0.1.0?
  - Options: Remove the enumerator | Keep it and keep throwing | Implement it
  - Recommended: Remove it; adding an enumerator later is not a break, deleting one is.

- Do record calls and `Engine::execute()` keep returning `sycl::event`?
  - Options: Keep (always-empty event stays in about 110 signatures) | Return `void` | Return a small library-owned handle
  - Recommended: `void` for `execute()`; for record calls choose between `void` and a handle now, because this is the widest signature change in the survey and should happen once.

- Where does tensor creation live for callers?
  - Options: `engine.create_tensor(...)` stays and forwards to a `TensorFactory` (no caller break) | Only `engine.tensors().create(...)` (about 800 call sites change, mostly tests) | Both for 0.1.0, old one deprecated
  - Recommended: Forwarders stay; the factory is extracted underneath and gains the device index and the empty-shape rule.

- What does `create_tensor` return for a shape with a zero dimension?
  - Options: A tensor with null storage and zero elements | Throw, as the code does today through Storage | A tensor with a minimal non-empty allocation
  - Recommended: Null storage, zero elements; it matches what the elementwise, nonlinear and logic executors already treat as a no-op, and it has to be settled together with the reduction-executor item on the backlog.

- Is recording on one Engine from several threads supported?
  - Options: No, one recording thread per Engine (drop the RngStream mutex) | Yes (graph and mode need the same guarantee as the RNG)
  - Recommended: Needs your call after reading SushiRuntime's add_task; I could not verify its thread-safety from the header.

- Does the SushiAI update ship in the same wave?
  - Options: Yes, sushiai is adapted alongside each breaking brick | No, sushiblas breaks first and sushiai pins the old commit
  - Recommended: Same wave; sushiai touches get_context, get_graph, get_mode, get_layout and the Dispatch seams, so every proposal above reaches it.

## scout/facade (Scout: facade)

Read-only survey; nothing was written, built or run, so items 6 (syntax check) and 7 (scratchpad) are vacuous: no file was created and the scratchpad is untouched. The humanizer skill was loaded and applied to this report. Counts come from grep over `(\.|->)\s*<accessor>\(\)`, excluding build/ and third_party/; they count textual call sites, not unique expressions.

What Engine exposes: D:\Projects\sushiblas\include\SushiBLAS\engine.hpp:30-44 includes io.hpp plus seven family headers, and lines 103-145 define eight inline accessors that each return a view by value (`BLASOps(*this)` and so on). Engine uses none of those types anywhere else; the accessors are the only reason for the includes.

How the views are built: every view has a public `explicit X(Engine& e)` constructor and one `Engine& engine_` member, and its header only forward-declares `class Engine;` (blas.hpp:37,53; spatial.hpp:37,50,178; logic/logic.hpp:35,46,159; math/elementwise.hpp:35,50,330; math/nonlinear.hpp:35,51,192; math/random.hpp:35,54,194; math/reductions.hpp:35,50,198; io.hpp:42,118,258). BLASOps inherits Level1, Level2, Level3 and SparseBLAS, each holding its own `Engine&` (blas/level1.hpp:44-45 and siblings). So no view header needs engine.hpp, and a caller can already write `SushiBLAS::BLASOps(engine)` today. The experimental views (experimental/lapack/linalg.hpp:51, experimental/signal/transforms.hpp:48) are reached only that way; no accessor exists for them.

What the views need from Engine at run time (calls on `engine`/`engine_` in src/ and include/): get_graph 29, execute 9 (all in src/io.cpp:525-696), get_context 2 (engine/scratch.hpp:62, engine/task.hpp:283), get_mode 1 (engine/task.hpp:282), rng 2 (engine/math/random/dispatch.hpp:71,77).

Accessor call sites, sushiblas (calls/files):
- tests: blas 162/38, io 59/5, logic 38/17, random 87/13, nonlinear 29/14, reductions 86/15, spatial 80/9, elementwise 143/45. Total 684 calls in 122 files.
- examples: blas 1/1, io 10/2. Total 11 in 2 files.
- benchmarks: blas 6/2, io 2/1, nonlinear 1/1, reductions 3/1, elementwise 2/1. Total 14 in 5 files.
- docs: blas 8/4, io 1/1. README.md: blas 3, logic 1, random 2, nonlinear 1, reductions 1, elementwise 1. Two more mentions in include/SushiBLAS/engine/README.md:206 and engine/math/README.md:25.
- src, kernels, cli: 0. No library code calls an accessor.

Sibling consumers under D:\Projects: only sushiai includes SushiBLAS headers (36 code files; 18 include engine.hpp, 6 include SushiBLAS.h). sushiengine, sushidsp, sushicore, sushistack, sushifx, sushicars, sushitrack and sushiruntime have zero includes and zero accessor calls. sushiai code: src blas 54, random 13, nonlinear 14, reductions 8, spatial 18, elementwise 22 (129 calls in 7 files: data/dataset.cpp, graph/lowering.cpp, nn/init.cpp, optim/adamw.cpp, optim/loss_scaler.cpp, optim/sgd.cpp, train/trainer.cpp); tests blas 4, elementwise 1 (4 files); sushiai docs 16 mentions. sushiai never calls io() or logic().

Include picture inside sushiblas: SushiBLAS.h is included 135 times (tests/examples/benchmarks reach the families through it and engine.hpp); engine.hpp directly 57 times (47 in src, 9 in include, 1 in tests). The src operation files already include their own family header, so they do not rely on engine.hpp for it.

The user's side question about kernel fusion is outside this area and was not surveyed; the only related fact seen is that sushiai has src/graph/fusion.cpp and include/SushiAI/graph/fusion_selector.hpp, and sushiblas has kernels/blas/level3 epilogue headers.

### Findings

#### Engine's eight accessors are its only dependency on the operation families

Size: S for engine.hpp itself (1 file) plus 1 tool file and 5 doc files; the caller migration is sized per shape below (breaking)

Evidence: engine.hpp:30,33-34,39-41,43-44 include io.hpp and seven family headers; engine.hpp:103-145 are eight one-line inline accessors returning `XOps(*this)`. No other member of Engine names a view type. tools/layering/check_layering.py:66-76 lists the seven accepted includes; docs/reference/KNOWN_ISSUES.md:18 and docs/design/REMAINING_WORK.md:73-75 record it as an owner decision.

Proposal: Delete the eight accessors and the eight includes from engine.hpp; pick one of the three shapes in the following findings for how callers reach a view. After any of them, empty K_KNOWN_VIOLATIONS (the checker reports stale entries otherwise, check_layering.py:267-268) and update KNOWN_ISSUES.md, REMAINING_WORK.md, GLOSSARY.md:7, ARCHITECTURE.md:64,261 and README.md:162.

#### Shape A: construct the view directly, `BLASOps(engine).gemm(...)`

Size: L by count, mechanical by nature: 709 call sites in 129 sushiblas files (tests 684/122, examples 11/2, benchmarks 14/5), about 20 doc mentions, plus sushiai 134 call sites in 11 code files and 16 doc mentions. Library src/: 0 files. (breaking)

Evidence: Every view already has a public `explicit X(Engine&)` constructor and its header forward-declares Engine only (blas.hpp:37,53; spatial.hpp:50; logic/logic.hpp:46; math/elementwise.hpp:50; math/nonlinear.hpp:51; math/random.hpp:54; math/reductions.hpp:50; io.hpp:118). The experimental views are already used this way and have no accessor (experimental/lapack/linalg.hpp:51, experimental/signal/transforms.hpp:48; CHANGELOG.md:129).

Proposal: Add no new API. Engine loses the accessors; SushiBLAS.h (include/SushiBLAS/SushiBLAS.h, currently core/common, core/logger, storage, engine, tensor, version) gains the eight family includes so umbrella users keep one include. Callers rewrite `e.blas()` to `SushiBLAS::BLASOps(e)`. A new family is one new header and one umbrella line; Engine is never edited again. Files that include engine.hpp directly and use a family (18 in sushiai) add the family header or switch to the umbrella.

#### Shape B: one free accessor per family header, `blas(engine).gemm(...)`

Size: Same call-site count as A (709 + 134), plus 8 family headers gaining one function each; SushiBLAS.h gains 8 includes. (breaking)

Evidence: Same construction facts as shape A. Engine lives in namespace SushiBLAS (engine.hpp:47,69), so an unqualified `blas(engine)` resolves through argument-dependent lookup without a using-declaration.

Proposal: Each family header declares `inline BLASOps blas(Engine& e) { return BLASOps(e); }` next to its view (io, logic, random, nonlinear, reductions, spatial, elementwise the same). Call sites change from `e.blas()` to `blas(e)`, a rewrite a single regex can do for `.` and `->` forms. Costs eight short public names in namespace SushiBLAS (`io`, `random`, `logic` are collision-prone with user code under `using namespace`), and it leaves two spellings for the same thing unless the constructors are made private with the function as friend.

#### Shape C: split the type, keep the facade spelling

Size: M-L inside the library: about 12 view headers, 9 dispatch/scratch/task headers, 47 src files that include engine.hpp (include line and `Engine&` type), engine.cpp, check_layering.py and its test. Consumers: 0 call sites.

Evidence: Views and executors take `Engine&` in 12 view headers and in the dispatch signatures (engine/task.hpp:250,274; blas/dispatch.hpp:206; blas/utils.hpp:153; logic/dispatch.hpp:57,112,171; math/elementwise/dispatch.hpp:65,118,178; math/nonlinear/dispatch.hpp:62,114; math/random/dispatch.hpp:223; math/reductions/dispatch.hpp:101-453; engine/scratch.hpp:56). 47 src files and 9 include files include engine.hpp.

Proposal: Move today's Engine minus accessors to a core class (for example `ExecutionCore` in engine/core.hpp) that views and executors depend on; keep a thin `Engine` in a facade header that derives from or wraps the core and carries the eight accessors. Consumer call sites do not change. The layering rule then has to be re-aimed at the core header. Weak point: each new family still edits the facade class, so the facade stays the one file every family touches, and the rule passes because the includes moved rather than because the coupling went away.

#### The layering rule does not see io.hpp, the eighth facade include

Size: S: 1 tool file, 1 test file; a header move would touch 2 includes of io.hpp plus SushiBLAS.h (breaking)

Evidence: K_OPERATION_FAMILY_INCLUDE (check_layering.py:41-43) matches only `SushiBLAS/engine/(blas|spatial|logic|math)`. engine.hpp:30 includes `SushiBLAS/io.hpp` for the `io()` accessor at engine.hpp:109, and IO is a view of the same shape (io.hpp:118,258). There are 8 accessors and 7 known violations.

Proposal: Treat IO as the eighth family in whichever shape is chosen, and either move io.hpp under engine/ or extend the pattern so the rule covers it. IO also calls `engine_.execute()` nine times (src/io.cpp:525-696), a member outside K_ENGINE_ALLOWED_MEMBERS; the rule misses it because it scans src/engine only (check_layering.py:33,239).

#### Engine owns the random family's state

Size: M if moved: engine.hpp, engine.cpp, rng_stream.hpp, random/dispatch.hpp, 13 random test files to recheck (breaking)

Evidence: engine.hpp:37 includes engine/rng_stream.hpp; engine.hpp:215 exposes `rng()` and line 222 holds `RngStream rng_`. Its only callers are engine/math/random/dispatch.hpp:71 and :77.

Proposal: Removing the accessors leaves this one family-specific member in Engine. Decide whether the stream stays on Engine as shared engine state or moves behind a family-neutral slot; this is a data-ownership question, so it is listed for the owner rather than proposed.

#### The stated rule 'an operation knows Engine only through get_graph()' holds in src/engine but not in the executor headers

Size: S for the rule and docs (check_layering.py, its test, AGENTS.md); M if executors are narrowed

Evidence: Executors under include/ call engine.get_mode() and engine.get_context() (engine/task.hpp:282-283), engine.get_context() (engine/scratch.hpp:62) and engine.rng() (math/random/dispatch.hpp:71,77). find_operation_engine_calls scans only K_OPERATION_FOLDER = src/engine (check_layering.py:33,237-244), so these are not checked.

Proposal: If the public interface is being frozen for v0.1.0, write down the member set executors may use (get_graph, get_mode, get_context, rng) as the real contract and extend the rule to include/SushiBLAS/engine, or narrow the executors to match AGENTS.md:16-17. This sets what shape C's core class would have to expose.

#### sushiai includes a SushiBLAS header path that is not in the current include tree

Size: S: 1 file in sushiai

Evidence: D:\Projects\sushiai\include\SushiAI\graph\fusion_selector.hpp:33 includes `<SushiBLAS/engine/device_policy.hpp>`; the sushiblas tree has only include/SushiBLAS/support/device_policy.hpp. Not verified whether sushiai pins an older sushiblas revision; its cli/sushistack.deps.toml would settle that.

Proposal: Fix this include in the same sushiai migration commit that follows the accessor change, since sushiai's 11 code files have to be touched anyway.

### Decisions raised

- Which shape replaces the Engine accessors?
  - Options: A: direct view construction, `SushiBLAS::BLASOps(engine).gemm(...)`; no new API, 709 sushiblas + 134 sushiai call sites rewritten | B: free accessor per family header, `blas(engine).gemm(...)`; same call-site count, adds 8 short names to namespace SushiBLAS | C: split Engine into a core class and a facade that keeps the accessors; 0 consumer call sites, about 70 library files touched, facade still edited per new family
  - Recommended: A. It uses constructors that are already public, matches how the experimental views are reached today, adds no names, and makes a new family a new header with no edit to Engine. B is the fallback if the shorter spelling matters more than the extra names. C keeps the coupling in a renamed place.

- Does SushiBLAS.h become the one include that pulls in every family, or does each consumer include the family headers it uses?
  - Options: Umbrella includes all eight families; engine.hpp alone gives Engine, Tensor and nothing else | No family includes in the umbrella; every consumer names its families | Umbrella includes all; additionally rename SushiBLAS.h to SushiBLAS.hpp for consistency with the rest of the tree
  - Recommended: Umbrella includes all eight families. 135 in-repo includes already go through SushiBLAS.h, so tests, examples and benchmarks need no include edits; the 18 sushiai files on engine.hpp choose per file. Renaming the umbrella is a separate break and should be decided on its own.

- Is IO the eighth operation family for layering purposes?
  - Options: Yes: lose `engine.io()` with the others and bring io.hpp under the rule | No: IO stays an Engine accessor as a documented exception
  - Recommended: Yes. IO has the same view shape, and REMAINING_WORK.md already plans an IO split (C4) whose boundary needs your approval, so both IO decisions are best taken together.

- Where does the RngStream live once Engine names no family?
  - Options: Stays on Engine as shared engine state (`rng()` remains public) | Moves out of Engine into random-family-owned state keyed by engine | Stays on Engine but `rng()` becomes reachable only by the random executors
  - Recommended: No recommendation without your call; this is data ownership. Keeping it on Engine is the smallest change and keeps one stream per engine, which the tests in 13 random test files assume.

- Should the sushiai migration (134 call sites in 11 code files, plus the stale device_policy include) run in the same wave as the sushiblas break?
  - Options: Same wave, so the stack never sits broken | sushiblas first, sushiai as a follow-up task after v0.1.0 is tagged
  - Recommended: Same wave. sushiai is the only sibling consumer found, and it cannot compile against the new headers until its call sites move.

## scout/gpu-perf (Scout: gpu-perf)

Read-only survey; nothing was written, built or run. Items 6 and 7 are vacuous: no source was written, so there is nothing to syntax-check, and the scratchpad was not touched (0 bytes made, nothing to delete). The humanizer skill was loaded before writing this report. No benchmark number is quoted because the repository holds none: docs/design/REMAINING_WORK.md:17 mentions "the numbers of 2026-10-07" from a Ryzen 5 7600X, but no file under docs/ or benchmarks/ records them, so "far below a tuned BLAS" is explained from the code only.

WHY CPU GEMM IS SLOW (all in D:\Projects\sushiblas\include\SushiBLAS\kernels\sycl\blas\level3\gemm.hpp unless noted)
1. No packing and no k-blocking on the CPU path. gemm_no_local runs kk over the whole k (line 426) and reads A and B from global memory through strides at every step (433-437). Only the GPU variant has tile_k = 16 (152). A CPU is forced onto no_local because device_policy.hpp:123 sets has_dedicated_local_memory false for every CPU.
2. Strided A reads. With the benchmark's COLUMN_MAJOR layout (benchmarks/bench_gemm.cpp:73) strides_for gives {row 1, col ld} (75), so A(row, kk) moves ld elements per k-step: 8 KiB per step at n = 2048 float, one new page every step, on 4 streams per work-item (item rows are 8 apart, line 433).
3. Small register block. K_NO_LOCAL_ACC_BYTES = 64 (101) gives 4x4 for float and 2x2 for double (181). The budget is a constant justified for AVX2's 16 registers (include/SushiBLAS/kernels/README.md:138-143); Device::Profile has no native vector width field (device_policy.hpp:67-101), so the block cannot follow a 32-register AVX-512 CPU.
4. Vector width is left to the OpenCL CPU JIT. Device::vector_width returns 1 on a CPU (device_policy.hpp:267). The lane axis is fixed by the nd_range (394-397) while the unit-stride axis flips with layout and transpose: in column-major, B(kk, col) and C(row, col) are ld apart between neighbouring columns (lines 436, 454), so if the JIT vectorises along columns those are gathers and scatters. README.md:249-254 claims contiguous accesses without stating the layout. Which axis the JIT picks was not verified.
5. A bounds branch on every operand load: load_operand (87) is called 8 times per k-step in the innermost loop.
6. Work-group shape: K_CPU_WORK_GROUP_TARGET = 64 (device_policy.hpp:58) and group_shape (205-215) give 8x8 items, so a group covers 32x32 of C for float and reads 2*32*k elements with no cache-sized inner block.
7. No FMA contraction: sushiruntime_deterministic_fp adds -ffp-contract=off and -fno-fast-math (D:\Projects\sushiruntime\cmake\Determinism.cmake:53-65) and SushiBLAS inherits it PUBLIC (src/CMakeLists.txt:194). Whether the SPIR-V JIT honours it was not verified.
8. The benchmark times graph recording and Engine::execute as well (benchmarks/README.md:47-49) and GEMM logs at INFO on every call (522), which weighs on n = 128.

SERIAL EXECUTION
Whole op in one single_task: TRMV (kernels/sycl/blas/level2/trmv.hpp:58), argmax (src/engine/math/reductions/argmax.cpp:49), argmin (argmin.cpp:50), var (var.cpp:50), std (std.cpp:50), logsumexp (logsumexp.cpp:51), shuffle (src/engine/math/random/shuffle.cpp:55). var, std and logsumexp ignore the scratch they are handed ([[maybe_unused]], var.cpp:41).
Conditionally serial: every Reduce::reduce_indexed user (sum, mean, prod, asum, nrm2, dot, dotc, iamax, max, min, count_nonfinite) takes the single_task branch at include/SushiBLAS/support/reduce_1d.hpp:644 when the device is a CPU and n < 16384 (line 54, 121), or when scratch capacity is under 2 (143-145). Stage two at reduce_1d.hpp:695 is a single_task over at most 256 partials, by design.
Serial inside one work-item: TRSM runs one substitution per right-hand-side column (kernels/sycl/blas/level3/trsm.hpp:62-128); TRSV calls it with one column (src/engine/blas/level2/trsv.cpp:96), so TRSV is one work-item doing O(n^2).
Trivial single_task fills: all.cpp:55, any.cpp:55, max.cpp:119, min.cpp:119.
Host loops: FORCE_REFERENCE runs the reference kernel on the host (include/SushiBLAS/engine/task.hpp:63). I did not survey src/io.cpp or constant.cpp for host loops.

NEVER COMPILED FOR A NON-SPIR-V TARGET
Everything: GPU_VALIDATION.md:30 says the AOT build was never compiled. Specific risks: the complex rewrite in include/SushiBLAS/engine/math/safe_math.hpp (KNOWN_ISSUES.md:17) replaced only the named functions (exp 127, tanh 153, erf 189, log 206, sqrt 231, abs 259). std::complex operator/ and operator* still reach device code: div.cpp:38, reciprocal.cpp:39, sigmoid.cpp:39, silu.cpp:39/46, softplus.cpp:51, complex GEMM accumulation (gemm.hpp:334, 441), and std::conj in kernels/sycl/blas/level1/level1.hpp:224. Whether MSVC's implementations of those compile for NVPTX is untested. Also untested there: the reinterpret_cast to sycl::vec in launch_1d.hpp:174/186, __restrict__ (gemm.hpp:286, 419), sycl::half kernels, and gelu.cpp's <numbers>.
Never executed anywhere: the wide pass and the grid-stride shape of launch_1d.hpp (273-286, 269); Launch::map plans internally (254), so a test cannot force them on the CPU.

KERNEL FUSION (the owner's question)
Fusion in this repository is the GEMM epilogue: a functor applied before each store of C (gemm.hpp:358, 462; include/SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp, channel_bias_epilogue.hpp), exposed through Dispatch::execute_gemm (include/SushiBLAS/engine/blas/dispatch.hpp:197). Launch::zip_inplace (launch_1d.hpp:300) is the seam for a fused in-place multi-array update. I found no general graph-level fusion pass in the files read; SushiAI and the runtime were not surveyed for one.

RANKED KERNEL CHANGES
Before 0.1.0 (they change a signature or a contract, or are cheap and remove an untested path): findings 1, 2, 3, 4, 5, 6 below, in that order.
Wait until after 0.1.0 (no interface change, or unmeasurable here): findings 7, 8, 9, plus everything GPU_VALIDATION.md sections 3.1-3.4 lists (double buffering, a sub-group-per-segment axis shape, bank-conflict tuning).

### Findings

#### 1. Launch and reduce seams plan internally, so GPU-shaped paths cannot run on the CPU device

Size: M, about 4 files (launch_1d.hpp, reduce_1d.hpp, 2 test files)

Evidence: Launch::map, zip_inplace and indexed call plan<T>(q, n, aligned) or plan_scalar themselves (include/SushiBLAS/support/launch_1d.hpp:254, 310, 369); reduce_indexed calls plan(Device::profile_for(queue), ...) itself (reduce_1d.hpp:636). The sibling reduce_axis takes the geometry as a parameter (reduce_axis.hpp:130-131), which is why Unit_AxisReductions can drive both shapes (GPU_VALIDATION.md:63). The wide pass and grid-stride shape have never executed (GPU_VALIDATION.md:99-109).

Proposal: Give map, zip_inplace, indexed and reduce_indexed an overload that takes the Shape1D / Plan, as reduce_axis does, and keep the queue-planning form as a thin wrapper. Add CPU tests that force vector_steps > 0 and flat = false at sizes that are not a multiple of the vector width. Siblings then share one shape and two never-run paths get correctness coverage before release.

#### 2. SYRK always takes the local-memory shape and skips GEMM's fit checks and padding

Size: M-L, 3-4 files (gemm.hpp, syrk.hpp, a new shared header, kernels README)

Evidence: kernels/sycl/blas/level3/syrk.hpp:121-122 allocates local_accessors and 163/177 barriers on every device, CPU included, which include/SushiBLAS/kernels/README.md:93-97 calls a double copy plus a thread rendezvous. No check of local_mem_bytes or max_work_group_size before launch (compare gemm.hpp:498-502); README.md:156-157 says an oversized local_accessor is a runtime launch failure. Tile is unpadded and ownership contiguous (syrk.hpp:169-171), the 4-way and 2-way bank-conflict shapes README.md:213-224 describes. Fixed 4x4 block (57-60) against GEMM's derived one. Already in REMAINING_WORK.md:39-41.

Proposal: Lift the tiled-product bodies of gemm.hpp into a shared helper parameterised on a store predicate (triangle mask) and epilogue, and make SYRK a caller with the same local / no-local dispatch. SYRK's own tile config and kernel body go away.

#### 3. FLOAT64 kernels launch without checking the fp64 aspect

Size: S-M, 2-3 files

Evidence: Profile::supports_fp64 is probed (device_policy.hpp:110) and read in exactly one place, src/engine/math/random/poisson.cpp:50. No other file under src/, include/ or kernels/ reads it or sycl::aspect::fp64. AGENTS.md states "an unsupported dtype throws"; on a device without fp64 a double kernel would instead fail inside the SYCL runtime at submit.

Proposal: Check the aspect once in the shared dtype dispatch (the place the with_element_type / execute_blas_task arms resolve T) and throw the library's own unsupported-dtype error. Same for sycl::half and aspect::fp16.

#### 4. Device::Profile carries a field nothing reads, and one kernel bypasses the profile

Size: S, 2 files (breaking)

Evidence: global_mem_cache_line (device_policy.hpp:100, 117) has no reader in src/, include/, kernels/, tests/ or benchmarks/. kernels/sycl/random/orthonormalize.hpp:67 queries max_work_group_size from the device directly instead of Device::profile_for.

Proposal: Drop global_mem_cache_line before the struct is frozen (or use it for a CPU tile size, see finding 7), and route orthonormalize through profile_for and Reduce::floor_pow2.

#### 5. Five whole-tensor reductions and two logic ops sit outside the fold seam

Size: M, 7 op files plus reduce_1d.hpp

Evidence: argmax.cpp:49, argmin.cpp:50, var.cpp:50, std.cpp:50, logsumexp.cpp:51 are single_task loops (REMAINING_WORK.md:54-55). iamax already folds (value, index) pairs in parallel through Reduce::RunningArgMax (kernels/sycl/blas/level1/level1.hpp:242, reduce_1d.hpp:556). all.cpp:64 and any.cpp:64 use sycl::reduction, the only two left; GPU_VALIDATION.md:124 says reductions gave it up.

Proposal: argmax/argmin: add a RunningArgMax-shaped accumulator with their own tie and NaN rule and call reduce_indexed. logsumexp: a max reduce then a sum-of-exp reduce. var/std: a mean reduce then a sum-of-squared-deviation reduce, which changes last bits against the Welford loop. all/any: Reduce::reduce with an integer min/max accumulator. On a CPU under 16384 elements they stay serial anyway (reduce_1d.hpp:121).

#### 6. The complex rewrite for NVIDIA covers named functions only

Size: S-M, safe_math.hpp plus about 8 call sites

Evidence: safe_math.hpp replaces exp, tanh, erf, log, sqrt, abs (lines 127-262). std::complex division and multiplication remain in device lambdas: src/engine/math/elementwise/div.cpp:38, reciprocal.cpp:39, nonlinear/sigmoid.cpp:39, silu.cpp:39 and 46, softplus.cpp:51; complex GEMM at gemm.hpp:334 and 441; std::conj at level1.hpp:224. KNOWN_ISSUES.md:17 records that no NVIDIA build was tried.

Proposal: Before 0.1.0 either add safe_div / safe_mul / safe_conj beside the others and use them at those sites, or state in KNOWN_ISSUES that complex dtypes are unverified on AOT targets. Only a compile for an NVPTX target settles which is needed; that compile needs an intel-llvm built with --cuda (GPU_VALIDATION.md:25).

#### 7. CPU GEMM has no packing, no k-blocking and a fixed 4x4 block (wait)

Size: L, gemm.hpp, device_policy.hpp, tests, README

Evidence: gemm.hpp:426-437 (full-k loop over strided global reads), :101 and :181 (64-byte budget, 4x4 float, 2x2 double), :87 (bounds branch per load), :394-397 (fixed lane axis); device_policy.hpp:58, 267. Details in the summary.

Proposal: After 0.1.0 and with bench_gemm baselines recorded: (a) pick the nd_range axis order from the strides so the unit-stride axis is the vectorised one; (b) split interior blocks from edge blocks so load_operand's branch leaves the inner loop; (c) add a k-block sized to L1/L2 with A packed per work-group into private or USM scratch; (d) derive the register block from a native-vector-width field in Profile. None changes the gemm signature, so none forces a later break. Each needs a measured before and after.

#### 8. TRMV, TRSV and TRSM run serially (wait, but decide the signature now)

Size: L, 4-6 files

Evidence: trmv.hpp:58 single_task; trsm.hpp:62-128 one serial substitution per column; trsv.cpp:96 calls trsm with one column. REMAINING_WORK.md:42-45 notes the HALF fix needs scratch the size of the right-hand side, "so a kernel signature changes".

Proposal: A parallel TRMV needs an n-element scratch (each output reads only old x). TRSM needs level-scheduled or blocked substitution. Both are kernel-internal under kernels/, which is not the installed GEMM seam, so they can follow 0.1.0; if kernels/ signatures are treated as public, add the scratch parameter now and leave the body serial.

#### 9. shuffle is a sequential Fisher-Yates in one task (wait)

Size: M, 1-2 files (breaking)

Evidence: src/engine/math/random/shuffle.cpp:55-66.

Proposal: A parallel shuffle (sort by Philox keys) changes the permutation a seed produces. Leave it for after 0.1.0 and say in the release notes that the stream for shuffle is not frozen, or freeze the serial one.

#### 10. sb has no device selection, no profile report and no GPU label, so a GPU gate cannot be expressed

Size: M, cli.py, suite.py, project.py, tests CMake labels, one new tool source, bench_common.hpp

Evidence: sb test takes only --suite, --filter, --repeat (cli/sushiblas/cli.py:116-128); Suite has no device label (cli/sushiblas/services/suite.py:13-30); the environment passes straight through (services/project.py:184-198), so ONEAPI_DEVICE_SELECTOR is set by hand (REMAINING_WORK.md:9-12). Nothing prints Device::Profile, which GPU_VALIDATION.md:36-42 asks the reader to confirm. GPU-only tests skip silently (tests/functional/unit/test_gemm.cpp:781-783). Benchmarks print a text table only (benchmarks/bench_common.hpp:216-231). cli/README.md:107 names the target nvptx64-nvidia-cuda where every other document says nvidia_gpu_sm_86.

Proposal: Minimal gate: (1) sb test/run --device <selector> that sets ONEAPI_DEVICE_SELECTOR for the child; (2) a small program or sb subcommand that prints the Device::Profile of the selected device, and exits non-zero when is_cpu is true under a GPU gate; (3) a CTest label `device` for tests that must not skip, with skips counted as failures under --suite device; (4) compile-only step: sb build -D SB_SYCL_TARGETS=... already exists and is the first gate, with -Xcuda-ptxas -v output captured for the spill count; (5) --json on the bench programs so a run can be stored and compared; (6) fix the target name in cli/README.md.

#### 11. The SYCL level 1 kernels depend on the reference kernels' header

Size: S, 3 files

Evidence: kernels/sycl/blas/level1/level1.hpp uses Reference::RealOf and Reference::abs1 (lines 107, 149-155, 239-246), so the SYCL brick cannot be detached from the reference brick. Two kernel roots exist: kernels/ and include/SushiBLAS/kernels/ (syrk.hpp:39 includes across them; KNOWN_ISSUES.md:45).

Proposal: Move RealOf and abs1 to a shared traits header both families include. The one-home question for kernels is already an owner decision in REMAINING_WORK.md:129-130.

### Decisions raised

- Is include/SushiBLAS/support (Launch, Reduce, Device::Profile) part of the 0.1.0 public interface? It is under the public include root, and findings 1 and 4 change it.
  - Options: Public: make the plan-injection overloads and the Profile field removal now, then freeze | Internal: mark it as not covered by the stability promise and change it freely later
  - Recommended: Public: make the plan-injection overloads and the Profile field removal now, then freeze

- How much CPU GEMM work goes into 0.1.0?
  - Options: None: ship the current kernel and state that CPU GEMM is not tuned | Only the two low-risk changes (axis order by stride, branch-free interior blocks), each with a recorded bench_gemm before and after | Full packing and k-blocking before release
  - Recommended: Only the two low-risk changes (axis order by stride, branch-free interior blocks), each with a recorded bench_gemm before and after

- FMA contraction is off through sushiruntime_deterministic_fp. Keep it for GEMM?
  - Options: Keep off everywhere; the determinism contract stays as written | Allow contraction in the BLAS level 3 kernels only and restate the contract
  - Recommended: Keep off everywhere; the determinism contract stays as written

- var and std in parallel form change the last bits against the current Welford loop. Accept that before 0.1.0?
  - Options: Yes, switch to two fixed-order reduces now, before any result is treated as frozen | No, leave var/std serial and move only argmax, argmin, logsumexp, all, any
  - Recommended: Yes, switch to two fixed-order reduces now, before any result is treated as frozen

- Complex dtypes on AOT (NVIDIA) targets cannot be verified on this machine. What does 0.1.0 say?
  - Options: Add safe_div, safe_mul and safe_conj now and still list the path as unverified | Leave the code and list complex on AOT targets as unverified in KNOWN_ISSUES
  - Recommended: Add safe_div, safe_mul and safe_conj now and still list the path as unverified

- Is a GPU run a release condition for v0.1.0, or does 0.1.0 ship as CPU-validated with the sb gate in place for the first person with hardware?
  - Options: Ship CPU-validated; build the sb device gate and say so in the release notes | Hold the tag until GPU_VALIDATION.md sections 1, 2 and 5 pass on real hardware
  - Recommended: Ship CPU-validated; build the sb device gate and say so in the release notes

- Does the shuffle permutation for a given seed count as frozen at 0.1.0?
  - Options: Not frozen; a parallel shuffle may change it later | Frozen; shuffle stays sequential
  - Recommended: Not frozen; a parallel shuffle may change it later

## scout/memory (Scout: memory)

Read-only survey of memory behaviour in D:\Projects\sushiblas and the allocator side of D:\Projects\sushiruntime. I wrote no file, ran no build and no `sb`; the only tools used were Read, Grep and Glob. Rule 6 (syntax check) and rule 7 (scratch) are vacuous here: nothing was written, the scratchpad was not touched (0 bytes). Rule 2: the humanizer skill was loaded before this report was written.

What was read: include/SushiBLAS/storage.hpp, tensor.hpp, engine/scratch.hpp, engine/task.hpp, engine/blas/dispatch.hpp, engine/blas/utils.hpp, engine/math/reductions/dispatch.hpp, engine/math/random/dispatch.hpp, support/device_policy.hpp, support/reduce_1d.hpp (560-731), support/reduce_axis.hpp, kernels/sycl/blas/level3/gemm.hpp (60-310), io.hpp, src/engine.cpp, src/io.cpp (40-110, 300-811), src/engine/blas/level1/dot.cpp, level3/trsm.cpp, spatial/im2col.cpp, math/random/orthogonal.cpp, shuffle.cpp, kernels/sycl/random/orthonormalize.hpp; in SushiRuntime: execution/runtime_context.hpp, src/execution/runtime_context.cpp (525-714), memory/README.md, memory/allocation.hpp, memory/usm_allocation_owner.hpp, core/resource_liveness.hpp, graph/task_graph.hpp, src/graph/task_graph.cpp (50-137, 600-758). The rest of src/ and kernels/ was covered by grep, not read line by line.

Headline facts:
1. There is one USM allocation path: Storage -> RuntimeContext::malloc_{device,host,shared}_on -> plain sycl::malloc_* (sushiruntime src/execution/runtime_context.cpp:543, 562, 581). No alignment argument is passed, so the 128-byte DEFAULT_ALIGNMENT (sushiruntime core/common.hpp:63) applies only to the rounded size (storage.hpp:73-76), not to the address.
2. SushiRuntime has no USM pool or arena. A grep for Pool/Arena classes in its src/ and include/ finds only WorkerPoolInfo; the one pool in memory/ recycles epoch retire nodes (memory/README.md:12-20). Its memory README states the design assumption "a measured run allocates nothing per step" (memory/README.md:78-80). SushiBLAS breaks that assumption in the reduction families.
3. The only per-call device allocation in SushiBLAS is ScratchBuffer, at 6 construction sites, each at most 256 partials x 16 bytes = 4 KiB, rounded up to 128 bytes.
4. No kernel allocates a host or USM temporary. A grep over kernels/, include/SushiBLAS/kernels and include/SushiBLAS/support for std::vector locals, malloc, sycl::buffer and memcpy finds only work-group local_accessor tiles and fixed private arrays. GEMM has no packing buffer; TRSM, im2col, col2im, pooling, shuffle and the axis reductions work in place on operand pointers.
5. Scratch lifetime is sound on the normal path: the task closure holds the ScratchBuffer by value (dot.cpp:73 via [=], reductions/dispatch.hpp:132, 143, 154, 270), TaskGraph::compile moves node ownership into the CompiledGraph (task_graph.cpp:610-625), and Engine::execute keeps that object alive until the execution engine returns (engine.hpp:163-170). Not verified: that IExecutionEngine::execute waits for every submitted SYCL event before returning, and what happens to in-flight kernels when it throws. engine.hpp:161 states the call blocks; I did not read the engine implementation.
6. Operand lifetime is not sound: every task captures raw element pointers and none retains the Storage.

On the question "is kernel fusion done here": SushiBLAS fuses only where the caller hands it a functor. The GEMM epilogue is the one built-in fusion seam (engine/blas/dispatch.hpp:205-305, ARCHITECTURE.md:406, 577), and the elementwise dispatch helpers take a caller-written functor. Nothing in SushiBLAS merges two recorded tasks into one kernel: each operation is recorded as one opaque std::function through TaskGraph::add_task (task.hpp:257, 286), and SushiRuntime's graph README says the runtime cannot look inside it (sushiruntime src/graph/README.md:64-66). The word "fusion" appears in SushiRuntime's fluent layer (api/graph/graph.hpp, api/graph/README.md, execution/README.md), which SushiBLAS does not use; I did not read that code, so what it fuses is not established here.

### Findings

#### Recorded tasks capture raw operand pointers and do not keep the Storage alive

Size: L, about 70 files (task.hpp, the 6 family dispatch headers, every op .cpp that builds reads/writes by hand)

Evidence: Every dispatch helper resolves operands to raw pointers at record time and captures only those: engine/blas/dispatch.hpp:251-253 and 280-303 (read_A, read_B, rw_C), engine/math/elementwise/dispatch.hpp:150 (pA_raw, pB_raw, pC_raw), engine/math/reductions/dispatch.hpp:110-111 and 132, engine/math/random/dispatch.hpp:230 and 248, src/engine/blas/level3/trsm.cpp:73-74, src/engine/spatial/im2col.cpp:56-57, src/engine/math/random/orthogonal.cpp:71 and 81, shuffle.cpp:80 and 97. A grep for a Tensor or sushi_ptr<Storage> captured in any lambda under src/ and include/SushiBLAS/engine finds none. Storage frees its USM when the last Tensor goes (storage.hpp:117-123). So a Tensor that is recorded into an operation and destroyed before Engine::execute leaves the task with a pointer to freed USM. SushiRuntime has a mechanism for exactly this (core/resource_liveness.hpp:45-59, used by Buffer at api/data/buffer.hpp:154, 363 and observed by the fluent graph at api/graph/graph.hpp:2299-2313, 3005), but TaskGraph::add_task takes no observer (graph/task_graph.hpp:176-195) and SushiBLAS has no reference to liveness anywhere (grep over src, include, kernels: no match). The only lifetime contract written down is for the context (storage.hpp:57, 95) and the epilogue bias pointer (kernels/blas/level3/channel_bias_epilogue.hpp:38). docs/reference/KNOWN_ISSUES.md does not list this.

Proposal: Add one brick, an operand set built from Tensors, that yields the three things every operation now assembles by hand: the read and write ResourceRegion lists, the host pointers for require_host_reachable, and a keep-alive list of sushi_ptr<Storage>. The task body holds the keep-alive list, the same way it already holds ScratchBuffer. This removes the hand-written reads/writes/ReferenceOperands triplet from each op file and closes the use-after-free without changing any public signature. Cost per task: one atomic increment and decrement per distinct operand storage.

#### Allocation base alignment is not guaranteed, though the code and the manual say it is

Size: S in SushiBLAS (storage.hpp, tensor.hpp, ARCHITECTURE.md); S in SushiRuntime (runtime_context.hpp/.cpp), which is a lower-tier release

Evidence: Storage rounds only the byte count to DEFAULT_ALIGNMENT (storage.hpp:71-76) and then calls ctx.malloc_device_on / malloc_host_on / malloc_shared_on (storage.hpp:139-141), which call sycl::malloc_device / malloc_host / malloc_shared with no alignment (sushiruntime src/execution/runtime_context.cpp:543, 562, 581). The only aligned entry point SushiRuntime offers is host-only: aligned_malloc_host_on -> sycl::aligned_alloc_host (runtime_context.cpp:640-647, runtime_context.hpp:244). DEFAULT_ALIGNMENT is 128 (sushiruntime core/common.hpp:63). Tensor::is_aligned tests the address against 128 (tensor.hpp:343-347) and has no caller in src/, include/ or kernels/ (grep). docs/architecture/ARCHITECTURE.md:206-208 says the BLAS layer uses is_aligned() to pick the flat path, and ARCHITECTURE.md:226-227 says 'the allocation base is 64-byte aligned'; neither statement is backed by the code. `class alignas(64) Storage` (storage.hpp:46) aligns the control object, not the data. A view at a non-zero storage_offset is aligned only to its element size (tensor.hpp:451).

Proposal: Give SushiRuntime aligned shared and device entry points beside the host one (sycl::aligned_alloc_shared / aligned_alloc_device), have Storage pass DEFAULT_ALIGNMENT, and state the guarantee in storage.hpp. If the owner does not want a SushiRuntime change now, the alternative is to delete Tensor::is_aligned and the two ARCHITECTURE.md sentences so that v0.1.0 does not publish a promise it cannot keep. Either way, decide which of 64 and 128 is the contract; the manual says 64 and the constant says 128.

#### One device USM malloc and free per recorded reduction (the only per-call device allocation)

Size: M, 3-4 files (scratch.hpp, a new pool header, engine.hpp to own it, utils.hpp for the mode check)

Evidence: ScratchBuffer allocates a fresh Residency::Device Storage each time (engine/scratch.hpp:56-65). Construction sites: engine/blas/utils.hpp:137 (called from src/engine/blas/level1/asum.cpp:54, nrm2.cpp:54, dot.cpp:62, iamax.cpp:56), engine/math/reductions/dispatch.hpp:118 (execute_reduction, reached by every folding reduction through dispatch.hpp:298-309: sum, mean, prod, min, max and others) and dispatch.hpp:253 (count_nonfinite). Size class: partials x partial size, where partials <= K_MAX_GROUPS = 256 (support/reduce_1d.hpp:48, 128) and a partial is 16 bytes for Level 1 (kernels/sycl/blas/level1/level1.hpp:128), 4 to 16 bytes for the math reductions (reductions/dispatch.hpp:70-74) and 8 bytes for counting; so at most 4 KiB, rounded up to a multiple of 128. planned_partials returns 0 for a single group (reduce_1d.hpp:131), so small inputs allocate nothing. Each allocation also costs: one `new Storage` (sushiruntime core/sushi_ptr.hpp:252), one UsmRegistry::record and one take under the registry's single lock (sushiruntime memory/README.md:78-99), and one sycl::free at task destruction. The allocation happens at record time, so N pending reductions hold N device blocks until execute. Under FORCE_REFERENCE the Level 1 scratch is still allocated and never read (dot.cpp:62-64 is unconditional; the reference lambda at dot.cpp:84, 93 does not use it).

Proposal: Introduce a per-Engine scratch pool: one free list per device index and per size class (128 B to 4 KiB in powers of two covers every current site). ScratchBuffer becomes a lease that returns its block to the pool when the last copy dies, keeping its present interface (data, bytes, is_empty, region). Skip the lease when the engine mode is FORCE_REFERENCE. Expected effect, not measured: after warm-up a reduction performs no sycl::malloc_device, no sycl::free, no registry lock and no heap `new Storage`. A training loop that calls a few reductions per step currently pays all four per call.

#### IO::to_string copies the whole view and widens every element to 16 bytes before printing a handful

Size: S, 2 files (src/io.cpp, io.hpp for the private print_recursive signature)

Evidence: src/io.cpp:703-707 allocates host_buf of element_span() x bytes_per_element and copies the whole span through the queue; io.cpp:709-713 then builds a std::vector<std::pair<double,double>> of span elements (16 bytes each) and decodes every element. print_recursive prints at most 2 x edge_items entries per axis (io.cpp:773, 791; default edge_items = 3, io.hpp:240). For a float32 tensor of N elements the host peak is 4N + 16N bytes; a 100-million-element tensor costs about 2 GB to print 36 numbers at rank 2. For a strided view the span can also exceed num_elements.

Proposal: Decode only the elements print_recursive visits. For Host and Shared residency read through the pointer; for Device residency copy one innermost run at a time (at most 2 x edge_items elements when elided). The decode step becomes a small reader object handed to print_recursive in place of the pre-decoded array. Expected effect: host memory for a print drops from 20 bytes per element of the span to a constant bounded by (2 x edge_items)^rank elements.

#### File IO stages the whole payload in a second host buffer

Size: S, 1 file (src/io.cpp)

Evidence: read_payload always allocates std::vector<char> buf(bytes), reads the file into it and then copies it into the tensor, for Host and Shared residency as well as Device (src/io.cpp:442-448); peak host memory is twice the payload for a tensor the host could have read into directly. write_payload does the same for Device residency only (io.cpp:344-349). Both vectors are value-initialised, so the buffer is zero-filled before it is overwritten. The doc comment on read_payload gives the reason for the read side: 'stores them once all arrived' (io.cpp:435), so a short file leaves the tensor untouched. IO::read and IO::write copy directly with one queue memcpy and no staging (io.cpp:674-675, 685-686). to_vector zero-fills its result before read overwrites it (io.hpp:170-171).

Proposal: Check the remaining file length against the payload size before any byte is stored (one seek), then stream in fixed chunks of a few MiB: straight into the tensor for Host and Shared, through one reused chunk buffer and queue memcpy for Device. That keeps the all-or-nothing behaviour for truncated files and caps the extra host memory at one chunk. A read error after the length check would leave a partly written tensor, which is the trade the owner has to accept or refuse.

#### Host heap allocations made for every recorded operation

Size: S for the three local changes (task.hpp, random/dispatch.hpp and its ~12 callers); the rest depends on SushiRuntime

Evidence: Per call, at record time: (a) two std::vector<ResourceRegion> for reads and writes, built in every op, for example engine/blas/dispatch.hpp:256-261, src/engine/blas/level1/dot.cpp:56-60, engine/math/reductions/dispatch.hpp:114-115; TaskGraph::add_task takes them as const std::vector& (sushiruntime graph/task_graph.hpp:190-195). (b) ReferenceOperands owns a std::vector<const void*> built from an initializer list (engine/task.hpp:87, 94), used by every BLAS and spatial op (dispatch.hpp:279, trsm.cpp:94, im2col.cpp:71) and read only under FORCE_REFERENCE (task.hpp:109). (c) TypedTaskBody stores the name as std::string (task.hpp:169-171, 232) although TaskMetadata::name is a const char* (sushiruntime graph/task_types.hpp:114); names of 16 or more characters allocate. (d) the closure goes into a std::function (HostWork, task_graph.hpp:64), which heap-allocates for closures of this size. (e) execute_random takes params as const std::vector<double>& (engine/math/random/dispatch.hpp:223), so each random op builds a vector for two or three doubles. (f) GemmEpilogue holds two more region vectors that are then copied into the task's (blas/dispatch.hpp:126-127, 189-192). Inside SushiRuntime each task adds a Node, a NodeEntry and up to two RawNodeEntry allocations (task_graph.cpp:54-67).

Proposal: Within SushiBLAS: make ReferenceOperands a fixed-capacity array (no operation lists more than 4 pointers), store the name as const char* in TypedTaskBody, and take RNG params as a span or std::array. These three remove one to three heap allocations per recorded op with no public signature change outside Internal. The region vectors and the std::function need add_task to accept a span and a small-buffer callable, which is a SushiRuntime interface change and should be raised there, not worked around here. If the operand-set brick from the first finding is built, (a) and (b) are solved inside it.

#### Engine::execute compiles and discards the graph on every call

Size: L, not estimated in files; touches Engine's public surface (breaking)

Evidence: engine.hpp:163-170: execute() calls graph_.compile(), runs the result and lets it go out of scope. compile() moves every node out of the builder (sushiruntime src/graph/task_graph.cpp:610-625), builds an unordered_map of in-degrees and a ready vector for the cycle check (task_graph.cpp:665-671) and resets the tracker (715). SushiRuntime's own graph README describes that cost as 'paid once at compile and amortised over every replay of the plan' (src/graph/README.md:118-119), but SushiBLAS never replays: a loop that runs the same operations each step re-records every task, re-allocates every closure, node and scratch buffer, and recompiles.

Proposal: This is an architecture question more than a memory one, so I only record the fact and its memory cost: all per-op allocations in the previous finding recur on every step. A record-once, run-many plan object on Engine would remove them at the source; whether v0.1.0 should carry that is for the architecture stream and the owner to decide.

#### create_tensor and reshape build a std::vector for a brace list, and create_tensor cannot name a device

Size: S, 3 files (engine.hpp, engine.cpp, tensor.hpp) plus callers that pass a named vector, which keep compiling (breaking)

Evidence: src/engine.cpp:81 and 88 turn the initializer_list into a std::vector<int64_t> before delegating; tensor.hpp:381 does the same for reshape. Tensor's own constructor already takes SushiRuntime::span<const int64_t> (tensor.hpp:162, 174). The public overloads take const std::vector<int64_t>& (engine.hpp:199, 207; tensor.hpp:389). Separately, Engine::create_tensor passes no device index (engine.cpp:103), so every tensor lands on device 0 by the Storage default (storage.hpp:66), although Storage and ScratchBuffer both support a device index and IO routes through storage->get_device_index() (io.cpp:67).

Proposal: Before the interface freezes, let create_tensor and reshape take span<const int64_t> (a vector and a brace list both convert), which removes the temporary vector and two of the four create_tensor overloads. Decide at the same time whether create_tensor takes a device index; adding it later as a defaulted trailing parameter is source-compatible but changes the exported symbol.

#### Storage exposes its sizes as public mutable fields; one of them is never read

Size: S, 4 files (storage.hpp, tensor.hpp, scratch.hpp unaffected, test_storage_offset.cpp) (breaking)

Evidence: storage.hpp:50 and 53: size_bytes and requested_bytes are public data members, writable by any holder of a sushi_ptr<Storage>. Tensor's bounds check trusts size_bytes (tensor.hpp:206-208). A grep over src/, include/, kernels/ and tests/ finds no reader of requested_bytes outside storage.hpp:67. size_bytes is read by tests (tests/functional/unit/test_storage_offset.cpp:70, 85). The other three fields are private with accessors (storage.hpp:147-156), so the class is half encapsulated.

Proposal: Make both private behind const accessors, or drop requested_bytes if nobody needs the pre-rounding size. Doing this before v0.1.0 avoids breaking it afterwards.

#### Engine::create_tensor throws for a zero-sized shape although Tensor supports empty tensors

Size: S, 2 files (engine.cpp, engine.hpp)

Evidence: checked_allocation_bytes returns 0 when any dimension is 0 (src/engine.cpp:47-58); Storage's constructor throws 'Requested size cannot be zero' (storage.hpp:69). Tensor itself handles num_elements == 0 (tensor.hpp:198-199, 300, 328), the operations skip empty tensors, and tests/functional/unit/test_empty_tensors.cpp exercises them only as storage-less tensors built by hand (lines 104, 118 say 'storage-less empty tensor'). I did not run anything; this follows from reading the two functions.

Proposal: Decide the contract: either create_tensor returns a storage-less empty Tensor for a zero-sized shape, or the throw is documented as intended on engine.hpp. The first matches what the operations already accept.

#### Small avoidable costs on the tensor and device-lookup paths

Size: S, 1 file for (a) and (b) (tensor.hpp)

Evidence: (a) The Tensor constructors take sushi_ptr<Storage> by value and copy it at each of up to three delegation steps, ending in storage(s) without a move (tensor.hpp:162-165, 174-177, 215-218); each copy is an atomic increment and decrement on the Storage (sushiruntime core/ref_counted.hpp:47, 82-86). slice and reshape go through this chain (tensor.hpp:408, 437). (b) Tensor::get_device looks the base pointer up in the USM registry under its lock (tensor.hpp:227-230, sushiruntime memory/README.md:78) although the Storage already knows its device index (storage.hpp:109), which io.cpp:67 uses directly. (c) Device::profile_for takes a process-wide mutex and scans a deque comparing sycl::device objects on every call (support/device_policy.hpp:146-159). It is called at record time (engine/blas/utils.hpp:136, reductions/dispatch.hpp:87, count_nonfinite.cpp:41) and again at run time inside the kernels' launch code (reduce_1d.hpp:636, reduce_axis.hpp:146, reductions/dispatch.hpp:433, gemm.hpp:248 and 251 inside SB_ASSERT). (d) orthonormalize reassigns a one-element std::vector<sycl::event> for each of its 2 x (count - 1) projection submissions (kernels/sycl/random/orthonormalize.hpp:201-208).

Proposal: (a) std::move the pointer through the constructor chain. (b) Implement get_device as context.get_queue(storage->get_device_index()).get_device(). (c) Leave profile_for as it is unless a profile shows contention; if it does, cache the Profile pointer per device index on the Engine. (d) Leave; the cost is noise next to the kernel submissions. None of these is measured; (a) and (b) are certain small wins, (c) is a guess until measured.

#### Inventory: what does not allocate

Size: none

Evidence: Checked by grep over kernels/, include/SushiBLAS/kernels and include/SushiBLAS/support for std::vector locals, resize/reserve/push_back, malloc, sycl::buffer, memcpy and make_shared: the only hits are device_policy.hpp:127 (one-time device probe), device_policy.hpp:150 (the intentionally leaked profile cache) and orthonormalize.hpp:201. Work-group local memory, which the SYCL runtime provides per launch and is not a heap or USM allocation: GEMM tiles of block_m x 17 and 16 x block_n elements (kernels/sycl/blas/level3/gemm.hpp:267-270, geometry at 131-170), SYRK tiles (kernels/sycl/blas/level3/syrk.hpp:121-122), the reduction tile of `local` elements (reduce_1d.hpp:660, reduce_axis.hpp:177), the Level 2 row tile (kernels/sycl/blas/level2/row_reduction.hpp:68) and 2 x group elements with group <= 64 in orthonormalize (orthonormalize.hpp:59, 121, 161). Private per-work-item arrays are fixed size (gemm.hpp:288, 322-323, 428-429; launch_1d.hpp:320, 341, 353). Shuffle swaps in place in one single_task (src/engine/math/random/shuffle.cpp:55-65). im2col, col2im, pooling, upsample, broadcast and TRSM pass operand pointers straight to the kernel (im2col.cpp:77-84, trsm.cpp:99-110). No USM-to-USM copy exists outside io.cpp and the user-requested Level 1 copy. I did not open the TRSM, SYRK, pooling or Level 2 kernel bodies beyond the grep hits.

Proposal: Nothing to change. Keep this property: the documented rule that an epilogue 'neither allocate nor throw' (engine/blas/dispatch.hpp:199) could be stated once for all kernels in kernels/README.md.

### Decisions raised

- How should a recorded task treat an operand Tensor that is destroyed before Engine::execute?
  - Options: The task retains the Storage (strong reference), so the memory stays valid until the task has run | The task observes liveness and execute throws by name when an operand has died, as SushiRuntime's Buffer does | Keep raw pointers and document that every operand must outlive execute
  - Recommended: The task retains the Storage. It costs one atomic increment per operand, needs no SushiRuntime change, and removes a use-after-free from the public contract of v0.1.0.

- Should the tensor allocation base carry an alignment guarantee, and which one?
  - Options: Add aligned shared and device allocation to SushiRuntime and guarantee DEFAULT_ALIGNMENT (128) in Storage | Same, but guarantee 64 as ARCHITECTURE.md says, and change the constant or the check | Guarantee nothing: remove Tensor::is_aligned and the two ARCHITECTURE.md sentences
  - Recommended: Add the aligned entry points in SushiRuntime and guarantee DEFAULT_ALIGNMENT. It is a lower-tier change, so it needs your approval for that repository; until it lands, remove the two manual sentences so the docs stop claiming it.

- Where does the scratch pool live?
  - Options: In SushiBLAS, owned by Engine, behind ScratchBuffer | In SushiRuntime, as a pooled allocator on RuntimeContext that Storage also uses | No pool; keep one malloc per reduction
  - Recommended: In SushiBLAS behind ScratchBuffer for v0.1.0: the blocks are small (at most 4 KiB), the change is local and no public interface moves. A runtime-wide pool is a larger design that also changes tensor allocation and should be planned separately.

- May a file load leave a partly written tensor when the read fails after the length check passed?
  - Options: Yes: check the file length first, then stream in chunks straight into the tensor | No: keep the full staging buffer and the present all-or-nothing behaviour
  - Recommended: Yes. A length check catches the truncated-file case the staging buffer was written for, and peak host memory drops from twice the payload to one chunk.

- Do create_tensor and reshape change signature before the interface freezes?
  - Options: Take span<const int64_t>, and add a device index to create_tensor now | Take span<const int64_t> only; add the device index later | Leave both as they are
  - Recommended: Take span and add the device index now. Storage, ScratchBuffer and IO already work per device, and create_tensor is the one place that cannot reach it; adding it after v0.1.0 changes an exported symbol.

- What does Engine::create_tensor do for a shape with a zero dimension?
  - Options: Return a storage-less empty Tensor | Keep throwing and document it
  - Recommended: Return a storage-less empty Tensor; the operations and the tests already treat that as the empty case.

- Do Storage::size_bytes and requested_bytes become private before v0.1.0?
  - Options: Make both private with const accessors | Make size_bytes private with an accessor and delete requested_bytes | Leave them public
  - Recommended: Make size_bytes private with an accessor and delete requested_bytes, which nothing reads. This is a deletion of something I did not create, so it needs your word.

## scout/numerics-left (Scout: numerics-left)

Read-only survey of the nine open numeric and behavioural items in D:\Projects\sushiblas. Nothing was written, built or run, so items 6 (syntax check) and 7 (scratchpad) are vacuous: no file was created and the scratchpad was not touched. The humanizer skill was loaded before writing this report. Every claim below comes from reading the cited lines; nothing was executed, so claims about runtime behaviour on a device are marked where they rest on reasoning only.

Result per item against docs/design/REMAINING_WORK.md and docs/reference/KNOWN_ISSUES.md:
- single_task reductions: doc accurate. Five files, all confirmed.
- max_along NaN: defect is real but the doc describes it wrongly. The code ignores NaN deterministically; it is not order-dependent. There is no min_along, and "the other axis reductions" do not use RunningMax.
- TRSM/TRSV HALF rounding: doc accurate.
- Mixed exception types: doc accurate for BLAS but too narrow; the same split runs through every family, and two more types (std::logic_error) are in play.
- Empty tensor in reductions: doc accurate; a null dereference sits one line behind the throw.
- ElementScalar: not recorded in either document. The whole five-member object is captured into the device kernel. Whether that trips fp64/fp16 aspect checks was not verified.
- wrapping_mul INT16: not recorded in either document. Latent only: there is no INT16 dtype, so the undefined multiplication is never instantiated.
- Epilogue operands outside the FORCE_REFERENCE guard: confirmed, and recorded only as a caller duty in a README warning, not in KNOWN_ISSUES.
- ExecutionMode::VERIFY: doc accurate. A public enumerator exists whose only behaviour is to throw.

Three of these touch the public interface or its contract and are worth settling before v0.1.0: the exception type, the VERIFY enumerator, and what a reduction of an empty tensor returns. The rest are internal and can land without a break.

Not done: no test was run to confirm any behaviour, the audit report under docs/agent/2026_10_05_ESTATE_AUDIT was not opened, Storage's behaviour for a zero-byte allocation was not read, and the test suite was not searched for EXPECT_THROW sites that a change of exception type would touch.

### Findings

#### var, std, logsumexp, argmax and argmin run as one sequential single_task

Size: M, about 7 files (5 reduction .cpp, reduce_1d.hpp, reductions/dispatch.hpp) plus tests

Evidence: D:\Projects\sushiblas\src\engine\math\reductions\var.cpp:50, std.cpp:50, logsumexp.cpp:51, argmax.cpp:49, argmin.cpp:50 each submit h.single_task with a host-style loop over all elements. REMAINING_WORK.md:54-55 states this accurately. var.cpp:52-62 is a Welford recurrence, logsumexp.cpp:53-63 is two sequential passes (max, then sum of exp(x - max)), argmax.cpp:51-61 is a scan with `pT[i] > max_val`. The other single_task hits are legitimate: reduce_1d.hpp:644 and :695 (sequential plan and the final fold of partials), max.cpp:119 / min.cpp:119 / all.cpp:55 / any.cpp:55 (empty input), kernels/sycl/blas/level2/trmv.hpp:58 and random/shuffle.cpp:55 (not opened). Side effect in argmax.cpp:55: a NaN at index 0 wins and a NaN elsewhere is skipped, which is exactly the rule Reduce::RunningArgMax documents at include/SushiBLAS/support/reduce_1d.hpp:566-595.

Proposal: argmax: replace the scan with Reduce::reduce_indexed<Reduce::RunningArgMax> over IndexedValue, the way iamax already does (sibling: blas.lvl1.iamax, src/engine/blas/level1/iamax.cpp with kernels/sycl/blas/level1/level1.hpp); the NaN rule is already identical so results do not change. argmin: add Reduce::RunningArgMin beside RunningArgMax in reduce_1d.hpp, same four members (identity, combine, accumulate, value). var/std/logsumexp: build each from two chained Reduce::reduce calls inside one task, the second depending on the first's event: mean via CompensatedSum into one scratch slot, then CompensatedSum of (x - mean)^2 with a finalize that divides by n (std adds sqrt); logsumexp is NaNPropagatingMax into the slot, then CompensatedSum of exp(x - max) with finalize max + log. Sibling shape: ReductionOps::max in src/engine/math/reductions/max.cpp:101-130 (planned_partials_for, then execute_reduction with scratch). execute_reduction needs one extra scratch element for the intermediate. No public signature changes. This is performance work, not interface work, so it can follow v0.1.0 without a break.

#### max_along ignores NaN; the known-issues row calls it order-dependent, which the code does not support

Size: S, 4 files (reduce_1d.hpp, max.cpp, min.cpp, max_along.cpp) plus a test and two doc rows

Evidence: src/engine/math/reductions/max_along.cpp:38 folds with Reduce::RunningMax. RunningMax::accumulate (include/SushiBLAS/support/reduce_1d.hpp:419-423) is `if (max_ < x) max_ = x;` starting from -infinity, so a NaN element is never stored and max_ can never become NaN. reduce_axis.hpp:163 and :194 feed raw elements only through accumulate; combine (reduce_1d.hpp:410-413, used at reduce_axis.hpp:203) only ever sees NaN-free partials. So max_along drops NaN in every order, and an all-NaN segment returns -infinity. KNOWN_ISSUES.md:16 says the result is order-dependent: inaccurate. The real defect is that max_along disagrees with whole-tensor max, which propagates NaN through NaNPropagatingMax (src/engine/math/reductions/max.cpp:47-98, combine at :66-75). REMAINING_WORK.md:46-47 and KNOWN_ISSUES.md:16 also say 'the other axis reductions': the only other callers are sum_along.cpp:38 and mean_spatial.cpp:41, both on CompensatedSum, which carries NaN by arithmetic. There is no min_along and no RunningMin anywhere in src, include or kernels.

Proposal: Move NaNPropagatingMax out of max.cpp's anonymous namespace and NaNPropagatingMin out of min.cpp's into include/SushiBLAS/support/reduce_1d.hpp beside RunningMax, as Reduce:: accumulators, and pass Reduce::NaNPropagatingMax at max_along.cpp:38. Sibling: ReductionOps::max. RunningMax then has no caller; deleting it needs the owner's word. Correct both documents: the behaviour is 'ignores NaN', and only max_along is affected. If min_along is wanted for v0.1.0 it is one new file shaped like max_along.cpp plus a declaration, which adds to the public interface.

#### HALF TRSM and TRSV round the solution to HALF between substitution steps

Size: M, 4 files (two trsm kernels, trsm.cpp, trsv.cpp) plus a HALF regression test

Evidence: Doc accurate (KNOWN_ISSUES.md:15, REMAINING_WORK.md:42-44). kernels/sycl/blas/level3/trsm.hpp:89-91: `W sum = W(alpha) * W(B_ref(i, j))`, then `sum -= W(A_eff(i, k)) * W(B_ref(k, j))`, then `B_ref(i, j) = ... T(sum ...)`. B_ref returns T& (:74-77), so each solved element is stored as HALF and read back as HALF by every later row. The same pattern repeats at :98-100, :115-117, :124-126 and in kernels/reference/blas/level3/trsm.hpp:71-73, :80-82, :99-101, :108-110. TRSV has no kernel of its own: src/engine/blas/level2/trsv.cpp:93 and :96 call Reference::trsm and SYCL::trsm with n right-hand-side columns = 1, so it inherits the defect. For FLOAT32/FLOAT64 W equals T (inferred from the doc statement and Accumulator's use; accumulator.hpp was not opened), so only HALF is affected.

Proposal: Give both trsm kernels a `W* work` operand holding the solution in the accumulation type, one W per right-hand-side element, written back to B as T once per column at the end. Allocate it at record time in src/engine/blas/level3/trsm.cpp and trsv.cpp only when the dtype is HALF, through ScratchBuffer with its region appended to writes. Sibling: Internal::allocate_level1_scratch in include/SushiBLAS/engine/blas/utils.hpp:128-141 as used by src/engine/blas/level1/nrm2.cpp:54-56 and :77. The kernel signature changes, but kernels/ is not the installed GEMM header set under include/SushiBLAS/kernels, so this is not a public break. Reference and SYCL kernels must change together so FORCE_REFERENCE and FORCE_SYCL keep agreeing.

#### Record-time checks throw two unrelated exception types, in every family and not only BLAS

Size: L if done library-wide (about 40 files plus tests); M for BLAS alone (19 .cpp under src/engine/blas plus dispatch.hpp and utils.hpp) (breaking)

Evidence: Doc accurate for BLAS (KNOWN_ISSUES.md:13, REMAINING_WORK.md:77-78) but too narrow. Mechanism: SB_THROW_IF throws std::runtime_error (include/SushiBLAS/core/logger.hpp:432-439); Dispatch::Detail::throw_operand_error throws std::invalid_argument (include/SushiBLAS/engine/operand_check.hpp:50-57). These share no base below std::exception.

BLAS level 1, std::runtime_error: dtype mismatch at swap.cpp:41, rot.cpp:41, copy.cpp:42, axpy.cpp:41, dot.cpp:42; length mismatch at swap.cpp:45, rot.cpp:45, copy.cpp:50, axpy.cpp:45, dot.cpp:49; non-contiguous rank > 1 at engine/blas/utils.hpp:70. std::invalid_argument: missing storage utils.hpp:61, non-positive increment utils.hpp:72, scalar result rules utils.hpp:112-118 (asum, nrm2, dot, iamax).

BLAS level 2, std::runtime_error: rank < 2 at gemv.cpp:43, ger.cpp:42, symv.cpp:43, syr.cpp:42, syr2.cpp:42, trmv.cpp:43, trsv.cpp:43; dtype mismatch at gemv.cpp:45, ger.cpp:44, symv.cpp:45, syr.cpp:44, syr2.cpp:44, trmv.cpp:45, trsv.cpp:45; not square at symv.cpp:51, syr.cpp:50, syr2.cpp:50, trmv.cpp:51, trsv.cpp:51; vector length at gemv.cpp:64-65, ger.cpp:60-61, symv.cpp:61-62, syr.cpp:57, syr2.cpp:60-61, trmv.cpp:58, trsv.cpp:59; trsv.cpp:60 (b not contiguous). std::invalid_argument: rank > 2 via require_level2_matrix_rank (utils.hpp:84-90), require_storage and require_contiguous on A in each file (lines 47-49), and get_vec_params.

BLAS level 3, std::runtime_error: engine/blas/dispatch.hpp:211 (rank), :213 (dtype), :219 (complex outside Dom), :116 (epilogue operand without storage); syrk.cpp:47-48; trsm.cpp:48-49. std::invalid_argument: everything in engine/blas/matrix_operand.hpp (describe_matrix_operand :62-67, batch_stride :98, require_same_batch_axes :122, require_same_layout :137, require_matrix_extent :156). describe_matrix_operand's own rank < 2 branch at :62 is unreachable from GEMM, SYRK and TRSM because the SB_THROW_IF above it fires first.

Shared by all BLAS: dtype outside the domain is std::runtime_error (engine/dispatch.hpp:92, engine/task.hpp:211); device-only USM under FORCE_REFERENCE is std::invalid_argument (task.hpp:122); an unhandled mode is std::logic_error (task.hpp:75); Engine::set_mode(VERIFY) is std::logic_error (src/engine.cpp:72).

Other families show the same split: dtype mismatch is runtime_error at engine/math/elementwise/dispatch.hpp:130 and :197, engine/logic/dispatch.hpp:65/124/187, engine/math/nonlinear/dispatch.hpp:126, engine/math/reductions/dispatch.hpp:107/402/457, while shape, layout, storage and contiguity in the same functions are invalid_argument. The random family already uses throw_operand_error for a dtype (engine/math/random/dispatch.hpp:137) and SB_THROW_IF for HALF (:207). A grep for both macros finds 238 sites in 57 files.

Proposal: The checks already exist as bricks and are simply not used: Dispatch::require_dtype (operand_check.hpp:77-83) for every 'Data type mismatch' line, Dispatch::require_same_element_count (:130-136) for every level 1 length line, Dispatch::require_matrix_extent (matrix_operand.hpp:153-159) for the level 2 square and vector-length lines, and throw_operand_error for rank. Replace each SB_THROW_IF pre-check with its named sibling; the message then carries operation, operand and rule, which the current 'Data type mismatch in SWAP.' lines do not. Sibling to copy: src/engine/blas/level3/trsm.cpp:52-67, which already validates through the matrix_operand functions. Delete the now-redundant rank SB_THROW_IF lines above describe_matrix_operand. Which type the library standardises on is the owner's decision (see decisions). Every test that asserts std::runtime_error for an operand error changes with it; those sites were not counted.

#### A reduction of a zero-element tensor without storage throws from data(), with a null dereference one line behind it

Size: S, 1 file (reductions/dispatch.hpp) plus tests and two doc rows

Evidence: Doc accurate (KNOWN_ISSUES.md:14, REMAINING_WORK.md:79-80). require_storage only objects when num_elements > 0 (operand_check.hpp:65-70), so it passes. execute_reduction then calls t.data() at include/SushiBLAS/engine/math/reductions/dispatch.hpp:110, which reaches SB_THROW_IF(storage == nullptr, ...) at include/SushiBLAS/tensor.hpp:448: std::runtime_error. Same at dispatch.hpp:192 (index), :245 (counting) and :405 (span/axis). If that throw were removed, dispatch.hpp:119 and :254 would dereference t.storage->get_device_index() on a null storage. var, std, logsumexp, argmax and argmin never get there: require_min_elements(t, 1, ...) throws std::invalid_argument first (var.cpp:38, std.cpp:38, logsumexp.cpp:39, argmax.cpp:38, argmin.cpp:39). The elementwise executors return before touching data (engine/math/elementwise/dispatch.hpp:73-74, :135-136, :205-206). The dispatchers below the executor already know what an empty input means: max.cpp:114-121 writes lowest(), and reduce_indexed with n = 0 takes the sequential branch and writes finalize(identity, 0) (reduce_1d.hpp:638-650). Whether an empty tensor that does have a zero-byte Storage gets a non-null allocation_base (tensor.hpp:449) was not checked.

Proposal: Make the four reduction executors read operands the way LogicOps::all does: data_or_null() and regions pushed only for non-null pointers (sibling: src/engine/logic/all.cpp:83-89, whose dispatcher writes the identity for size 0 at :50-57). Take the scratch through a helper that returns an empty ScratchBuffer when the operand has no storage (sibling: Internal::allocate_level1_scratch, engine/blas/utils.hpp:132-133). The result then follows each accumulator's identity: sum 0, prod 1, max lowest(), count 0, and mean 0/0. Whether 'identity' or 'throw' is the contract is a real choice and belongs to the owner; a no-op as in the elementwise family would leave the result tensor unwritten, which is a different thing for an operation whose whole output is that scalar.

#### ElementScalar travels into the device kernel whole, with its sycl::half and double members

Size: M, about 7 files (elementwise/dispatch.hpp, nonlinear/dispatch.hpp, clamp.cpp, pow.cpp, elu.cpp, leaky_relu.cpp, math/README.md)

Evidence: Not recorded in REMAINING_WORK.md or KNOWN_ISSUES.md. ElementScalar holds five members: sycl::half, float, double, int32_t, int64_t (include/SushiBLAS/engine/math/elementwise/scalar.hpp:112-125). The operations capture the whole object by value in the functor that becomes the kernel body: clamp.cpp:44 `[lo, hi](auto x)`, pow.cpp:40 `[e]`, elu.cpp:41 and :49 `[a]`, leaky_relu.cpp:39 and :47 (same class, lambdas not opened). That functor is handed unchanged to Launch::map inside the command group (engine/math/elementwise/dispatch.hpp:101). as<T>() picks one member at compile time (scalar.hpp:66-83), so the README's statement that 'a HALF or FLOAT32 kernel does no double arithmetic' (include/SushiBLAS/engine/math/README.md:197-198) is true of the arithmetic, but the kernel closure still contains a double and a half field in every instantiation. Not verified: whether the pinned intel-llvm marks such a kernel as using aspect::fp64 and aspect::fp16 and refuses it on a device that lacks one. DPC++'s aspect propagation works on types, composite types included, so the risk is plausible; it needs a build for, or a run on, a device without fp64 to settle. The library already treats missing fp64 as real: Device::Profile::supports_fp64 (support/device_policy.hpp:79, :110) and poisson's record-time rejection (docs/reference/CHANGELOG.md:12).

Proposal: Keep ElementScalar on the host and let only the one narrowed T cross into the kernel. The executors already know T on the host inside the with_element_type arm (elementwise/dispatch.hpp:96-102), so bind there: accept either a plain functor or a 'typed' one that yields its device functor for T, resolved by an overload pair. Sibling shape: Detail::epilogue_functor for a bare epilogue versus a GemmEpilogue bundle (include/SushiBLAS/engine/blas/dispatch.hpp:148-163), and the host-side `static_cast<T>(alpha)` before the kernel in execute_gemm (:288-289). ElementScalar itself keeps its one job (narrow once); the five call sites change from capturing the object to supplying a factory. Correct the README sentence to say what crosses to the device.

#### wrapping_mul is undefined for a 16-bit T, but no 16-bit dtype exists

Size: S, 1 file

Evidence: Not recorded in either document. include/SushiBLAS/engine/math/elementwise/wrapping_math.hpp:78-89 computes `static_cast<U>(a) * static_cast<U>(b)` with U = make_unsigned_t<T>. For T = int16_t both operands promote to int and 65535 * 65535 overflows int: undefined behaviour. Core::DataType has no 16-bit or 8-bit integer (include/SushiBLAS/core/common.hpp:49-58: HALF, FLOAT32, FLOAT64, COMPLEX32, COMPLEX64, INT32, INT64), and the typed dispatch only produces int32_t and int64_t (engine/task.hpp:200-207). With U = uint32_t or uint64_t there is no promotion to int and the product wraps as intended. Callers: src/engine/math/elementwise/mul.cpp:39 and elementwise/dispatch.hpp:273. So the defect is latent in the template, not live.

Proposal: One line in wrapping_mul: multiply in `std::common_type_t<U, unsigned>` and cast back to T, which is a no-op for the two live types and defined for any narrower one. Apply the same line to wrapping_add, wrapping_sub and wrapping_neg (wrapping_math.hpp:42-107) so the four siblings stay shaped the same, even though their promoted results cannot overflow int. The alternative of a static_assert(sizeof(T) >= sizeof(int)) is smaller but turns a future INT16 dtype into a compile error in a header that has no reason to know the dtype list.

#### Pointers carried by a GEMM epilogue are not checked by the FORCE_REFERENCE guard

Size: S, 2 files (engine/blas/dispatch.hpp, engine/task.hpp) plus a test and the README

Evidence: Not in KNOWN_ISSUES.md or REMAINING_WORK.md; include/SushiBLAS/engine/blas/README.md:201-204 states it as a duty of the caller. execute_gemm passes `Internal::ReferenceOperands{read_A, read_B, rw_C}` (include/SushiBLAS/engine/blas/dispatch.hpp:279) to require_host_reachable (engine/task.hpp:104-128), which throws std::invalid_argument for device-only USM. The epilogue then runs on the host inside Kernels::Reference::gemm (dispatch.hpp:294-296). GemmEpilogue records only ResourceRegions for what it reads and writes (dispatch.hpp:76-92, :126-127), never the pointers, so nothing reaches the guard. ChannelBiasEpilogue dereferences its raw `bias` pointer per element (include/SushiBLAS/kernels/blas/level3/channel_bias_epilogue.hpp:44, :59). A device-only bias under FORCE_REFERENCE therefore becomes a host dereference of device memory instead of the named exception the three main operands get. In-tree, only plain gemm with GemmIdentityEpilogue calls execute_gemm (src/engine/blas/level3/gemm.cpp:43-46); the exposed callers are outside this repository.

Proposal: Have GemmEpilogue::reads and ::writes also keep operand.data_or_null(), and add a Detail::append_epilogue_host_operands overload pair, empty for a bare epilogue and appending for a bundle, exactly as Detail::append_epilogue_access is shaped (dispatch.hpp:169-193). ReferenceOperands needs a way to grow beyond its initializer list (task.hpp:83-95): an `append(const void*)` member, or a second constructor taking a vector. This changes GemmEpilogue and ReferenceOperands, both in installed headers, by addition only. A bare functor that hides a pointer stays unguarded; that limit should move from the README warning into the @warning of execute_gemm and into KNOWN_ISSUES until the fix lands.

#### ExecutionMode::VERIFY is a public enumerator that only throws

Size: S for removing VERIFY (about 6 files); M for the shared FORCE_REFERENCE refusal across five families (about 8 dispatch headers and .cpp files plus tests) (breaking)

Evidence: Doc accurate (KNOWN_ISSUES.md:11, REMAINING_WORK.md:85-86). The enumerator is declared at include/SushiBLAS/engine.hpp:58 with the comment that set_mode refuses it; Engine::set_mode throws std::logic_error at src/engine.cpp:71-74, so mode_ can never hold it. The fallback branch in dispatch_with_mode (engine/task.hpp:73-76) is therefore unreachable through the public interface. Other references: benchmarks/bench_common.hpp:101-102 (name table), tests/functional/unit/test_execution_mode.cpp:64 (asserts the throw), docs/architecture/ARCHITECTURE.md:79 and :113. The first half of the same row is also accurate: get_mode, execute_blas_task, dispatch_with_mode and KernelPairDispatcher appear only under src/engine/blas, src/engine/spatial, engine/task.hpp, engine/blas/utils.hpp and engine/blas/dispatch.hpp; no file under src/engine/math or src/engine/logic reads the mode. The single-kernel overload of execute_typed_task hard-codes KernelPairDispatcher(ExecutionMode::AUTO) (task.hpp:260), so FORCE_REFERENCE is silently ignored by elementwise, nonlinear, logic, reduction and random operations.

Proposal: For an interface that should not break again: remove the enumerator before the tag. Adding a mode later is compatible; removing one later is a break. The edit is engine.hpp:58 and the @throws at :83, engine.cpp:71-74, the test at test_execution_mode.cpp:64, the name table at bench_common.hpp:101-102 and two architecture lines. Keep the logic_error fallback in dispatch_with_mode for an out-of-range cast. Separately, the five families that ignore the mode need the owner's C2 decision: until they have reference kernels, the honest shape is for the single-kernel execute_typed_task path and the elementwise, logic, reduction and random executors to throw under FORCE_REFERENCE through one shared check, shaped like require_host_reachable (task.hpp:104-128), instead of running the SYCL kernel under a mode that promises the reference one.

### Decisions raised

- Which exception type do record-time operand violations throw across the library?
  - Options: std::invalid_argument everywhere, by routing every pre-check through operand_check.hpp; SB_THROW_IF stays for genuine runtime conditions (allocation, IO, overflow) | std::runtime_error everywhere, by changing throw_operand_error | A SushiBLAS exception hierarchy with one base, so callers can catch the library's errors in one clause | Leave the split and document it per function
  - Recommended: std::invalid_argument everywhere through the existing operand_check functions. The bricks exist (require_dtype, require_same_element_count, require_matrix_extent), a bad operand is a logic error by the standard's own taxonomy, and no new public type is introduced. Settle it before v0.1.0, because the thrown type is part of the interface.

- Is the unification done for BLAS only, as the backlog words it, or for every family?
  - Options: BLAS only (19 .cpp files plus two headers) | Every family in one pass (about 40 files)
  - Recommended: Every family. The same dtype-mismatch line throws runtime_error in elementwise, logic, nonlinear and reductions while its neighbours throw invalid_argument; fixing BLAS alone leaves siblings shaped differently.

- What does a reduction of a zero-element tensor return?
  - Options: The accumulator's identity, written to the result (sum 0, prod 1, max lowest(), count 0; mean becomes NaN) | Throw std::invalid_argument through require_min_elements, as var, std, logsumexp, argmax and argmin already do | No-op that leaves the result tensor unwritten, as the elementwise family does
  - Recommended: Identity for sum, prod, count, asum and nrm2, and require_min_elements for mean, max and min, whose empty value has no natural meaning. This matches what all/any, the dispatchers and five reductions already do. Note that max and min currently write lowest() for an empty tensor that has storage (max.cpp:114-121), so choosing 'throw' for them changes that.

- Does ExecutionMode::VERIFY stay in the public enum for v0.1.0?
  - Options: Remove it now; add it back when it is implemented | Keep it and keep throwing std::logic_error | Implement it for BLAS and spatial before the tag
  - Recommended: Remove it now. Adding an enumerator later is compatible, removing one is a break.

- What do the five families without reference kernels do under FORCE_REFERENCE (backlog item C2)?
  - Options: Throw at record time through one shared check until they have reference kernels | Keep running the SYCL kernel silently, as today | Write reference kernels for all five before v0.1.0
  - Recommended: Throw at record time. It is small, honest about what runs, and can be relaxed family by family later without a break.

- After max_along moves to the NaN-propagating accumulator, Reduce::RunningMax has no caller. Delete it?
  - Options: Delete it | Keep it as the NaN-ignoring alternative
  - Recommended: Delete it; it lives in an installed header (include/SushiBLAS/support/reduce_1d.hpp:390), so this needs your approval as a removal.

- Is min_along added for v0.1.0? The documents mention it but it does not exist.
  - Options: Add it beside max_along | Leave it out and correct the documents
  - Recommended: Your call on interface scope. It is one file shaped like max_along.cpp and adding it later is compatible, so leaving it out costs nothing.

- Do the parallel var, std, logsumexp, argmax and argmin land before the tag?
  - Options: Before v0.1.0 | After v0.1.0
  - Recommended: After. No signature changes, so it cannot force a later break; argmax alone is cheap because RunningArgMax already exists.

- The ElementScalar aspect risk is unverified. Fix on reasoning, or verify first?
  - Options: Fix now: keep the scalar on the host and pass only T into the kernel | First build for, or run on, a device without fp64 and fix only if it fails
  - Recommended: Fix now. The change is internal, and the GPU validation backlog shows no such device has been tried, so waiting for evidence means shipping the risk.

## scout/release (Scout: release)

Read-only survey of v0.1.0 readiness for D:\Projects\sushiblas. Nothing was written to the repository, nothing was built and `sb` was not run. Items 6 and 7 are vacuous: I wrote no source, so there is nothing to syntax-check; the only scratch was one capture file (`sc.txt`) in the session scratchpad, which I deleted. The scratchpad holds 261K, all of it files that were there before this task. Skills loaded: versioning-and-release, repository-layout, humanizer. documentation and project-tools were not loaded; I compared against their checkers' output instead.

VERSION. Four copies say 0.1.0 and agree: CMakeLists.txt:21 `project(SushiBLAS VERSION 0.1.0)`, include/SushiBLAS/version.hpp:34-41 (hand-written, checked against CMake at CMakeLists.txt:78-87, so it is "checked", not "generated" as the skill words it), cli/pyproject.toml:7, docs/reference/CHANGELOG.md:3. One disagrees: Doxyfile:22 `PROJECT_NUMBER = "v1 (unreleased)"`. Nothing checks cli/pyproject.toml against CMake. sushi-module.toml has no version field. `git tag` prints nothing; main is 16 commits ahead of origin/main.

CHANGELOG. docs/reference/CHANGELOG.md has one `## Unreleased` section with 128 entries (lines 9-136); check_changelog.py passes. No `## v0.1.0 — date` heading exists. No entry marks a break, although 5 commit subjects carry `!` and several entries are breaks (CHANGELOG.md:26 removed the `sb` namespace alias, :24 and :40 change every seeded stream, :22 scalars became double). docs/archive/CHANGELOG.md holds the frozen pre-reshape text; docs/archive/changelog/ does not exist and is not needed for a first release.

INSTALL/EXPORT. The package shape is complete: src/CMakeLists.txt:226-249 (install TARGETS, DIRECTORY, EXPORT, build-tree export), CMakeLists.txt:103-122 (config + version file, SameMinorVersion), cmake/SushiBLASConfig.cmake.in:25-29 (find_dependency(SushiRuntime) then targets). tests/package/CMakeLists.txt:53 is a consumer and `sb test --suite package` drives it (cli/sushiblas/services/project.py:284-355). I did not run it, so whether find_package works today is unverified. Three gaps: find_dependency(SushiRuntime) names no version; install(DIRECTORY include/) has no FILES_MATCHING, so ten README.md files and the experimental/ headers are installed; the package suite is in no CI job (ci.yml:173-177).

CI. One workflow, .github/workflows/ci.yml. SushiRuntime is checked out unpinned from the default branch (ci.yml:74-89). The local sibling is at v0.3.0-60-g6ab3303, on main and level with origin/main, so the allocation interface the library needs is on the runtime's main now, but no runtime tag carries it (latest tag v0.3.0). The `checkers` job (ci.yml:179-198) runs check_source_comments.py with all rules, which exits 1 today, so that job is red on both OS; it does not run check_docs_layout.py. pyflakes still ends in `|| true` (ci.yml:140). Stale text: "no Doxyfile" (ci.yml:168-171), "356-test figure" (ci.yml:108) against 792 in REMAINING_WORK.md:9, `docs/CLAUDE.md` (ci.yml:24). No Windows build job, no release workflow. The file header says the workflow had never run when written (ci.yml:9-19).

LAYOUT. Against repository-layout for a one-module repository: `src/` should be `source/`; `kernels/`, `examples/`, `benchmarks/` are top-level folders the table does not name; `Doxyfile` is forbidden in the root; `CMakePresets.json` and `sushi-module.toml` are not in the root list; tests have a `functional/` level and a `package/` folder; `.clang-format`, `.editorconfig`, `.gitattributes` are absent. SushiRuntime has the same `src/` and root `Doxyfile`, so this is a stack-wide decision. CMakePresets.json still describes icx-cl and oneAPI, which the README says is no longer the toolchain.

CHECKERS (run from D:\Projects\sushiblas):
- `python tools/documentation/check_docs_layout.py .` exit 1, three findings: `docs/api:1: rule_docs_entries`, `docs/api-site:1: rule_docs_entries`, `docs/api/mainpage.md:1: rule_document_names`.
- `python tools/documentation/check_changelog.py .` exit 0, no output.
- `python tools/documentation/check_source_comments.py . --rule rule_license_block` exit 0, no output.
- `python tools/layering/check_layering.py .` exit 0, 7 `note:` lines, all `rule_engine_header_includes: known` for include/SushiBLAS/engine.hpp:33,34,39,40,41,43,44.
- `python -m unittest discover -s tools/tests`: `Ran 66 tests in 1.210s OK`.
- `python tools/documentation/check_source_comments.py . --report`: rule_license_block 0, rule_file_header 0, rule_block_ceiling 18, rule_comment_runs 0, rule_separators 0, rule_history 0, rule_no_date 0, total 18. Without --report it exits 1. All 18 are in public headers (engine/blas/dispatch.hpp:196, engine/blas/utils.hpp:143, engine/math/elementwise.hpp:318, engine/math/elementwise/dispatch.hpp:52,108, engine/task.hpp:240,264, io.hpp:123,188,200, support/reduce_1d.hpp:617,705, tensor.hpp:349 and five more).

LICENCE/NOTICE. LICENSE is PolyForm Noncommercial 1.0.0 with a Required Notice at LICENSE:75; COMMERCIAL.md exists; cli/pyproject.toml:9 declares the licence; the license-block rule passes on every file. NOTICE.md gaps: licences marked "unverified" for Random123/Philox (NOTICE.md:35), Contributor Covenant (:44), DPC++, oneTBB, GoogleTest, hwloc, typer, rich, tomli, setuptools, pytest/pyflakes, the container image (:55-65); portBLAS version "unverified" (:10); the Poisson PTRS algorithm (kernels/sycl/random/poisson.hpp:24,117, Hoermann) is not listed under published references; stale path `docs/CODE_OF_CONDUCT.md` (:44); NOTICE.md:28 embeds an absolute path on this machine and :65 names a container image the workflow no longer uses (setup-oneapi/action.yml installs from apt). third_party/licenses/ holds Apache-2.0.txt only.

README. Matches the code where I checked: eight accessors (engine.hpp:103-145), VERIFY refused (engine.hpp:58,83), the header tree under include/SushiBLAS, `sr` alias (sushiruntime core/common.hpp:68), `gemm(A,B,C)` (level3.hpp:51), `sb build --benchmarks`, `sb test --suite package`, the layered runtime resolution (cmake/Runtime.cmake:22-41). One claim is wrong against KNOWN_ISSUES.md:11: README.md:9 says set_mode selects the mode "for every later operation", while only BLAS and spatial read it. README.md:14-18 does not say which SushiRuntime version is required.

STALE RECORDS. KNOWN_ISSUES.md:31 says 1 051 comment findings (now 18); :67 says 88 Unreleased entries (now 128); :68 says sushicore is tagged up to v0.4.0, while `pip index versions sushicore` lists 0.8.0 and 0.7.0 on PyPI; :56 names the runtime's ci-fix branch, which is merged; :57 and :60 describe a cli job and a missing checker job that ci.yml no longer has. AGENTS.md says the other comment rules "do not pass yet"; six of seven do.

Kernel fusion (the owner's side question, outside this area): the only fusion in this repository that I saw is the GEMM epilogue seam, a per-element functor applied before the store (NOTICE.md:114-119, CHANGELOG.md:68,77,82). I did not survey the engine or SushiRuntime for graph-level fusion.

RELEASE CHECKLIST, in order:
1. Breaking interface work the owner asked for lands first. NOT DONE (REMAINING_WORK.md:74-93 lists the open public-interface items: bool flags to enums, one exception type, execution-mode contract, engine.hpp includes).
2. Mark breaking entries in the changelog. NOT DONE (0 marked entries; 5 `!` commits).
3. All checkers clean. NOT DONE: check_docs_layout exit 1 (3 findings), check_source_comments 18 findings. DONE: check_changelog, license rule, check_layering (7 known), 66 tool tests.
4. Full test suite through `sb`, including `--suite package`. NOT VERIFIED here; REMAINING_WORK.md:9 records 792 of 792 on the OpenCL CPU device, no GPU run.
5. CI green. NOT DONE: checkers job fails on current tree; functional job has never been confirmed to run; main is 16 commits unpushed.
6. Pin the SushiRuntime the release is built against. NOT DONE (ci.yml:79 unpinned; NOTICE.md:54 "not pinned"; no runtime tag at or after 6ab3303; find_dependency has no version).
7. One version source. PARTLY DONE: CMake, version.hpp, pyproject agree at 0.1.0; Doxyfile:22 disagrees; no check ties pyproject to CMake.
8. NOTICE complete. NOT DONE (unverified licences, PTRS reference, stale path).
9. README and manual accurate. PARTLY DONE (README.md:9 mode claim; KNOWN_ISSUES and AGENTS.md stale counts; CMakePresets.json stale).
10. Layout decisions (src/source, benchmarks, kernels, examples, Doxyfile). NOT DONE, owner decisions (REMAINING_WORK.md:98,125-134).
11. Rename `## Unreleased` to `## v0.1.0 — date`, open an empty Unreleased. NOT DONE.
12. Commit `chore(release): v0.1.0`, annotated tag. NOT DONE (no tags).
13. Push, on the owner's word. NOT DONE.

### Findings

#### CI checkers job fails on the current tree

Size: M, about 12 headers plus Doxyfile, .gitignore, ci.yml, AGENTS.md

Evidence: .github/workflows/ci.yml:191-192 runs `python tools/documentation/check_source_comments.py .` with all rules; the same command exits 1 here with 18 rule_block_ceiling findings (--report: total 18, the other six rules 0). The job also omits check_docs_layout.py, which exits 1 with 3 findings (docs/api, docs/api-site, docs/api/mainpage.md).

Proposal: Shorten the 18 comment blocks (all in public headers: task.hpp:240,264, io.hpp:123,188,200, tensor.hpp:349, reduce_1d.hpp:617,705, elementwise/dispatch.hpp:52,108 and eight more), move the Doxygen main page and site out of docs/ as REMAINING_WORK.md:125-127 already plans, then add check_docs_layout.py to the job. Update AGENTS.md, which still says the other rules do not pass.

#### SushiRuntime is not pinned anywhere a release can cite

Size: S, 5 files here plus a tag in sushiruntime

Evidence: ci.yml:74-81 checks out the runtime's default branch unpinned. cmake/SushiBLASConfig.cmake.in:26 calls find_dependency(SushiRuntime) with no version. NOTICE.md:54 says 'not pinned'. The sibling checkout is at v0.3.0-60-g6ab3303 (main, level with origin/main); the latest runtime tag is v0.3.0, 60 commits behind the allocation interface this library uses (CHANGELOG.md:31). The runtime's package uses SameMinorVersion (sushiruntime/CMakeLists.txt:110).

Proposal: Tag the runtime first (0.4.0 by its own rules), then state that version in find_dependency(SushiRuntime 0.4), in cmake/Runtime.cmake's find_package, in README Requirements and NOTICE.md, and pin the CI checkout to the tag for the release job while a second lane keeps following main.

#### Changelog has no release section and marks no breaking change

Size: S, 1 file

Evidence: docs/reference/CHANGELOG.md:7 is the only section, 128 entries (lines 9-136). No entry carries a break marker; `git log` has 5 subjects with `!`. Entries that break a consumer include :26 (global alias `sb` removed), :24 and :40 (every seeded random stream changes), :22 (scalar parameters taken as double), :129 (LinalgOps/TransformsOps removed from Engine). `git tag` is empty.

Proposal: Before the cut, mark the breaking entries as the documentation skill's entry shape requires, add the entries the coming interface changes produce, then rename the section to `## v0.1.0 — <date>` with an empty Unreleased above it. No archive move is needed for a first release.

#### Version copies: Doxyfile disagrees and the CLI version is unchecked

Size: S, 3 files

Evidence: Doxyfile:22 `PROJECT_NUMBER = "v1 (unreleased)"`. CMakeLists.txt:21, version.hpp:41 and cli/pyproject.toml:7 say 0.1.0; CMakeLists.txt:78-87 checks version.hpp only. KNOWN_ISSUES.md:44,69 and REMAINING_WORK.md:128,153 already record both.

Proposal: Feed PROJECT_NUMBER from the CMake version (the docs command can pass it, or the Doxyfile is configured), and add a checker under tools/ that compares cli/pyproject.toml with project(). Whether the CLI package must share the library's version is an owner decision.

#### Install rule ships READMEs and the experimental headers

Size: S, 1 file (breaking)

Evidence: src/CMakeLists.txt:234-236 installs include/ with no FILES_MATCHING. `git ls-files include` lists ten README.md files, and include/SushiBLAS/experimental/ (lapack, signal) holds quarantined declarations (README.md:184, NOTICE.md:151-153) that would become part of the installed 0.1.0 surface.

Proposal: Restrict the install to *.hpp and *.h, and decide whether experimental/ is installed at all; leaving it out keeps it from being a public promise.

#### Package consumer test is outside CI and was not run in this survey

Size: S, 1 file

Evidence: ci.yml:173-177 states tests/package/ is not wired in. cli/sushiblas/services/project.py:284-355 implements `sb test --suite package`. KNOWN_ISSUES.md:24 records a traceback in that suite when a DLL is locked.

Proposal: Run `sb test --suite package` as a release gate and add it to CI once the functional job is confirmed; it is the only evidence that find_package(SushiBLAS) works.

#### NOTICE.md is incomplete for a release

Size: S, 1-2 files

Evidence: NOTICE.md:10 portBLAS version 'unverified'; :35 Random123 licence 'unverified' and :37-38 provenance not recorded; :44 stale path docs/CODE_OF_CONDUCT.md and licence 'unverified'; :55-65 ten components 'unverified'; :65 lists a container image that .github/actions/setup-oneapi/action.yml no longer uses (apt install); :28 embeds D:/Projects absolute paths. kernels/sycl/random/poisson.hpp:24,117 implements Hoermann's PTRS and is absent from the 'published reference' table. third_party/licenses/ has Apache-2.0.txt only.

Proposal: Read each licence from its source and fill the column, add the PTRS reference, fix the path, drop the container row, make the command repository-relative. Build-time tools that are not redistributed can be moved to a shorter table if the dependencies skill allows it.

#### README and records state things the code no longer does

Size: S, 5 files

Evidence: README.md:9 says set_mode selects the mode 'for every later operation'; KNOWN_ISSUES.md:11 says only BLAS and spatial read it. KNOWN_ISSUES.md:31 says 1 051 comment findings (now 18), :67 says 88 entries (now 128), :68 says sushicore is tagged to v0.4.0 (PyPI lists 0.8.0 and 0.7.0), :56-57,60 describe CI as it was. ci.yml:108 says 356 tests, REMAINING_WORK.md:9 says 792. ci.yml:168-171 says no Doxyfile exists. CMakePresets.json names icx-cl and oneAPI, README.md:14 names intel-llvm clang++.

Proposal: Correct README.md:9 or make the mode a real library-wide contract (REMAINING_WORK.md:85-86, owner decision). Refresh KNOWN_ISSUES, the ci.yml comments and AGENTS.md in the release-preparation commit. Remove or rewrite CMakePresets.json.

#### Layout departs from repository-layout in the root and in tests

Size: L, 300+ tracked paths moved, about 15 files edited

Evidence: Root holds src/ (skill: source/), kernels/, examples/, benchmarks/ (none in the skill's root table; CMakeLists.txt:90-96), Doxyfile (forbidden), CMakePresets.json and sushi-module.toml (not listed). tests/ has functional/{unit,integration,regression} and package/ where the skill names tests/{unit,integration,regression,common}. .clang-format, .editorconfig, .gitattributes are absent. SushiRuntime's root has the same src/ and Doxyfile. KNOWN_ISSUES.md:43-48 and REMAINING_WORK.md:98,125-134 record all of it.

Proposal: Decide once for the stack. src/ to source/ touches CMake paths, 62 layering table entries, READMEs and the docs; benchmarks could go under tests/performance or applications/. Paths are not public interface, so this can follow v0.1.0, but doing it before the tag avoids a second churn of every file path cited in the changelog.

#### Local main is unpushed and merged branches remain

Size: S, 0 files

Evidence: `git status -sb`: `## main...origin/main [ahead 16]`. Local branches ci-fix, feature/blas-spatial-domain, feature/sushicore_help_screen exist (REMAINING_WORK.md:137-138).

Proposal: Push main before tagging so CI runs on the commit that gets the tag; delete the merged branches on the owner's word.

### Decisions raised

- Which SushiRuntime version does SushiBLAS v0.1.0 require, and is the runtime tagged first?
  - Options: Tag SushiRuntime v0.4.0 at or after 6ab3303 and require 0.4 in find_dependency, README and CI | Release against an untagged runtime commit and record the SHA in NOTICE.md | Leave it unpinned
  - Recommended: Tag SushiRuntime v0.4.0 first and require it; a release that names no runtime version cannot be reproduced.

- Do the layout moves (src/ to source/, homes for kernels/, examples/, benchmarks/, Doxyfile, tests/functional/) happen before v0.1.0?
  - Options: Before the tag, together with SushiRuntime | After the tag, as a refactor that needs no release | Keep the current layout and amend the repository-layout skill for the stack
  - Recommended: Decide together with SushiRuntime; if the answer is to move, do it before the tag, since it is not a public break but rewrites every cited path.

- Is include/SushiBLAS/experimental/ part of the installed v0.1.0 package?
  - Options: Not installed until a backend is chosen | Installed, documented as unstable
  - Recommended: Not installed; an installed header is a public promise and these are declarations without a backend.

- Does the CLI package version follow the library version?
  - Options: One version, checked by a tool against project() | Independent versions, stated in cli/README.md
  - Recommended: One version with a checker, which is what the versioning skill's single source of truth asks for.

- Is the execution mode a library-wide contract in v0.1.0?
  - Options: Yes: the five families that ignore it get a reference path or throw | No: README.md:9 is corrected to say BLAS and spatial only
  - Recommended: Settle it before the tag, because either answer changes observable behaviour or the documented interface.

- Is a green CI run on the tagged commit a condition of the release?
  - Options: Yes, including a first confirmed run of the functional job and the package suite | Checkers and local `sb test` only for v0.1.0
  - Recommended: Yes; the workflow header says it had never run when written, and the checkers job fails today.

- Approve v0.1.0 as the first tag once the checklist is done, with the 128 Unreleased entries cut under it?
  - Options: Approve v0.1.0 | Start with v0.1.0-rc.1 | Wait
  - Recommended: v0.1.0-rc.1 first if the interface changes land in the same window, then v0.1.0 after CI and the package suite pass.

## PKG_DEVICE_GATE (Package: the sb device gate)

Written on 2026-10-07 for finding 10 of `scout/gpu-perf.md`, part (1) only.

### What was added

- `cli/sushiblas/services/device.py`: `K_DEVICE_SELECTOR`, `device_selector(device)` and
  `with_device(env, device)`. `with_device` copies the environment, sets
  `ONEAPI_DEVICE_SELECTOR` in the copy and returns the value it replaced.
- `cli/sushiblas/services/errors.py`: `InvalidDeviceError`, exit code 2.
- `cli/sushiblas/services/project.py`: `ProjectService.test` and `ProjectService.run` take
  `device`. Both validate it before they read the configuration or look for the build tree.
  `_device_env` prints the override line.
- `cli/sushiblas/cli.py`: `--device` on `sb test` and `sb run`.
- `cli/tests/test_device.py` (19 cases) and 15 cases in `cli/tests/test_project.py`.
- `cli/README.md`, `cli/sushiblas/README.md`, `cli/tests/README.md`.

| `--device` | `ONEAPI_DEVICE_SELECTOR` | Source |
| --- | --- | --- |
| `cpu` | `opencl:cpu` | `sushiruntime/cli/sushiruntime/services/project.py:503` |
| `cuda` | `cuda:*` | The dispatch for this package |
| `selector:<text>` | `<text>` unchanged | The dispatch for this package |

Precedence: without `--device` the child gets the caller's environment as it is. With
`--device` the option wins, and when the caller had a selector `sb` prints
`--device overrides ONEAPI_DEVICE_SELECTOR=<old> from the environment with <new>.` once.

### What was left out

- `gpu` is not a value. SushiRuntime's CLI sets `opencl:cpu` in one place and names no GPU
  selector. sushiengine's CLI has no `ONEAPI_DEVICE_SELECTOR` and no `--device` at all.
  `--device gpu` exits 2; `--device "selector:level_zero:gpu"` is the way to say it.
- There is no `sb bench` command. A benchmark runs through `sb run bench_gemm`, so `sb run`
  is the command that carries the option for benchmarks.
- No CTest `device` label, no profile report, no `--json` on the benchmarks, and the target
  name at `cli/README.md:107` is unchanged. Those are parts (2) to (6) of the finding.
- `docs/reference/CHANGELOG.md` and `docs/design/REMAINING_WORK.md` are not in this
  package's file list; the orchestrator writes them.

### Things the orchestrator should know

- `--device` also applies to `sb test --suite package`. The selector is set on the one
  environment that suite uses for install, configure, compile and the consumer run.
- `sushicore.build_env` strips `ONEAPI_DEVICE_SELECTOR` from the cached toolchain snapshot and
  merges the snapshot over `os.environ`, so the caller's own value reaches `with_device`. I read
  that code; I did not run `sb test` or `sb run` against a real build.
- The existing stubs `_Driver` and `_BuildEnv` in `cli/tests/test_project.py` gained a
  `ctest_run` method and a settable `variables` attribute.
- `check_docs_layout.py` reports this file's name, since a work folder holds `SPEC.md`,
  `PLAN.md` and `REPORT.md` only. The dispatch named the file.

### Verification

Item 6 of the dispatch is replaced by the unit tests for Python. No C++ or GLSL was written.

```
$ cd cli && python -m pytest tests -q
........................................................................ [ 66%]
.....................................                                    [100%]
109 passed in 0.70s
```

The suite had 75 tests before the change.

```
$ cd cli && python -m pytest tests/test_device.py tests/test_project.py -q -k device
..................................                                       [100%]
34 passed, 23 deselected in 0.31s
```

```
$ python tools/documentation/check_source_comments.py cli/sushiblas/services/device.py cli/tests/test_device.py cli/sushiblas/services/errors.py
(no output, exit 0)
$ python tools/documentation/check_source_comments.py cli/sushiblas/services/project.py cli/sushiblas/cli.py cli/tests/test_project.py
(no output)
$ python tools/documentation/check_source_comments.py . --rule rule_license_block
(no output, exit 0)
$ python tools/documentation/check_changelog.py .
(no output, exit 0)
$ python tools/layering/check_layering.py .
(notes only, exit 0)
$ python -m unittest discover -s tools/tests
Ran 66 tests in 1.027s
OK
```

Scratchpad: this package wrote nothing to it. The session's shared scratchpad measured 701K
when the package finished, all of it from other agents.

### Review

A second agent read every changed file and re-ran the commands above; the counts matched.
It changed three things:

- `device_selector` stripped the blanks around the text after `selector:`, while its
  docstring and `cli/README.md` both say the text is passed through as written. The text now
  goes through untouched and only an all-blank text is refused. `test_device.py` gained the
  case `"selector: cuda:0 "`.
- One line of the `K_DEVICE_SELECTOR` section in `cli/sushiblas/README.md` was 130 columns;
  it is rewrapped.
- The same section now says the `selector:` prefix is matched in lower case only, which
  `test_device.py` already pinned (`Selector:x` exits 2) and no document stated.

Left as it is, for the orchestrator to weigh:

- `ProjectService` resolves the device twice per call: `_check_device` up front, so a bad
  value fails before the configuration is read, and `with_device` again once the environment
  is loaded. Passing the resolved selector to `with_device` would remove the second parse and
  change that function's signature.
- When `--device` names the selector the environment already holds, the override line still
  prints, with the same value on both sides.

```
$ cd cli && python -m pytest tests -q
........................................................................ [ 65%]
......................................                                   [100%]
110 passed in 0.57s
```

## PKG_RELEASE_MECHANICS (Package: release mechanics)

Written by the release-mechanics agent on 2026-10-07. No C++, CMake, `.github/`, `AGENTS.md` or
changelog file was touched, and nothing was built.

### Done

1. `tools/versioning/check_version.py`, in the shape of the other checkers (`common.checker`,
   one `rule_` per row of `K_CHECKER`, `K_` constants, `--report`, `--rule`, no writes).
2. `tools/tests/test_check_version.py`, 24 tests: agreement, a disagreeing value in each of the
   five files, a missing required file, a missing optional file, a file without the value, a
   missing or version-less `CMakeLists.txt`, and the printed line and exit code of a run.
3. `Doxyfile:22`: `PROJECT_NUMBER` changed from `"v1 (unreleased)"` to `"0.1.0"`. It was the
   only disagreeing copy; `cli/pyproject.toml:7` already said `0.1.0` and was not edited.
4. `NOTICE.md`, lines 1 to 89 rewritten; "Port notes" and everything below it is unchanged.

### The source of truth

`project(SushiBLAS VERSION 0.1.0 ...)` at `CMakeLists.txt:21`, as the versioning-and-release
skill's C++ row says. `CMakeLists.txt:78-87` already fails the configure when
`SUSHIBLAS_VERSION_STRING` in `version.hpp` differs, so the header is a checked copy too.

| Rule | File | What must agree | File required |
| --- | --- | --- | --- |
| `rule_source_version` | `CMakeLists.txt` | a `project(<Name> VERSION x.y.z)` exists | yes |
| `rule_version_header` | `include/SushiBLAS/version.hpp` | `SUSHIBLAS_VERSION_STRING`, `_MAJOR`, `_MINOR`, `_PATCH` | yes |
| `rule_doxyfile` | `Doxyfile` | `PROJECT_NUMBER`, with or without a leading `v` | yes |
| `rule_cli_package` | `cli/pyproject.toml` | `version` in `[project]` | yes |
| `rule_module_manifest` | `sushi-module.toml` | `version` in `[module]`, when the key exists | no |
| `rule_readme` | `README.md` | every `SushiBLAS x.y.z` or `SushiBLAS vx.y.z` | no |

A finding reads `Doxyfile:22: rule_doxyfile: PROJECT_NUMBER is 'v1 (unreleased)', CMakeLists.txt:21 expects 0.1.0`.
When `CMakeLists.txt` states no version, only `rule_source_version` reports; the copy rules stay
silent because they have nothing to compare with.

`sushi-module.toml` has no version key today and the README states no SushiBLAS version, so both
rules pass on nothing. The CMake configure check on the header checks the string only; the
checker also checks the three numeric macros.

### NOTICE.md, claim by claim

| Claim | Checked against |
| --- | --- |
| portBLAS lines in two files | `gemm.hpp:21-22` and `syrk.hpp:21-22`; a search of all tracked files for `Copyright` finds no other third-party line outside `LICENSE`, `NOTICE.md`, `docs/` and `third_party/` |
| portBLAS is Apache-2.0 | upstream `LICENSE` on `master`, fetched 2026-10-07; `third_party/licenses/Apache-2.0.txt` exists |
| Random123 is BSD-3-Clause, Copyright 2010-2012 D. E. Shaw Research | upstream `LICENSE` on `main`, fetched 2026-10-07 |
| Philox constants | `kernels/sycl/random/philox.hpp:65-68`; `kernels/README.md:397-401` |
| PTRS and Knuth | `kernels/sycl/random/poisson.hpp:24,95,117`; `kernels/README.md:434-439` |
| Code of conduct path | `docs/reference/CODE_OF_CONDUCT.md:70-76` (the old `docs/CODE_OF_CONDUCT.md` path was stale) |
| SushiRuntime licence | `../sushiruntime/LICENSE`, line 1 |
| GoogleTest, hwloc | `D:/Projects/sushistack/dependencies/vcpkg/installed/x64-windows/share/{gtest,hwloc}/copyright` |
| typer, rich, tomli, setuptools, pytest | `importlib.metadata` of the installed packages |
| apt packages in place of the container row | `.github/actions/setup-oneapi/action.yml:16-17` |

The license-block command is now relative to the repository root, since
`tools/licensing/write_license_block.py` exists here.

### Not done

- The portBLAS commit the port was taken from is still unrecorded. Nothing in the tree states it.
- DPC++, oneTBB and pyflakes stay "unverified": none of their licence files was on this machine
  where I looked. sushicore at 0.7.0 or later was not read; the installed copy is 0.5.0.
- The Contributor Covenant licence stays "unverified". The 2.1 text I fetched names no licence.
- `third_party/licenses/` still holds only `Apache-2.0.txt`. It is outside my file list. A
  BSD-3-Clause text belongs there only if the owner states that `philox.hpp` was adapted from
  Random123 source, and a CC BY 4.0 text only once that licence is confirmed.
- `tools/README.md` has no row for the new checker; the file is outside my list.
- Item 6 of the dispatch block does not apply: I wrote no C++ and no GLSL. The Python was
  checked by its unit tests, below.

### For the orchestrator

Add to the check list in `AGENTS.md`, after the `check_layering.py` line:

```
python tools/versioning/check_version.py .
```

Add to the `checkers` job in `.github/workflows/ci.yml`, after the `check_layering.py` step
(line 196) and before the unittest step:

```
      - name: python tools/versioning/check_version.py .
        run: python tools/versioning/check_version.py .
```

Add to the table in `tools/README.md`, after the `layering/check_layering.py` row:

```
| `versioning/check_version.py` | `version.hpp`, `Doxyfile`, `cli/pyproject.toml`, `sushi-module.toml` and `README.md` against `project()` in the root `CMakeLists.txt` | Repository root |
```

Changelog lines for `## Unreleased`, if wanted:

```
- 2026-10-07 — tools: Added a checker that compares every version copy with project() (`check_version.py`, `test_check_version.py`).
- 2026-10-07 — docs: Set the Doxygen project number to 0.1.0 (`Doxyfile`).
- 2026-10-07 — licence: Listed the shipped third-party works and the licences read for each dependency (`NOTICE.md`).
```

One decision the owner has not made is assumed here: the checker ties the CLI package version to
the library version, which the scout listed as open.

Four backlog lines need editing, in files outside this package's list:

- `docs/reference/KNOWN_ISSUES.md:44` and `docs/design/REMAINING_WORK.md:128`: the version half
  is closed. The Doxyfile still sits in the root, so that half of each line stays.
- `docs/reference/KNOWN_ISSUES.md:69` and `docs/design/REMAINING_WORK.md:153`: closed once the
  checker runs in CI; remove both then.

`check_docs_layout.py` reports this file's name under `rule_work_folders`, as it does for the
other `PKG_` and scout files in this folder.

### Review, 2026-10-07

The reviewer re-ran every command below and got the same results, then changed these:

- `check_version.py`: a `project()` call on a CMake comment line is skipped; a `project()`
  version of four numbers counts as no version, where the first three were taken before; a TOML
  `version` in single quotes is read; a table header that ends in a comment, and an
  array-of-tables header, now close the previous table. Six tests cover these and the command
  line. They were written after the fixes and were not seen failing first.
- `check_version.py`: the three trailing `#` comments on `Copy` moved into its docstring, and
  every line over 100 columns was wrapped, in the test file too.
- `NOTICE.md`: the opening said three third-party works ship. Two do; Random123 is listed
  because the tree does not record where `philox.hpp` came from. The `Copyright` search
  sentence now names the `tools/` fixtures it excludes.
- This record: the backlog note above said four lines need removing; two of them only lose
  their version half.

Left as it is: `collect` drops a path that is not a folder without a finding, as
`check_layering.py` does.

### Verification

```
$ python -m unittest discover -s tools/tests
Ran 90 tests in 1.134s

OK

$ python tools/versioning/check_version.py .
(no output, exit 0)

$ python tools/versioning/check_version.py . --report
rule_source_version: 0
rule_version_header: 0
rule_doxyfile: 0
rule_cli_package: 0
rule_module_manifest: 0
rule_readme: 0
total: 0

$ python tools/documentation/check_source_comments.py tools/versioning tools/tests/test_check_version.py
(no output, exit 0)

$ python tools/documentation/check_changelog.py .
(no output, exit 0)

$ python tools/layering/check_layering.py .
(7 known notes, exit 0)
```

The unittest run prints three lines from the license-block tests (`0 files pending`,
`2 files written`, `graph.py`) before the summary; they were there before this package.

Scratchpad: nothing written by this package; `du -sh` reports 701K, from other sessions' files.
I deleted `tools/versioning/__pycache__`, which my runs created.

## EXCEPTIONS (Exceptions by layer)

Read from the working tree on 2026-10-07: HEAD `82504db` plus the uncommitted change that adds
`UnsupportedExecution`. Nothing here was built or run. Twelve files were syntax-checked with
`clang++ -fsyntax-only`; the list is at the end. Line numbers are those of the working tree.

"Record time" is the call on an operation class or a dispatch executor, before `Engine::execute`.
"Execute time" is inside `Engine::execute`, or inside an `IO` call, which runs the graph first.

### Types

| Type | Base | Declared in | Meaning |
| --- | --- | --- | --- |
| `SushiBLAS::UnsupportedExecution` | `std::invalid_argument` | `include/SushiBLAS/core/errors.hpp` | The operation cannot run under this execution mode or on this device. |
| `std::invalid_argument` | `std::logic_error` | `<stdexcept>` | An operand, flag, scalar or parameter is wrong. |
| `std::runtime_error` | `std::exception` | `<stdexcept>` | A tensor or storage rule, a file, an allocation, or a dtype switch that fell through. |
| `SushiRuntime::Core::SushiException` | `std::runtime_error` | `sushiruntime/include/SushiRuntime/core/error.hpp` | The runtime refused a graph or a run. |

The declaration, complete:

```cpp
namespace SushiBLAS
{
    class UnsupportedExecution final : public std::invalid_argument
    {
        public:
            using std::invalid_argument::invalid_argument;
    };
}
```

A `catch (const std::invalid_argument&)` written before this type existed still catches it. A
caller that wants the two apart puts `catch (const SushiBLAS::UnsupportedExecution&)` first.

### The helpers that throw

All four log at error level and then throw. No other `throw` statement exists under `include/`,
`src/` or `kernels/`, apart from two in `src/io.cpp` (a rethrow at 459 and an unreachable
`throw std::runtime_error` at 171, after an `SB_THROW_IF(true, ...)`).

| Helper | Throws | File | Message |
| --- | --- | --- | --- |
| `Dispatch::Detail::throw_operand_error(op, operand, fmt, args...)` | `std::invalid_argument` | `engine/operand_check.hpp:52` | `<op>: operand '<operand>' <rule>` |
| `Dispatch::Detail::throw_execution_error(op, fmt, args...)` | `UnsupportedExecution` | `engine/operand_check.hpp:67` | `<op>: <refusal>` |
| `Dispatch::Detail::throw_dtype_error(family, name, dtype)` | `std::invalid_argument` | `engine/operand_check.hpp:81` | `<family> '<name>' does not support dtype <dtype>. ...` |
| `SB_THROW_IF(condition, fmt, args...)` | `std::runtime_error` | `core/logger.hpp:432` | the formatted text |

`throw_execution_error` takes no operand argument because the recorder's refusal has none to
name: `OperandSet` knows that one of its tensors is device-only, not which. The two sites that do
know their operand write it into the format, so their text is what it was before.

### Sites that throw `UnsupportedExecution`

Three functions call `throw_execution_error`, the first of them twice. Every other refusal
reaches one of the three.

| Site | When | Text |
| --- | --- | --- |
| `Dispatch::require_device_dtype`, `engine/operand_check.hpp:291` | FLOAT64 or COMPLEX64 storage on a device without `aspect::fp64` | `<op>: operand '<x>' has dtype <D> but lives on a device without the fp64 aspect` |
| `Dispatch::require_device_dtype`, `engine/operand_check.hpp:296` | HALF storage on a device without `aspect::fp16` | `<op>: operand '<x>' has dtype HALF but lives on a device without the fp16 aspect` |
| `TaskRecorder::require_host_reachable`, `engine/task_recorder.hpp:144` | mode is FORCE_REFERENCE and a declared tensor is `Residency::Device` | `<name>: <family> cannot run under FORCE_REFERENCE: an operand is device-only USM, which the host reference kernel cannot dereference` |
| `RandomOps::poisson`, `src/engine/math/random/poisson.cpp:50` | `t` has elements and its device lacks `aspect::fp64`, whatever the dtype | `random.poisson: operand 't' lives on a device without fp64; random.poisson draws in double precision for every dtype` |

The text of the recorder's refusal changed shape: it began `<family> '<name>' cannot run` and now
begins `<name>: <family> cannot run`. The tests that read it look for the operation name and for
`FORCE_REFERENCE`, and both are still there.

Callers of `require_host_reachable`:

- `TaskRecorder::record_pair` (`task_recorder.hpp:163`). It is the only public recording call, so
  every task of every family passes it: BLAS and sparse through `Internal::execute_blas_task`,
  spatial through `Internal::execute_typed_task`, the elementwise, nonlinear and logic executors
  through the map tasks, and the operations that call `record_pair` themselves (`convert`,
  `constant`, `all`, `any`, `orthogonal`, `shuffle`).
- `Dispatch::execute_random` (`math/random/dispatch.hpp:294`), before it reserves a stream offset.
- `Dispatch::execute_reduction`, `execute_index_reduction` and `execute_counting_reduction`
  (`math/reductions/dispatch.hpp:283`, `564`, `620`), before they lease scratch.
- `RandomOps::orthogonal` (`orthogonal.cpp:62`) and `RandomOps::shuffle` (`shuffle.cpp:123`).

An epilogue operand declared through `Dispatch::GemmEpilogue::reads` or `writes` is merged into
the task's `OperandSet` (`engine/blas/dispatch.hpp:249`), so `record_pair` refuses it the same
way. `Internal::allocate_blas_scratch` (`engine/blas/utils.hpp:119`) does not refuse: under
FORCE_REFERENCE it returns an empty buffer.

Callers of `require_device_dtype`: the four BLAS operand describers (`matrix_operand.hpp`,
`packed_operand.hpp`, `scalar_operand.hpp`, `vector_operand.hpp`; `band_operand.hpp` through the
matrix one), the two sparse describers, every executor in `logic/dispatch.hpp`,
`math/elementwise/dispatch.hpp`, `math/nonlinear/dispatch.hpp`, `math/random/dispatch.hpp` and
`math/reductions/dispatch.hpp`, `ReadBinding::require` (`read_binding.hpp:269`), and these
operation files under `src/engine/`: `logic/all.cpp`, `logic/any.cpp`,
`math/elementwise/convert.cpp`, `math/random/constant.cpp`, `orthogonal.cpp`, `shuffle.cpp`, and
all ten files of `spatial/` that record a task.

### Record time

| Layer | Where | Type | When |
| --- | --- | --- | --- |
| Operand checks | `engine/operand_check.hpp`: `require_storage`, `require_dtype`, `require_same_shape`, `require_same_layout`, `require_same_element_count`, `require_min_elements`, `require_contiguous`, `require_operand`, `require_rank`, `require_scalar`, `require_extent` | `std::invalid_argument` | missing storage, dtype mismatch, rank, shape, layout, count, stride, or a rule the caller states |
| Operand checks | `require_device_dtype` | `UnsupportedExecution` | see above |
| Dispatch seam | `Dispatch::require_domain` (`engine/dispatch.hpp:92`) | `std::invalid_argument` | dtype outside the domain the operation serves |
| Dispatch seam | `Dispatch::set_task_params` (`engine/dispatch.hpp:239`) | `std::invalid_argument` | more parameters than `TaskMetadata` holds |
| Dispatch seam | `Dispatch::with_element_type` (`engine/dispatch.hpp:227`) when called while recording (`logic/all.cpp:114`, `logic/any.cpp:114`, the index, counting and span reductions at `math/reductions/dispatch.hpp:584`, `640`, `819`) | `std::runtime_error` | dtype outside the domain; the checks before it make this unreachable |
| Dispatch seam | `execute_reduction` (`math/reductions/dispatch.hpp:321`), `convert.cpp:140` and `202`, `constant.cpp:117` | `std::runtime_error` | a dtype switch fell through; unreachable after the checks before it |
| Recorder | `TaskRecorder::require_host_reachable`, `record_pair` | `UnsupportedExecution` | see above |
| BLAS seam | `engine/blas/argument_check.hpp`, `matrix_operand.hpp`, `vector_operand.hpp`, `packed_operand.hpp`, `band_operand.hpp`, `scalar_operand.hpp`, `utils.hpp` (`require_scalar_result`), `dispatch.hpp` (`GemmEpilogue`, `execute_gemm`) | `std::invalid_argument` | a flag outside its enumeration, a complex scalar for a real dtype, rank, batch, layout, extent, increment, band width, packed length, an epilogue tensor without storage, a complex dtype for a real-only epilogue |
| Sparse seam | `engine/sparse/index_check.hpp`, `sparse_matrix_operand.hpp`, `sparse_vector_operand.hpp` | `std::invalid_argument` | index base, extent, array rank, length, stride, storage, index dtype or width, entry count |
| Spatial seam | `engine/spatial/window.hpp` and the files of `src/engine/spatial/` | `std::invalid_argument` | geometry field, output extent, rank, batch or channel mismatch, a window of padding only |
| Random | `require_random_dtype` (`math/random/dispatch.hpp:109`) and each file of `src/engine/math/random/` | `std::invalid_argument` | dtype the distribution does not draw, a parameter outside its range |
| Random | `RandomOps::poisson` | `UnsupportedExecution` | see above |
| Scratch | `ScratchBuffer` constructor through `Internal::ScratchShelf::take` (`engine/scratch_shelf.hpp:137`) | `std::runtime_error` | device index out of range, or the device allocation failed |
| Tensor | `Tensor::data()` and the view calls (`include/SushiBLAS/tensor.hpp`) when an operation calls them | `std::runtime_error` | no storage, no data pointer, a bad dimension index |
| Runtime | `TaskGraph::add_task`, `reserve_registration` | not read | not traced in sushiruntime |

`SB_ASSERT` in `engine/blas/task_metadata.hpp:94` is not an exception: it logs and calls
`std::abort` in a build without `NDEBUG`, and is empty with it.

### Execute time

| Layer | Where | Type | When |
| --- | --- | --- | --- |
| Engine | `Engine::execute` (`src/engine.cpp:43`) through `TaskGraph::compile` | `std::runtime_error` | a graph with nodes and no root (`sushiruntime/src/graph/task_graph.cpp:660`) or a dependency cycle (`709`). Whether a graph with no nodes reaches 660 was not traced; `Engine::execute` returns without running when `compile` hands back no roots. |
| Engine | `ExecutionEngine::execute` | whatever the first failing task threw | rethrown by `std::rethrow_exception` (`sushiruntime/src/execution/execution_engine.cpp:502`) |
| Runtime | scheduler | `SushiRuntime::Core::SushiException` | a task's buffers span device contexts (`sushiruntime/src/scheduler/scheduler.cpp:676`) |
| Task body | `Internal::Detail::TypedTaskBody::operator()` (`engine/task.hpp:109`) | `std::runtime_error` | dtype outside the domain; unreachable after `require_domain` |
| Task body | `with_element_type` inside a map task (`math/elementwise/map_task.hpp:158`, `167`, `302`, `311`), `execute_random`'s body (`math/random/dispatch.hpp:318`, and `RandomArm` at `264`), `orthogonal.cpp:91` and `96`, `shuffle.cpp:137` | `std::runtime_error` | dtype outside the domain, or HALF in an RNG body; unreachable after the record-time checks |
| SYCL kernel | `Kernels::SYCL::syrk` tiled path (`kernels/sycl/blas/level3/syrk.hpp:218`), `Kernels::SYCL::syr2k` (`syr2k.hpp:129`) | `std::runtime_error` | the device cannot host the kernel's local tile or work-group |
| SYCL | `queue::submit` and the queue's async handler (`sushiruntime/src/execution/runtime_context.cpp:320`, `411`, `461`) | `sycl::exception` | a device error; the handler rethrows the first one |
| Kernels | `SB_ASSERT` in `kernels/reference/sparse/*.hpp`, `kernels/sycl/sparse/spmv.hpp`, `spsymv.hpp`, `kernels/sycl/blas/level3/trsm.hpp`, `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp:226`, `packed_product.hpp:258` | none | `std::abort` in a build without `NDEBUG` |

The `syrk` and `syr2k` rows are device refusals too, and they are still `std::runtime_error`. They
fire at execute time, which the owner's decision does not cover, and a kernel may not include an
engine header (`tools/layering/check_layering.py`), so the helper is out of their reach. Making
them `UnsupportedExecution` would also move them from the `runtime_error` branch to the
`logic_error` branch. `Kernels::SYCL::gemm` has no such throw: it falls back to another kernel
(`gemm.hpp:471`).

### Outside both

| Layer | Where | Type | When |
| --- | --- | --- | --- |
| Tensor | constructors, `transpose`, `slice`, `reshape`, `get_device`, `data` (`include/SushiBLAS/tensor.hpp`) | `std::runtime_error` | rank above 6, negative dimension or stride, overflow, storage too small, element count mismatch, strided reshape, no storage |
| Storage | `Storage` constructor (`include/SushiBLAS/storage.hpp:71` to `86`) | `std::runtime_error` | zero or overflowing size, failed allocation, misaligned address |
| Factory | `TensorFactory::create` (`include/SushiBLAS/tensor_factory.hpp:83` to `110`) | `std::runtime_error` | rank, negative dimension, device index out of range |
| Sparse assembly | `support/sparse_assembly.hpp`, `src/sparse_builder.cpp` | `std::runtime_error` | index outside the matrix, a repeated position, a count the index type cannot hold, a dense operand that is not a contiguous rank-2 tensor with storage, an index or value dtype the format does not hold |
| I/O | `src/io.cpp`, operand rules (`478`, `720`, and the operand checks) | `std::invalid_argument` | strided tensor, no storage, dtype or byte count mismatch, a column-major tensor of rank 3 or more for `.npy` |
| I/O | `src/io.cpp`, every `SB_THROW_IF` | `std::runtime_error` | a file that cannot be opened, read, written or closed, a malformed header, another format version, a header that disagrees with the tensor |

Every `IO` call that reads or writes a tensor runs `Engine::execute` first, so it can also throw
anything in the execute-time table.

### `@throws` lines found out of date

These are not refusal sites and this change left them alone. Each says `std::runtime_error` where
the code calls `require_operand`, `require_rank` or `require_min_elements`, which throw
`std::invalid_argument`:

- `include/SushiBLAS/engine/math/elementwise.hpp:322` (`clamp`, `min_val > max_val`) and `332`
  (`convert`, a dtype outside `Domain::REAL`).
- `include/SushiBLAS/engine/math/random.hpp:151` (`truncated_normal`, the bounds) and `198`
  (`orthogonal`, a tensor that is not rank 2 or is empty).

Three READMEs still describe a refusal by `ExecutionMode::VERIFY` in `Engine::set_mode`
(`kernels/README.md:155`, `include/SushiBLAS/engine/README.md:235`,
`tests/functional/unit/README.md`). The enumeration has no `VERIFY` and `set_mode` is `noexcept`
(`include/SushiBLAS/engine.hpp:94`).

### Syntax check

`clang++ -fsyntax-only` from `D:/Projects/.sushisystems/toolchains/llvm-sycl/bin`, with the flags
of the neighbouring entry of `build/compile_commands.json` (`-std=gnu++17 -Wall -Wextra -Werror
-pedantic -fsycl -fsycl-targets=spir64,nvidia_gpu_sm_86`). Each exited 0 with no diagnostic:

`tests/functional/unit/test_unsupported_execution.cpp`, `test_blas_types.cpp`,
`test_task_recorder.cpp`, `src/engine/math/random/poisson.cpp`, `src/engine/blas/level3/gemm.cpp`,
`src/engine/logic/all.cpp`, `src/engine/math/reductions/argmax.cpp`,
`src/engine/sparse/axpyi.cpp`, `src/engine/spatial/window_dot.cpp`, `src/io.cpp`,
`src/engine/math/elementwise/convert.cpp`, `src/engine/math/nonlinear/gelu.cpp`.

## NAMING_TABLE (the naming table)

The naming sweep ran on 2026-10-07 over the tree at `070581f`. A script applied
`tools/naming/sushiblas_0_1_0.tsv`, which has the columns `tools/naming/README.md` of sushiai
describes, and this is the same table. Nothing was built or run after it. Function-local
`constexpr` values, 136 of them, keep their names.

| Kind | Old name | New name | Declared in |
| --- | --- | --- | --- |
| constant | `ACTIVE_LOG_LEVEL` | `K_ACTIVE_LOG_LEVEL` | `include/SushiBLAS/core/logger.hpp` |
| constant | `IM2COL_NAME` | `K_IM2COL_NAME` | `src/engine/spatial/im2col.cpp` |
| constant | `MAX_TENSOR_RANK` | `K_MAX_TENSOR_RANK` | `include/SushiBLAS/core/common.hpp` |
| constant | `N` | `K_N` | `tests/functional/unit/test_shuffle.cpp` |
| constant | `block_m` | `K_BLOCK_M` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `block_n` | `K_BLOCK_N` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `item_cols` | `K_ITEM_COLS` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `item_rows` | `K_ITEM_ROWS` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `kColMax` | `K_COL_MAX` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kColMin` | `K_COL_MIN` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kColSum` | `K_COL_SUM` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kColumnMajor` | `K_COLUMN_MAJOR` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kHalfMax` | `K_HALF_MAX` | `tests/functional/unit/test_convert.cpp` |
| constant | `kLarge` | `K_LARGE` | `tests/functional/unit/test_count_nonfinite.cpp`, `tests/functional/unit/test_reduction_determinism.cpp` |
| constant | `kMomentRuns` | `K_MOMENT_RUNS` | `tests/functional/unit/test_reduction_determinism.cpp` |
| constant | `kMoments` | `K_MOMENTS` | `tests/functional/unit/test_reduction_determinism.cpp` |
| constant | `kPad` | `K_PAD` | `tests/functional/unit/test_storage_offset.cpp` |
| constant | `kPoisonByte` | `K_POISON_BYTE` | `tests/functional/unit/test_storage_offset.cpp` |
| constant | `kRealDTypes` | `K_REAL_DTYPES` | `tests/functional/unit/test_convert.cpp` |
| constant | `kRowMajor` | `K_ROW_MAJOR` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kRowMax` | `K_ROW_MAX` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kRowMin` | `K_ROW_MIN` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kRowSum` | `K_ROW_SUM` | `tests/functional/unit/test_axis_reductions.cpp` |
| constant | `kSmall` | `K_SMALL` | `tests/functional/unit/test_count_nonfinite.cpp`, `tests/functional/unit/test_reduction_determinism.cpp` |
| constant | `kTenthAsHalf` | `K_TENTH_AS_HALF` | `tests/functional/unit/test_convert.cpp` |
| constant | `kTinyAsHalf` | `K_TINY_AS_HALF` | `tests/functional/unit/test_convert.cpp` |
| constant | `tile_a_ld` | `K_TILE_A_LD` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp` |
| constant | `tile_k` | `K_TILE_K` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `wg_cols` | `K_WG_COLS` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `wg_items` | `K_WG_ITEMS` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| constant | `wg_rows` | `K_WG_ROWS` | `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syrk.hpp` |
| enumerator | `BackwardOrder::ASCENDING` | `Ascending` | `tests/functional/unit/test_blas_half_accumulation.cpp` |
| enumerator | `BackwardOrder::DESCENDING` | `Descending` | `tests/functional/unit/test_blas_half_accumulation.cpp` |
| enumerator | `Core::DataType::COMPLEX32` | `Complex32` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::COMPLEX64` | `Complex64` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::FLOAT32` | `Float32` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::FLOAT64` | `Float64` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::HALF` | `Half` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::INT32` | `Int32` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::DataType::INT64` | `Int64` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::Diag::NON_UNIT` | `NonUnit` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Diag::UNIT` | `Unit` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Dispatch::Domain::ALL` | `All` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `Dispatch::Domain::COMPLEX` | `Complex` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `Dispatch::Domain::INTEGRAL` | `Integral` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `Dispatch::Domain::REAL` | `Real` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `Dispatch::Domain::REAL_AND_COMPLEX` | `RealAndComplex` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `Dispatch::Domain::REAL_AND_INTEGRAL` | `RealAndIntegral` | `include/SushiBLAS/engine/dispatch.hpp` |
| enumerator | `ExecutionMode::AUTO` | `Auto` | `include/SushiBLAS/engine/execution_mode.hpp` |
| enumerator | `ExecutionMode::FORCE_REFERENCE` | `ForceReference` | `include/SushiBLAS/engine/execution_mode.hpp` |
| enumerator | `ExecutionMode::FORCE_SYCL` | `ForceSycl` | `include/SushiBLAS/engine/execution_mode.hpp` |
| enumerator | `Core::IndexBase::ONE` | `One` | `include/SushiBLAS/core/sparse_types.hpp` |
| enumerator | `Core::IndexBase::ZERO` | `Zero` | `include/SushiBLAS/core/sparse_types.hpp` |
| enumerator | `Core::Layout::COLUMN_MAJOR` | `ColumnMajor` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::Layout::ROW_MAJOR` | `RowMajor` | `include/SushiBLAS/core/common.hpp` |
| enumerator | `Core::LogType::DEBUG` | `Debug` | `include/SushiBLAS/core/logger.hpp` |
| enumerator | `Core::LogType::ERR` | `Err` | `include/SushiBLAS/core/logger.hpp` |
| enumerator | `Core::LogType::INFO` | `Info` | `include/SushiBLAS/core/logger.hpp` |
| enumerator | `Core::LogType::OFF` | `Off` | `include/SushiBLAS/core/logger.hpp` |
| enumerator | `Core::LogType::WARN` | `Warn` | `include/SushiBLAS/core/logger.hpp` |
| enumerator | `Mirror::HERMITIAN` | `Hermitian` | `tests/functional/unit/test_sparse_spsymv.cpp` |
| enumerator | `Mirror::SYMMETRIC` | `Symmetric` | `tests/functional/unit/test_sparse_spsymv.cpp` |
| enumerator | `Outcome::INVALID_ARGUMENT` | `InvalidArgument` | `tests/functional/unit/test_unsupported_execution.cpp` |
| enumerator | `Outcome::RETURNED` | `Returned` | `tests/functional/unit/test_unsupported_execution.cpp` |
| enumerator | `Outcome::UNSUPPORTED_EXECUTION` | `UnsupportedExecution` | `tests/functional/unit/test_unsupported_execution.cpp` |
| enumerator | `PanelLines::COLUMNS` | `Columns` | `include/SushiBLAS/kernels/sycl/blas/level3/packed_panel.hpp` |
| enumerator | `PanelLines::ROWS` | `Rows` | `include/SushiBLAS/kernels/sycl/blas/level3/packed_panel.hpp` |
| enumerator | `RebuildValue::INVERSE_COSINE` | `InverseCosine` | `tests/functional/unit/test_givens.cpp`, `tests/functional/unit/test_rotg.cpp` |
| enumerator | `RebuildValue::ONE` | `One` | `tests/functional/unit/test_givens.cpp`, `tests/functional/unit/test_rotg.cpp` |
| enumerator | `RebuildValue::SINE` | `Sine` | `tests/functional/unit/test_givens.cpp`, `tests/functional/unit/test_rotg.cpp` |
| enumerator | `RebuildValue::ZERO` | `Zero` | `tests/functional/unit/test_givens.cpp`, `tests/functional/unit/test_rotg.cpp` |
| enumerator | `Routine::HEMM` | `Hemm` | `tests/functional/unit/test_packed_symm.cpp`, `tests/functional/unit/test_symm.cpp` |
| enumerator | `Routine::SYMM` | `Symm` | `tests/functional/unit/test_packed_symm.cpp`, `tests/functional/unit/test_symm.cpp` |
| enumerator | `Dispatch::Sampling::DRAWS_VALUES` | `DrawsValues` | `include/SushiBLAS/engine/math/random/dispatch.hpp` |
| enumerator | `Dispatch::Sampling::MOVES_VALUES` | `MovesValues` | `include/SushiBLAS/engine/math/random/dispatch.hpp` |
| enumerator | `Scheme::GENERAL_BAND` | `GeneralBand` | `tests/functional/unit/test_blas_structured_validation.cpp` |
| enumerator | `Scheme::PACKED` | `Packed` | `tests/functional/unit/test_blas_structured_validation.cpp` |
| enumerator | `Scheme::TRIANGLE_BAND` | `TriangleBand` | `tests/functional/unit/test_blas_structured_validation.cpp` |
| enumerator | `Core::Side::LEFT` | `Left` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Side::RIGHT` | `Right` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `StructureDefect::DECREASING_OFFSET` | `DecreasingOffset` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `StructureDefect::FIRST_OFFSET` | `FirstOffset` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `StructureDefect::INDEX_OUT_OF_RANGE` | `IndexOutOfRange` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `StructureDefect::LAST_OFFSET` | `LastOffset` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `StructureDefect::NONE` | `None` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `StructureDefect::REPEATED_INDEX` | `RepeatedIndex` | `kernels/reference/sparse/structure.hpp` |
| enumerator | `Core::Transpose::CONJ_TRANS` | `ConjTrans` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Transpose::NO_TRANS` | `NoTrans` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Transpose::TRANS` | `Trans` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Uplo::LOWER` | `Lower` | `include/SushiBLAS/core/blas_types.hpp` |
| enumerator | `Core::Uplo::UPPER` | `Upper` | `include/SushiBLAS/core/blas_types.hpp` |

Not renamed:

- `is_complex_v` (`include/SushiBLAS/core/traits.hpp`): a trait variable template keeps its type-like name.
- `is_real_scalar_v` (`include/SushiBLAS/engine/math/reductions/dispatch.hpp`): a trait variable template keeps its type-like name.
- `value` (`include/SushiBLAS/engine/math/reductions/dispatch.hpp`): the member name the standard trait protocol reads.
