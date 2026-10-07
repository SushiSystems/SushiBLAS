# Remaining work

The single backlog of this repository. A defect that is already in the code is in
[the known issues](../reference/KNOWN_ISSUES.md); this page holds what is to be done. Finding
numbers refer to `docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`.

## Verification of the 2026-10-05 changes

The tree of waves 0 to 9 was built with `sb build --benchmarks` and `sb test` passed 792 of 792 tests on
the OpenCL CPU device (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`), two of them after a fix to `all` and `any` of
an empty tensor. The runtime picks an NVIDIA device first and the default build cannot reach it, so the
device selector stays necessary on this machine.

- Confirm that each new validation, parity and edge-case test fails on the old code and passes on the
  new; no new test has been seen failing before its fix.
- Re-point the `functional` CI job at a SushiRuntime revision that has the allocation interface.
- Run the benchmarks on a device other than the CPU; the numbers of 2026-10-07 are from one Ryzen 5 7600X.

## GPU validation

Nothing has run on GPU hardware. [GPU_VALIDATION.md](GPU_VALIDATION.md) holds the steps and the
expected results.

- Build with `SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"` on a machine with an NVIDIA GPU.
- Confirm `Device::Profile` reports `has_dedicated_local_memory = true` there.
- Measure the bank conflicts of the tiled GEMM and the register spill of its 8x8 block.
- Run the 128-bit vector path and the grid-stride launch shape of `launch_1d.hpp`, which have
  never executed.
- Measure where an axis reduction should switch to one work-group per segment.
- Run the determinism regressions of SushiAI on the GPU.
- Build for an NVIDIA target on Windows to confirm that the nonlinear and elementwise complex kernels no
  longer reach MSVC's `std::complex` functions; the replacement in `safe_math.hpp` was never compiled for
  that target.
- Run the group-per-row GEMV and SYMV paths on a device where they are chosen; on the CPU only a 4 x 16384
  GEMV reaches them and SYMV never does.

## Kernels and reductions

- Give SYRK the local and no-local kernel pair and an epilogue; it always takes the tiled shape.
- Give SYRK GEMM's local-memory padding and store interleave by lifting them out of `gemm.hpp` into
  shared helpers.
- Keep the TRSM and TRSV solution values in `FLOAT32` between substitution steps for `HALF`; today each
  element's sum is formed in `FLOAT32` and the values later steps reuse are read back as `HALF`. It
  needs scratch the size of the right-hand side, so a kernel signature changes.
- Parallelise the TRSM substitution and coalesce its right-side reads.
- Give `max_along` and the other axis reductions a NaN-propagating accumulator in `reduce_1d.hpp`; only
  the whole-tensor `max` and `min` propagate NaN.
- Confirm whether `blas().nrm2` uses the scaled form that `reductions().nrm2` uses
  (`Reduce::ScaledSumOfSquares`); the plain two-norm overflows above about 1e19 in `FLOAT32`.
- Bring the batch feed into the compiled plan; it costs one extra dispatch round per step and
  needs an operand-address late binding that SushiRuntime does not offer.
- Add batch support for tensors above rank 1 to the level 1 and level 2 BLAS operations
  (13 `TODO` lines under `src/engine/blas/level1/` and `level2/`, one in `engine/blas/utils.hpp`).
- Replace the `single_task` kernels of `argmax`, `argmin`, `logsumexp`, `std` and `var` with
  parallel reductions (`src/engine/math/reductions/`).
- Implement `SparseBLAS`: a sparse tensor type, SpMV, SpMM and the CSR format
  (`include/SushiBLAS/engine/blas/sparse.hpp`).
- Decide the backend of the quarantined FFT and LAPACK declarations
  (`include/SushiBLAS/experimental/`).
- Add complex scalars to the BLAS interface, the Hermitian routines (`hemv`, `her`, `herk`) and the
  conjugate transpose.
- Decide whether ELU accepts complex input; it is real-only today.
- Move each of the 32 `TODO` lines out of the source once its item is on this page (C10).

## IO

- Make `save` asynchronous through the task graph (`include/SushiBLAS/io.hpp`).
- Split `IO` into one reader and writer per format over a shared host-copy helper, with a load
  that returns a status (C4). The boundary needs the owner's approval.
- Add tests for `Core::format`, the logger and the four CLI services (C18).

## Code shape

- Decide how `engine.hpp` stops including every operation family: the seven entries of
  `K_KNOWN_VIOLATIONS` are the facade's includes, and removing them makes every consumer include the
  family headers it uses. Owner decision.
- Give the BLAS layer one exception type: its rank, dtype and length pre-checks throw
  `std::runtime_error` while the operand checks throw `std::invalid_argument`.
- Decide what an empty tensor without storage means to the reduction executors; the elementwise, nonlinear
  and logic executors treat it as a no-op.
- Replace the positional `bool` flags of the BLAS calls (`trans`, `upper`, `unit`) with enum classes. This
  changes every call site and the public interface.
- Route every runtime dtype switch through one table; `constant.cpp` and
  `engine/math/reductions/dispatch.hpp` come first (C1).
- Decide whether the execution mode is a library-wide contract, then give the five families a
  reference path or make the mode throw there (C2). Owner decision.
- Add categories, TRACE and FATAL to the logger and send log output to stderr (C3, C6).
- Introduce one task-record builder for the 27 direct `add_task` sites (C7).
- Narrow what the operation views need from `Engine`; the RNG stream already has its own class (C8).
- Give reduction kernels one home (C9).
- Split `level1.hpp` into one file per routine and lift operand addressing out of GEMM's
  `Detail` namespace (C12).
- Remove `core/export.hpp`, `kernels/math.hpp` and the `<numbers>` include (C13). Owner approval.
- Run the source comment pass per module until `check_source_comments.py` prints nothing (C14).
- Split `test_regression.cpp` per defect and state every test tolerance (C17).
- Add column-major and other-dtype cases to the reference tests of BLAS levels 1 and 2 and TRSM, and
  raise the remaining single-case smoke tests to the edge cases the five edited ones now carry.
- Decide where `benchmarks/` lives; it is a new top-level folder outside the layout.

## Command line

- Take build, test, run and clean from a shared sushicore registrar (command line C1, C12).
- Rename `--sort` to `--pick` with a hidden alias, in step with the sibling CLIs (C4).
- Settle the panel names, short options and defaults across the CLIs (C5, C6).
- Document colour, theme and exit codes once in sushicore and link it from `cli/README.md` (C14).
- Remove the unused `find_executables` and `select_interactive` re-exports (C17). Owner approval.
- State the default run target once (C18) and choose one Python floor for the CLIs (C22).

## Documentation

- Split `architecture/ARCHITECTURE.md` by its ten sections and run the humanizer pass over it,
  `guides/INTEGRATION.md` and `GPU_VALIDATION.md` (D12, D15).
- Add a `README.md` to `kernels/`, `src/`, `include/SushiBLAS/`, `tests/` and `examples/`, and
  move each folder's facts out of the architecture and contributing pages (D17).
- Write a design document for the public seam decisions the audit raised: SYCL types in the
  interface, exceptions against statuses, and why GEMM's kernels are installed (C15, C21).
- Correct what the manual says about `sushicore` and PyPI once the published versions are known
  (licence L12).
- Add the third-party licence texts that `NOTICE.md` marks unverified.

## Layout, needing the owner

Each of these edits a file outside the documentation, so none was done with the tree.

- Move the Doxygen main page to `docs/reference/` and the generated site out of `docs/`, in the
  same change as `Doxyfile` and `.gitignore` (D19). Until then `check_docs_layout.py` reports
  `docs/api`, `docs/api-site` and the name `mainpage.md`.
- Move `Doxyfile` out of the root and take its version from CMake (L7, L11).
- Decide `src/` against `source/`, the home of `kernels/` and `examples/`, and the `functional/`
  level of `tests/`, together with SushiRuntime (L8, L14).
- Add `.clang-format`, `.editorconfig` and `.gitattributes`, and the missing `.gitignore` rules
  (L6, L18).
- Decide where `SECURITY.md` and `CODE_OF_CONDUCT.md` live; GitHub reads them from the root,
  `docs/` or `.github/`, and they are now in `docs/reference/`.
- Update the paths that `NOTICE.md`, `Doxyfile`, `.github/workflows/ci.yml` and
  `tests/package/CMakeLists.txt` cite for documents that moved on 2026-10-05.
- Delete the merged local branches `feature/sushicore_help_screen` and
  `feature/blas-spatial-domain`, and push main (L17).

## Continuous integration

Workflow files change after the code programme, when the source comment checker can pass.

- Install the CLI with `pip install -e cli` once `sushicore` 0.7.0 is published, and remove the
  SushiStack checkout from the `cli` job and the dead candidate from `cli/tests/conftest.py`
  (L2, L13, L15).
- Remove `|| true` from the pyflakes step and correct the stale comments (L4, L5).
- Add a Windows job, a checker job and a release workflow (L3).

## Release

- Tag v0.1.0 and cut the Unreleased section under it (L11). Owner decision.
- Check the CLI package version against the CMake version (L16).
