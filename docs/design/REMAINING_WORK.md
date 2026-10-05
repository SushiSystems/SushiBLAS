# Remaining work

The single backlog of this repository. A defect that is already in the code is in
[the known issues](../reference/KNOWN_ISSUES.md); this page holds what is to be done. Finding
numbers refer to `docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`.

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

## Kernels and reductions

- Give SYRK the local and no-local kernel pair and an epilogue; it always takes the tiled shape.
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
- Move each of the 32 `TODO` lines out of the source once its item is on this page (C10).

## IO

- Make `save` asynchronous through the task graph (`include/SushiBLAS/io.hpp`).
- Parse the whole `.npy` header and check dtype, `fortran_order` and the exact shape
  (`src/io.cpp`).
- Split `IO` into one reader and writer per format over a shared host-copy helper, with a load
  that returns a status (C4). The boundary needs the owner's approval.
- Add tests for IO, `Core::format`, the logger and the four CLI services (C18).

## Code shape

- Port `Storage` and `Engine::create_tensor` to the allocation interface SushiRuntime has had
  since its commit `6e4b87b`; the library does not compile against the sibling checkout until then.
- Route every runtime dtype switch through one table; `constant.cpp` and
  `engine/math/reductions/dispatch.hpp` come first (C1).
- Decide whether the execution mode is a library-wide contract, then give the five families a
  reference path or make the mode throw there (C2). Owner decision.
- Add categories, TRACE and FATAL to the logger and send log output to stderr (C3, C6).
- Introduce one task-record builder for the 27 direct `add_task` sites (C7).
- Narrow what the operation views need from `Engine` and move the RNG stream out of it (C8).
- Give reduction kernels one home (C9) and move the launch, reduce, device-policy and geometry
  headers below both `engine/` and `kernels/` (C11, C21).
- Split `level1.hpp` into one file per routine and lift operand addressing out of GEMM's
  `Detail` namespace (C12).
- Remove `core/export.hpp`, `kernels/math.hpp` and the `<numbers>` include (C13). Owner approval.
- Give `execute_random` a `Domain` parameter so the nine distributions reject at record time
  (C20, C25).
- Run the source comment pass per module until `check_source_comments.py` prints nothing (C14).
- Seed the shared test fill helper, state tolerances, and split `test_regression.cpp` per defect
  (C17).

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
