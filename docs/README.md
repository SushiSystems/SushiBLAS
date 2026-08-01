# SushiBLAS

SushiBLAS is a C++20 tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. You create tensors, queue operations against them through an `Engine`, and the runtime's automatic dependency tracking orders and parallelizes the work across CPU and GPU using SYCL. It has zero dependency on a vendor math library (no oneMKL, no cuBLAS) — every kernel is SushiBLAS's own SYCL or plain-C++ code.

It is built around four ideas:

- **A tensor and task-graph facade over SushiRuntime.** `Engine::create_tensor()` allocates a `Tensor` (shape/stride/dtype metadata plus a ref-counted `Storage` backed by SushiRuntime USM); calling an op (`engine.blas().axpy(...)`, `engine.elementwise().add(...)`) does not run it — it registers a task on the engine's `TaskGraph`, and SushiRuntime's dependency tracker orders it against every other queued task automatically. `engine.execute()` compiles and runs the graph.
- **Two kernels per operation, always.** Every op has a pure-C++ **reference** kernel under `kernels/reference/` (the mathematical ground truth, zero external dependencies) and a **SYCL** kernel under `kernels/sycl/` (the hardware dispatcher). Neither kernel calls into a vendor BLAS — GEMM/SYRK use SushiBLAS's own local-memory-tiled SYCL kernels (ported from portBLAS's tiling algorithm, see [NOTICE.md](../NOTICE.md)), and every other BLAS level and every non-BLAS op is original, from-scratch SYCL.
- **Runtime-selectable execution mode.** `Engine::set_mode()` switches every subsequent op between `AUTO` (device dispatch), `FORCE_SYCL`, `FORCE_REFERENCE` (the CPU ground truth, for debugging), and `VERIFY` (run the SYCL kernel, then run the reference kernel and cross-check) — see [ARCHITECTURE.md §3](ARCHITECTURE.md#3-executionmode-the-dispatch-triad).
- **Uniform type coverage.** `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX32`, and `COMPLEX64` are supported across BLAS levels 1-3, elementwise, nonlinear, logic, reductions, and random — dispatched at run time from one `Core::DataType` switch (`Internal::execute_blas_task`), not duplicated per call site.

## Requirements

- A SYCL 2020 compiler — the bundled **intel-llvm** `clang++ -fsycl`, provisioned by SushiStack (see below). SushiBLAS selects no SYCL toolchain of its own; it builds with whatever [SushiRuntime](https://github.com/SushiSystems/SushiRuntime) sibling checkout it resolves against.
- **CMake** 3.20 or newer (3.22 for the test suite's `gtest_discover_tests`).
- **GoogleTest** for the test suite.
- A **SushiRuntime** checkout, resolved via [D3 layered CMake resolution](ARCHITECTURE.md#8-the-sushiruntime-dependency-d3): an in-tree superproject target, an installed `find_package(SushiRuntime)`, or (the common case) a sibling `../sushiruntime` checkout.
- **C++20.**

C++ library dependencies and the SYCL toolchain are provisioned by [SushiStack](https://sushisystems.io) (the `ss` CLI), which owns dependency installation for the whole stack (not just this repo) into one shared, workspace-level `dependencies/` tree.

## Quick start (one script)

On a fresh machine you can go from nothing to a working build with a single command. SushiStack's bootstrap script installs Python/Git, installs `ss`, then runs `ss install` to fetch every toolchain and library the workspace's modules declare (SushiBLAS declares its own in `cli/sushistack.deps.toml`):

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

On Windows use `irm`, not `curl` — in PowerShell `curl` is an alias for `Invoke-WebRequest` and does not pipe a script the same way.

See the [SushiStack](https://sushisystems.io) docs for linking an existing checkout of this repo into a workspace (`ss link`) instead of a fresh clone, and for `ss doctor` / `ss status` to inspect what's installed.

### Using SushiBLAS from another project

`find_package(SushiBLAS REQUIRED)` and link `SushiBLAS::sushiblas`. Read [INTEGRATION.md](INTEGRATION.md) first — like SushiRuntime, the SYCL-touching parts of your build have real requirements this document explains. `tests/package/` is a complete working consumer.

## Building

The fastest path is the bundled CLI, which loads your compiler/toolchain paths from config and then drives CMake. Install it through the umbrella — the single install seam for the whole stack (it sets up an isolated pipx venv and wires in the shared `sushicli` presentation layer):

```bash
ss install-cli sushiblas           # add --no-editable for a non-dev install
```

This gives you two equivalent commands on your PATH: **`sb`** (short) and **`sushiblas`** (long). Toolchain paths are read from `cli/config.toml` (committed, shared defaults); machine-specific absolute paths go in a gitignored `cli/config.local.toml` using the same `[tool.<platform>]` section layout. In the common case (inside a SushiStack workspace, or a sibling checkout of SushiRuntime present) no local config is needed at all — the CLI finds the runtime sibling and its bundled `clang++` on its own.

```bash
sb build                      # release (default)
sb build --type debug         # debug build
sb build --type relwithdebinfo
sb build --clean              # wipe the build tree first
```

Add `--clean` to wipe the build tree first. `sb clean` removes the `build/` tree entirely.

### Building with CMake directly

If you prefer to skip the wrapper:

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/clang \
  -DSUSHIRUNTIME_DIR=/path/to/sushiruntime
cmake --build build --config Release
```

`CMAKE_C_COMPILER` must point at the toolchain's plain `clang` binary, not `clang++` — a `clang++`-personality binary hardcodes C++ mode regardless of file extension and breaks CMake's C-language probe (the project declares `LANGUAGES CXX C`). `SUSHIRUNTIME_DIR` is optional; without it, [D3 resolution](ARCHITECTURE.md#8-the-sushiruntime-dependency-d3) falls back to a sibling `../sushiruntime` checkout or an installed package.

### Other CLI commands

```bash
sb config         # print resolved config with per-field source annotation
sb env            # print the environment cmake/ctest/run subprocesses see
sb doxygen        # build the Doxygen API reference
```

## Running the tests

```bash
sb test                          # functional suite (the default)
sb test --suite unit             # unit only
sb test --suite integration      # integration only
sb test --suite regression       # regression only
sb test --suite all              # everything
sb test --filter 'Gemm.*'        # ctest -R over 'Suite.Case' test names
sb test --repeat 5               # re-run until first failure (flaky hunt)

# or run the binary directly
sb run sushiblas_tests -- --gtest_filter='Unit_GEMMTest.*'

# or via ctest, by label
ctest --test-dir build -L unit
ctest --test-dir build -L integration
ctest --test-dir build -L regression
```

## How it works

```cpp
#include <SushiBLAS/SushiBLAS.h>

sr::Execution::RuntimeContext ctx;
sb::Engine engine(ctx, sb::Core::Layout::ROW_MAJOR);

auto A = engine.create_tensor({1024, 1024});   // FLOAT32, Shared USM, by default
auto B = engine.create_tensor({1024, 1024});
auto C = engine.create_tensor({1024, 1024});

engine.random().uniform(A, 0.0f, 1.0f);
engine.random().uniform(B, 0.0f, 1.0f);

engine.blas().gemm(A, B, C);          // queues the task, does not run it yet
engine.execute();                     // compiles and runs the graph
ctx.wait_all();                       // block until device work completes
```

Every accessor off `Engine` — `blas()`, `elementwise()`, `nonlinear()`, `logic()`, `reductions()`, `random()`, `io()` — returns a lightweight view object over the same engine; none of them own state. Calling an operation on any of them queues one task on `engine.get_graph()`; nothing runs until `engine.execute()`.

```cpp
engine.elementwise().add(A, B, C);           // C = A + B
engine.nonlinear().relu(A);                  // in-place ReLU
engine.logic().greater(A, B, mask);          // mask = A > B, element-wise
engine.reductions().sum(A, total);           // total = sum(A)
engine.blas().axpy(2.0f, x, y);              // y = 2*x + y
```

`Tensor` is a metadata view (shape, strides, rank, `Core::DataType`, layout) over a ref-counted `Storage`; `transpose()`, `reshape()`, and `slice()` all return new `Tensor` views over the *same* storage — no data movement. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete layer-by-layer design: the `Tensor`/`Storage` model, `Engine` and `ExecutionMode`, the reference/SYCL kernel split, the `execute_blas_task`/`dispatch_with_mode` dispatch helpers, and how the SushiRuntime dependency is resolved.

## Project layout

```
include/SushiBLAS/     Public headers (this is the API surface)
  engine.hpp            The Engine facade: create_tensor(), set_mode(), execute(),
                         and the blas()/elementwise()/nonlinear()/logic()/
                         reductions()/random()/io() accessors
  engine/               Per-domain op interfaces (BLASOps, ElementwiseOps, ...)
    blas/                 Level1 (vector-vector), Level2 (matrix-vector),
                           Level3 (matrix-matrix), SparseBLAS (stub), utils.hpp
                           (execute_blas_task, dispatch_with_mode)
    logic/                LogicOps: comparisons, boolean reductions, where()
    math/                 ElementwiseOps, NonLinearOps, ReductionOps, RandomOps
  core/                 common.hpp (DataType, Layout), logger.hpp
  experimental/         Quarantined FFT/LAPACK surface (see ARCHITECTURE.md §9)
  tensor.hpp            Tensor: shape/stride/dtype metadata over a Storage
  storage.hpp           Storage: ref-counted USM allocation
  io.hpp                Save/load/print (native, .npy, raw binary)
  version.hpp           SUSHIBLAS_VERSION_* macros
src/                    Implementations of the above (engine/<domain>/<op>.cpp)
kernels/
  reference/            Pure C++ kernels — the mathematical ground truth
  sycl/                 SYCL kernels — the hardware dispatchers
examples/               example.cpp (a 1024x1024x1024 GEMM), io_example.cpp
tests/
  common/               Shared GoogleTest entry point and helpers
  functional/           unit / integration / regression
  package/              Out-of-tree find_package(SushiBLAS) consumer
cli/                    Python `sb` / `sushiblas` CLI package (Typer + rich)
docs/                   This documentation
NOTICE.md               Apache-2.0 attribution for the portBLAS-derived GEMM/SYRK tiling
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.
