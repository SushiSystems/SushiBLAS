# SushiBLAS

SushiBLAS is a C++17 tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. You create tensors and queue operations on them through an `Engine`; the runtime tracks the dependencies between the queued tasks and runs them on a CPU or a GPU through SYCL. No kernel calls a vendor math library: there is no oneMKL and no cuBLAS behind it.

The manual is in [docs/README.md](docs/README.md).

- **Operations are recorded, then run.** `Engine::create_tensor()` allocates a `Tensor`, which is shape, stride and dtype metadata over a ref-counted `Storage` backed by SushiRuntime USM. Calling an operation, such as `engine.blas().axpy(...)`, registers a task on the engine's `TaskGraph`. `engine.execute()` compiles and runs the graph.
- **Every operation has two kernels.** A plain C++ reference kernel under `kernels/reference/` is the mathematical ground truth, and a SYCL kernel under `kernels/sycl/` runs on the device. GEMM's pair sits under `include/SushiBLAS/kernels/`, because the public `Dispatch::execute_gemm` seam instantiates it in a consumer's translation unit. The GEMM and SYRK tiling is ported from portBLAS; see [NOTICE.md](NOTICE.md).
- **The execution mode is chosen at run time.** `Engine::set_mode()` selects `AUTO`, `FORCE_SYCL`, `FORCE_REFERENCE` or `VERIFY` for every later operation. `VERIFY` runs the SYCL kernel and then the reference kernel over the same output; it does not compare the two. See [the architecture, section 3](docs/architecture/ARCHITECTURE.md#3-executionmode-the-dispatch-triad).
- **An unsupported dtype throws.** `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX32` and `COMPLEX64` are served across BLAS levels 1 to 3 and the nonlinear, logic and random families, dispatched from one `Core::DataType` switch. Where a dtype has no meaning for an operation, such as `max` on a complex tensor, the call throws and names the operation and the dtype.

## Requirements

- A SYCL 2020 compiler: the intel-llvm `clang++ -fsycl` that `sb setup` or SushiStack provisions. SushiBLAS selects no toolchain of its own; it builds with the one in the workspace `dependencies/` tree or under `~/.sushisystems`.
- CMake 3.20 or newer, and 3.22 for the test suite's `gtest_discover_tests`.
- GoogleTest for the test suite.
- A SushiRuntime checkout, found by [the layered resolution of section 8](docs/architecture/ARCHITECTURE.md#8-the-sushiruntime-dependency-d3): a superproject target, an installed `find_package(SushiRuntime)`, or a sibling `../sushiruntime` checkout.
- C++17 or later. The library is built at C++17; a C++20 consumer is supported.

## Quick start

SushiStack's bootstrap script installs Python and Git, installs `hub`, then runs `hub install` to fetch every toolchain and library the workspace's modules declare. SushiBLAS declares its own in `cli/sushistack.deps.toml`.

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

On Windows use `irm`. In PowerShell `curl` is an alias for `Invoke-WebRequest` and does not pipe a script the same way.

To use an existing checkout of this repository, link it into a workspace with `hub link`; `hub doctor` and `hub status` show what is installed. The [SushiStack](https://sushisystems.io) documentation covers both.

### Using SushiBLAS from another project

Call `find_package(SushiBLAS REQUIRED)` and link `SushiBLAS::sushiblas`. Read the [integration guide](docs/guides/INTEGRATION.md) first: the SYCL parts of a consumer's build have requirements it explains. `tests/package/` is a complete consumer.

## Building

The CLI loads the compiler and toolchain paths from its configuration and drives CMake. Install it through `hub`:

```bash
hub install-cli sushiblas           # always editable, against this checkout
```

This puts two equivalent commands on the PATH, `sb` and `sushiblas`. Run `sb setup` once before the first build; `sb setup --dry-run` shows what it would install. Shared defaults are in `cli/config.toml`; machine-specific paths go in the gitignored `cli/config.local.toml`. Inside a SushiStack workspace, or with a SushiRuntime checkout beside this one, no local configuration is needed.

```bash
sb build                      # release (default)
sb build --type debug         # debug build
sb build --type relwithdebinfo
sb build --clean              # wipe the build tree first
```

`sb clean` removes the `build/` tree. Every command is described in [cli/README.md](cli/README.md).

### Targeting an NVIDIA GPU

By default the device image is SPIR-V, compiled by the runtime for the device it finds. That reaches every OpenCL and Level Zero device and no NVIDIA GPU: NVIDIA's OpenCL driver does not implement `clCreateProgramWithIL`, so it cannot ingest SPIR-V. The CUDA image has to be compiled ahead of time:

```bash
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
```

Keep `spir64` in the list to keep the CPU device reachable from the same build.

The flag cannot check two requirements. The compiler must be an intel-llvm built with `--cuda`; a stock oneAPI release has no CUDA backend and rejects the triple. And intel-llvm documents support for `sm_75` and newer.

`-D` passes a cache variable to CMake and is repeatable. The value stays in the build tree after the flag is dropped; pass `-D SB_SYCL_TARGETS=` to clear it, or `--clean` to start over.

To see whether a kernel spills registers:

```bash
sb build --type release -D SB_SYCL_TARGETS=nvidia_gpu_sm_86 -D SB_SYCL_DEVICE_FLAGS="-Xcuda-ptxas;-v"
```

The configure step prints the active targets. None of the GPU path has run on hardware yet; [the GPU validation design](docs/design/GPU_VALIDATION.md) lists what is unverified.

### Building with CMake directly

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/clang \
  -DSUSHIRUNTIME_DIR=/path/to/sushiruntime
cmake --build build --config Release
```

`CMAKE_C_COMPILER` must point at the toolchain's plain `clang` binary. A `clang++` binary compiles every file as C++ and breaks CMake's C probe; the project declares `LANGUAGES CXX C`. `SUSHIRUNTIME_DIR` is optional; without it the resolution falls back to a sibling `../sushiruntime` checkout or an installed package.

### Other CLI commands

```bash
sb config         # print the resolved configuration and the source of each value
sb env            # print the environment the cmake, ctest and run subprocesses see
sb setup          # provision what the build needs, then report readiness
sb doctor         # report whether this machine can build the module
sb docs           # build the Doxygen API reference into docs/api-site/html/
```

## Running the tests

```bash
sb test                          # functional suite (the default)
sb test --suite unit             # unit only
sb test --suite integration      # integration only
sb test --suite regression       # regression only
sb test --suite all              # every CTest test
sb test --suite package          # the out-of-tree consumer of the installed package
sb test --filter 'Gemm.*'        # ctest -R over 'Suite.Case' test names
sb test --repeat 5               # re-run until first failure

# or run the binary directly
sb run sushiblas_tests -- --gtest_filter='Unit_GEMMTest.*'

# or through ctest, by label
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

The eight accessors of `Engine` are `blas()`, `elementwise()`, `nonlinear()`, `logic()`, `reductions()`, `random()`, `spatial()` and `io()`. Each returns a view object over the same engine and owns no state. Calling an operation on one queues a task on `engine.get_graph()`; nothing runs until `engine.execute()`.

```cpp
engine.elementwise().add(A, B, C);           // C = A + B
engine.nonlinear().relu(A);                  // in-place ReLU
engine.logic().greater(A, B, mask);          // mask = A > B, element-wise
engine.reductions().sum(A, total);           // total = sum(A)
engine.blas().axpy(2.0f, x, y);              // y = 2*x + y
```

`Tensor` is a metadata view (shape, strides, rank, `Core::DataType`, layout) over a ref-counted `Storage`. `transpose()`, `reshape()` and `slice()` return new `Tensor` views over the same storage and move no data. [The architecture](docs/architecture/ARCHITECTURE.md) describes each layer.

## Project layout

```
include/SushiBLAS/     Public headers
  SushiBLAS.h           The umbrella header
  engine.hpp            The Engine facade: create_tensor(), set_mode(), execute()
                         and the eight accessors
  engine/               Per-domain operation interfaces and the public seams
    task.hpp              execute_typed_task and dispatch_with_mode
    dispatch.hpp          The Domain vocabulary shared by the Dispatch seams
    device_policy.hpp     Device::Profile
    launch_1d.hpp         The per-element launch shape
    reduce_1d.hpp         The whole-tensor fold
    reduce_axis.hpp       The segmented fold
    blas/                 Level1, Level2, Level3, SparseBLAS (stub), utils.hpp
                           (execute_blas_task), dispatch.hpp (execute_gemm)
    logic/                LogicOps and its dispatch.hpp
    math/                 ElementwiseOps, NonLinearOps, ReductionOps, RandomOps,
                           their dispatch.hpp files and safe_math.hpp
    spatial/              geometry.hpp, window.hpp
    blas.hpp, spatial.hpp BLASOps and SpatialOps
  kernels/              The kernels a public seam instantiates in a consumer's
                         translation unit, and which are therefore installed:
                         GEMM's reference and SYCL pair and the two epilogue headers
  core/                 common.hpp (DataType, Layout), logger.hpp, format.hpp,
                         traits.hpp, export.hpp
  experimental/         Quarantined FFT and LAPACK declarations
  tensor.hpp            Tensor: shape, stride and dtype metadata over a Storage
  storage.hpp           Storage: ref-counted USM allocation
  io.hpp                Save, load and print (native, .npy, raw binary)
  version.hpp           SUSHIBLAS_VERSION_* macros
src/                    Implementations (engine/<domain>/<op>.cpp, engine.cpp, io.cpp)
kernels/                Private kernels, never installed
  reference/            Plain C++ kernels
  sycl/                 SYCL kernels
examples/               example.cpp, io_example.cpp
tests/
  common/               Shared GoogleTest entry point and helpers
  functional/           unit, integration, regression
  package/              Out-of-tree find_package(SushiBLAS) consumer
cli/                    The `sb` / `sushiblas` CLI package; see cli/README.md
cmake/                  CMake modules
tools/                  Documentation, layering and licence checkers; see tools/README.md
docs/                   The manual; see docs/README.md
Doxyfile                Input of `sb docs`
sushi-module.toml       What `hub` reads about this module
LICENSE                 PolyForm Noncommercial License 1.0.0
COMMERCIAL.md           How to ask for a commercial licence
NOTICE.md               Third-party notices, including the portBLAS-derived tiling
third_party/licenses/   Licence texts of the third-party parts NOTICE.md lists
```

## License

SushiBLAS is source-available, free for non-commercial use under the PolyForm Noncommercial License 1.0.0; see [LICENSE](LICENSE). Commercial use needs a licence from Sushi Systems; see [COMMERCIAL.md](COMMERCIAL.md).

The GEMM and SYRK tiling is ported from portBLAS and stays under Apache-2.0. [NOTICE.md](NOTICE.md) lists it with the other third-party parts, and states which earlier commits carried the MIT License and the Apache License 2.0.
