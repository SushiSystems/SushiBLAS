# SushiBLAS

SushiBLAS is a C++17 tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. You create tensors through an `Engine` and record operations on them through view objects built from it; the runtime tracks the dependencies between the recorded tasks and runs them on a CPU or a GPU through SYCL. No kernel calls a vendor math library: there is no oneMKL and no cuBLAS behind it.

No release is tagged; the version in `CMakeLists.txt` is 0.1.0. The tree has been built and tested on one machine with one card, and [the known issues](docs/reference/KNOWN_ISSUES.md) say what that leaves out. This page says what the code does; it gives no speed figure.

The manual is in [docs/README.md](docs/README.md).

- Operations are recorded, then run. `Engine::create_tensor()` allocates a `Tensor`, which is shape, stride and dtype metadata over a ref-counted `Storage` backed by SushiRuntime USM. Calling an operation, such as `SushiBLAS::BLASOps(engine).axpy(...)`, records one task on the engine's graph and returns a `TaskHandle`. `engine.execute()` runs the graph, returns when every task has finished, and reports the run as a `SushiRuntime::Result<void>`.
- There are eight operation families: dense BLAS levels 1 to 3 with the band, packed, rotation, complex and Hermitian routines; sparse BLAS over CSR; elementwise arithmetic; non-linear activations; logic and comparison; reductions; random generation; and spatial (windowed) operations. [The architecture, section 7](docs/architecture/KERNELS.md) lists the routines.
- Every operation has a reference path and a SYCL kernel. The BLAS, sparse and spatial families have a plain C++ reference kernel in a `reference/` folder beside the SYCL kernel in a `sycl/` folder, under `modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/`. The other families run one functor two ways: inlined into a SYCL kernel, or in a serial host loop. The GEMM and SYRK tiling is ported from portBLAS; see [NOTICE.md](NOTICE.md).
- The execution mode is stamped on each task as it is recorded. `Engine::set_mode()` selects `Auto`, `ForceSYCL` or `ForceReference` for every later operation. See [the architecture, section 3](docs/architecture/EXECUTION_MODES.md).
- A wrong call throws when it is recorded. A wrong operand, flag or scalar throws `std::invalid_argument` and names the operation and the operand; a dtype with no meaning for an operation, such as `max` on a complex tensor, is one of those. An operation that cannot run under the mode or on the device throws `SushiBLAS::UnsupportedExecution`, which derives from it. A failed run throws nothing: `execute()` returns the error. See [the architecture, section 11](docs/architecture/EXCEPTIONS.md).
- Fusion is written by the caller. The `SushiBLAS::Dispatch` seams take a functor and run it as one kernel, and `Dispatch::execute_gemm` takes an epilogue applied to each element of the product. No pass fuses recorded tasks automatically.

## Stability

SushiBLAS is below 1.0.0, and its public interface is not stable.

- A MINOR release (0.2, 0.3) may change or remove any public name, signature, header path or binary layout. A PATCH release does not.
- Pin an exact tag. Do not follow a branch. No tag exists yet; the version in `CMakeLists.txt` is 0.1.0.
- Reach SushiBLAS through one adapter of your own: a single module that includes `<SushiBLAS/SushiBLAS.hpp>` and exposes only what your code needs, in your own types. When a release breaks the interface, that module is the only place you edit.
- SushiBLAS's public headers include `<SushiRuntime/sushiruntime.hpp>` and name `SushiRuntime::Runtime`, `Result` and `Residency`, so a SushiRuntime release that changes those changes this interface too. The build names no SushiRuntime version: `cmake/Runtime.cmake` takes the package or the checkout it finds.

`applications/examples/adapter/` is such an adapter in a program this repository builds: its two adapter files, `sushiblas_adapter.cpp` and `status_mapping.hpp`, are the only ones that include the library, and they turn every exception SushiBLAS throws and every `SushiRuntime::Error` a run returns into the program's own `Status`. [The integration guide](docs/guides/INTEGRATION.md#5-write-one-adapter) walks through it. SushiAI's seam (`sushiai/modules/core/core/source/seam.cpp`) does the same for the failure channels of a larger consumer.

## Requirements

- A SYCL 2020 compiler: the intel-llvm `clang++ -fsycl` that `sb setup` or SushiStack provisions. SushiBLAS selects no toolchain of its own; it builds with the one in the workspace `dependencies/` tree or under `~/.sushisystems`.
- CMake 3.20 or newer, and 3.22 for the test suite's `gtest_discover_tests`.
- GoogleTest for the test suite.
- A SushiRuntime checkout, found by [the layered resolution of section 8](docs/architecture/RUNTIME_DEPENDENCY.md): a superproject target, an installed `find_package(SushiRuntime)`, or a sibling `../sushiruntime` checkout.
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

By default the build takes the device targets SushiRuntime resolved. When SushiRuntime publishes none, the device image is SPIR-V, compiled by the runtime for the device it finds. That reaches every OpenCL and Level Zero device and no NVIDIA GPU: NVIDIA's OpenCL driver does not implement `clCreateProgramWithIL`, so it cannot ingest SPIR-V. The CUDA image has to be compiled ahead of time:

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

The configure step prints the active targets. The functional suite has run on one card, an RTX 3080 Ti: on 2026-10-12, on Windows with the pinned intel-llvm toolchain, 2 207 tests ran and none failed. [The GPU validation design](docs/design/GPU_VALIDATION.md) lists what is unverified.

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
sb docs           # build the Doxygen API reference into build/docs/api-site/html/
```

## Running the tests

```bash
sb test                          # the small and medium tests (the default)
sb test --label small            # one size; small, medium or large, repeatable
sb test --package                # the out-of-tree consumer of the installed package
sb test --filter 'Gemm.*'        # ctest -R over 'Suite.Case' test names
sb test --repeat 5               # re-run until first failure

# or run one test program directly; a module's is sushiblas_<module>_tests
sb run sushiblas_blas_tests -- --gtest_filter='Unit_GEMMTest.*'
```

## How it works

```cpp
#include <vector>
#include <cstdint>
#include <SushiBLAS/SushiBLAS.hpp>

namespace sb = SushiBLAS;
namespace sr = SushiRuntime;

int main()
{
    constexpr int64_t K_SIZE = 256;
    constexpr float K_A_VALUE = 1.0f;
    constexpr float K_B_VALUE = 2.0f;

    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime, sb::Core::Layout::ColumnMajor);

    auto A = engine.create_tensor({K_SIZE, K_SIZE});
    auto B = engine.create_tensor({K_SIZE, K_SIZE});
    auto C = engine.create_tensor({K_SIZE, K_SIZE});

    const size_t count = static_cast<size_t>(K_SIZE * K_SIZE);
    SushiBLAS::IO(engine).from_vector(A, std::vector<float>(count, K_A_VALUE));
    SushiBLAS::IO(engine).from_vector(B, std::vector<float>(count, K_B_VALUE));

    SushiBLAS::BLASOps(engine).gemm(A, B, C);

    const std::vector<float> result = SushiBLAS::IO(engine).to_vector<float>(C);

    // ...
}
```

Source: `applications/examples/quickstart_demo.cpp`

The omitted lines compare every element of `C` with the value the host computes, print one element and return 1 on a mismatch. `create_tensor` gives FLOAT32 tensors in shared USM by default. `gemm` records a task and runs nothing. `IO::to_vector` runs the recorded graph and then copies `C` to the host, so the example never calls `execute()` itself.

A program that reads no tensor through `IO` runs the graph with `Engine::execute()`:

```cpp
SushiBLAS::BLASOps(engine).axpy(2.0f, x, y);
engine.execute().value();
```

Source: `tests/package/main.cpp`

`execute()` returns a `SushiRuntime::Result<void>` and is `[[nodiscard]]`. `.value()` on a result that holds an error stops the process; a caller that can go on tests the result and reads `error()`.

`Engine` is built on a `SushiRuntime::Runtime`, which must outlive it. It owns the graph and has no accessor per operation family. A family is a view class built from the engine: `BLASOps`, `SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`, `SpatialOps` and `IO`. A view holds references and no other state. Each record call returns a `TaskHandle`.

The BLAS flags are the enumerations `Core::Transpose`, `Core::Uplo`, `Core::Diag` and `Core::Side`, and a scalar argument is a `BLASScalar`, which converts from a `double` or a `std::complex<double>`.

`Tensor` is a metadata view (shape, strides, rank, `Core::DataType`, layout) over a ref-counted `Storage`. `transpose()`, `reshape()` and `slice()` return new `Tensor` views over the same storage and move no data. A recorded task keeps the storage of its operands alive until the graph drops it. [The architecture](docs/architecture/README.md) describes each layer.

## Project layout

```
modules/                One tree per module: README.md, CMakeLists.txt, include/, source/, tests/
  foundation/
    core/               Element types, traits, the logger, errors, the version macros
    arithmetic/         The accumulator trait, complex arithmetic, safe math, Math
    launch/             The device policy and the per-element launch shapes
    reduction/          The fixed-order whole-tensor and segmented reductions
    reference/          The reference accumulator and the serial host loops and folds
    substitution/       RunningSolution, the unrounded solution of a substitution
    orthonormal/        The layout and constants of the orthonormalisation sweep
    sparse_assembly/    The host assembly of CSR arrays from triplets
    geometry/           Vector3T, QuaternionT, Matrix4T, transforms, the window geometry
    graph_primitives/   Reduce, segmented reduce, prefix scan and counting sort on a task graph
  data/
    tensor/             Storage, Tensor, TensorFactory
    sparse_storage/     The CSR matrix, the sparse vector and the builder of both
  kernels/
    blas_seam_kernels/  The GEMM kernels, the packed product and the access policies a
                         public seam instantiates; installed
    blas_kernels/, random_kernels/, sparse_kernels/, spatial_kernels/
                        The reference and SYCL kernels of each family; never installed
  engine/
    engine/             Engine, TaskRecorder, the execution mode, scratch, the random stream
    operands/           The operand checks and views the families share
    functor_map/        The per-element seam: the map task and its bindings
  operations/
    blas/, elementwise/, nonlinear/, reductions/, random/, logic/, sparse/, spatial/
                        One operation view each, with one <op>.cpp per operation
    io/                 Host transfer, save, load and print (native, .npy, raw binary)
    experimental/       Quarantined FFT and LAPACK declarations
  library/
    umbrella/           SushiBLAS.hpp, every public header in one line
    sushiblas/          The one library a consumer links
applications/examples/  Eight example programs, the adapter example among them
tests/                  Every test, mirroring modules/; see tests/README.md
  <tier>/<module>/      test_<header>.cpp per public header, and benchmark_*.cpp
                        built with sb build --benchmarks
  package/              Out-of-tree find_package(SushiBLAS) consumer
cli/                    The `sb` / `sushiblas` CLI package; see cli/README.md
cmake/                  CMake modules
tools/                  Documentation, layering and licence checkers; see tools/README.md
docs/                   The manual; see docs/README.md
.config/                Configuration a flag can point a tool at; see .config/README.md
sushi-module.toml       What `hub` reads about this module
LICENSE                 PolyForm Noncommercial License 1.0.0
COMMERCIAL.md           How to ask for a commercial licence
NOTICE.md               Third-party notices, including the portBLAS-derived tiling
third_party/licenses/   Licence texts of the third-party parts NOTICE.md lists
```

## License

SushiBLAS is source-available, free for non-commercial use under the PolyForm Noncommercial License 1.0.0; see [LICENSE](LICENSE). Commercial use needs a licence from Sushi Systems; see [COMMERCIAL.md](COMMERCIAL.md).

The GEMM and SYRK tiling is ported from portBLAS and stays under Apache-2.0. [NOTICE.md](NOTICE.md) lists it with the other third-party parts, and states which earlier commits carried the MIT License and the Apache License 2.0.
