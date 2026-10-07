# SushiBLAS

SushiBLAS is a C++17 tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. You create tensors through an `Engine` and record operations on them through view objects built from it; the runtime tracks the dependencies between the recorded tasks and runs them on a CPU or a GPU through SYCL. No kernel calls a vendor math library: there is no oneMKL and no cuBLAS behind it.

The v0.1.0 interface described here has been written and not yet built or run as a whole. This page says what the code does as written; it does not say that an operation has been verified or how fast it is.

The manual is in [docs/README.md](docs/README.md).

- **Operations are recorded, then run.** `Engine::create_tensor()` allocates a `Tensor`, which is shape, stride and dtype metadata over a ref-counted `Storage` backed by SushiRuntime USM. Calling an operation, such as `SushiBLAS::BLASOps(engine).axpy(...)`, records one task on the engine's graph and returns a `TaskHandle`. `engine.execute()` compiles the graph, runs it and returns when every task has finished.
- **Eight operation families.** Dense BLAS levels 1 to 3 with the band, packed, rotation, complex and Hermitian routines; sparse BLAS over CSR; elementwise arithmetic; non-linear activations; logic and comparison; reductions; random generation; and spatial (windowed) operations. [The architecture, section 7](docs/architecture/ARCHITECTURE.md#7-kernels-reference-and-sycl) lists the routines.
- **Every operation has a reference path and a SYCL kernel.** The BLAS, sparse and spatial families have a plain C++ reference kernel under `kernels/reference/` beside the SYCL kernel under `kernels/sycl/`. The other families run one functor two ways: inlined into a SYCL kernel, or in a serial host loop. The GEMM and SYRK tiling is ported from portBLAS; see [NOTICE.md](NOTICE.md).
- **The execution mode is stamped on each task as it is recorded.** `Engine::set_mode()` selects `AUTO`, `FORCE_SYCL` or `FORCE_REFERENCE` for every later operation. See [the architecture, section 3](docs/architecture/ARCHITECTURE.md#3-execution-modes).
- **A wrong call throws when it is recorded.** A wrong operand, flag or scalar throws `std::invalid_argument` and names the operation and the operand; a dtype with no meaning for an operation, such as `max` on a complex tensor, is one of those. An operation that cannot run under the mode or on the device throws `SushiBLAS::UnsupportedExecution`, which derives from it. See [the architecture, section 11](docs/architecture/ARCHITECTURE.md#11-exceptions).
- **Fusion is written by the caller.** The `SushiBLAS::Dispatch` seams take a functor and run it as one kernel, and `Dispatch::execute_gemm` takes an epilogue applied to each element of the product. No pass fuses recorded tasks automatically.

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
SushiBLAS::Engine engine(ctx, SushiBLAS::Core::Layout::ROW_MAJOR);

auto A = engine.create_tensor({1024, 1024});   // FLOAT32, shared USM, by default
auto B = engine.create_tensor({1024, 1024});
auto C = engine.create_tensor({1024, 1024});

SushiBLAS::RandomOps random(engine);
random.uniform(A, 0.0, 1.0);
random.uniform(B, 0.0, 1.0);

SushiBLAS::BLASOps blas(engine);
blas.gemm(A, B, C);        // records the task; nothing has run yet
engine.execute();          // compiles the graph, runs it, returns when every task has finished

const std::vector<float> result = SushiBLAS::IO(engine).to_vector<float>(C);
```

`Engine` owns the graph and has no accessor per operation family. A family is a view class built from the engine: `BLASOps`, `SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`, `SpatialOps` and `IO`. A view holds a reference and no other state. Each record call returns a `TaskHandle`, and nothing runs until `engine.execute()`.

```cpp
using namespace SushiBLAS;

ElementwiseOps(engine).add(A, B, C);             // C = A + B
NonLinearOps(engine).relu(A);                    // in place
LogicOps(engine).greater(A, B, mask);            // mask = 1 where A > B, else 0
ReductionOps(engine).sum(A, total);              // total = sum(A)

BLASOps blas(engine);
blas.axpy(2.0, x, y);                                        // y = 2*x + y
blas.gemv(A, x, y, Core::Transpose::TRANS, 1.0, 1.0);        // y = A^T x + y
blas.trsm(A, B, Core::Side::LEFT, Core::Uplo::LOWER);        // solve A X = B, X replaces B

engine.execute();
```

The BLAS flags are the enumerations `Core::Transpose`, `Core::Uplo`, `Core::Diag` and `Core::Side`, and a scalar argument is a `BlasScalar`, which converts from a `double` or a `std::complex<double>`.

`Tensor` is a metadata view (shape, strides, rank, `Core::DataType`, layout) over a ref-counted `Storage`. `transpose()`, `reshape()` and `slice()` return new `Tensor` views over the same storage and move no data. A recorded task keeps the storage of its operands alive until the graph drops it. [The architecture](docs/architecture/ARCHITECTURE.md) describes each layer.

## Project layout

```
include/SushiBLAS/     Public headers
  SushiBLAS.h           The umbrella header
  engine.hpp            Engine: owns the scratch pool, the graph, the recorder, the
                         tensor factory and the random stream; create_tensor(),
                         set_mode(), execute()
  engine/               The recorder, the seams and the operation views
    task_recorder.hpp     TaskRecorder: the one caller of TaskGraph::add_task
    operand_set.hpp       OperandSet: what a task reads, writes and retains
    task_handle.hpp       TaskHandle: what a record call returns
    execution_mode.hpp    ExecutionMode: AUTO, FORCE_SYCL, FORCE_REFERENCE
    kernel_pair_dispatcher.hpp
                          The mode switch a task body calls
    task.hpp              execute_typed_task: one dtype-dispatched kernel-pair task
    dispatch.hpp          The Domain vocabulary shared by the Dispatch seams
    operand_check.hpp     The record-time operand checks
    rng_stream.hpp        RngStream: the seed and the offset counter
    scratch.hpp, scratch_pool.hpp, scratch_shelf.hpp
                          ScratchBuffer, the lease, and the pool it comes from
    blas/                 Level1, Level2, Level3, BlasScalar (scalar.hpp), the operand
                           describers, utils.hpp (execute_blas_task) and
                           dispatch.hpp (execute_gemm)
    sparse/               The record-time checks of the sparse operands
    logic/                LogicOps and its dispatch.hpp
    math/                 ElementwiseOps, NonLinearOps, ReductionOps, RandomOps,
                           their dispatch.hpp files and safe_math.hpp
    spatial/              window.hpp
    blas.hpp, sparse.hpp, spatial.hpp
                          BLASOps, SparseOps and SpatialOps
  kernels/              The kernels and access policies a public seam instantiates in
                         a consumer's translation unit, and which are therefore
                         installed: GEMM's reference and SYCL kernels, the packed
                         product and the epilogue headers
  support/              The layer below kernels and engine: launch_1d, serial_1d,
                         reduce_1d, reduce_axis, serial_reduce, device_policy,
                         accumulator, reference_accumulator, spatial_geometry,
                         sparse_assembly and three smaller headers
  core/                 common.hpp (DataType, Layout), blas_types.hpp (Transpose,
                         Uplo, Diag, Side), sparse_types.hpp (IndexBase), errors.hpp
                         (UnsupportedExecution), logger.hpp, format.hpp, traits.hpp,
                         export.hpp
  experimental/         Quarantined FFT and LAPACK declarations
  tensor.hpp            Tensor: shape, stride and dtype metadata over a Storage
  storage.hpp           Storage: ref-counted USM allocation, aligned to 128 bytes
  tensor_factory.hpp    TensorFactory: allocates dense tensors from one context
  sparse_matrix.hpp, sparse_vector.hpp, sparse_builder.hpp
                        The CSR matrix, the sparse vector and the builder of both
  io.hpp                Host transfer, save, load and print (native, .npy, raw binary)
  version.hpp           SUSHIBLAS_VERSION_* macros
src/                    Implementations (engine/<domain>/<op>.cpp, engine.cpp, io.cpp,
                         sparse_builder.cpp)
kernels/                Private kernels, never installed
  reference/            Plain C++ kernels
  sycl/                 SYCL kernels
examples/               example.cpp, io_example.cpp
benchmarks/             bench_*.cpp, built with sb build --benchmarks
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
