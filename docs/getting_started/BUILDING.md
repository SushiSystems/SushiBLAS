# Building SushiBLAS

What the build needs, how to install the toolchain and the CLI, and how to build and test.

## Requirements

- A SYCL 2020 compiler: the intel-llvm `clang++ -fsycl` that `sb setup` or SushiStack provisions. SushiBLAS selects no toolchain of its own; it builds with the one in the workspace `dependencies/` tree or under `~/.sushisystems`.
- CMake 3.20 or newer, and 3.22 for the test suite's `gtest_discover_tests`.
- GoogleTest for the test suite.
- The SushiRuntime, SushiLog and SushiProfiler packages, which `sb build` builds or reuses at the versions `cli/sushistack.deps.toml` pins ([section 8 of the architecture](../architecture/RUNTIME_DEPENDENCY.md)).
- C++17 or later. The library is built at C++17; a C++20 consumer is supported.

## Installing

SushiStack's bootstrap script installs Python and Git, installs `hub`, then runs `hub install` to fetch every toolchain and library the workspace's modules declare. SushiBLAS declares its own in `cli/sushistack.deps.toml`.

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

On Windows use `irm`. In PowerShell `curl` is an alias for `Invoke-WebRequest` and does not pipe a script the same way.

To use an existing checkout of this repository, link it into a workspace with `hub link`; `hub doctor` and `hub status` show what is installed. The [SushiStack](https://sushisystems.io) documentation covers both.

## Using SushiBLAS from another project

Call `find_package(SushiBLAS REQUIRED)` and link `SushiBLAS::sushiblas`. Read the [integration guide](../guides/INTEGRATION.md) first: the SYCL parts of a consumer's build have requirements it explains. `tests/package/` is a complete consumer.

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

`sb clean` removes the `build/` tree. Every command is described in [cli/README.md](../../cli/README.md).

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

The configure step prints the active targets. The functional suite has run on one card, an RTX 3080 Ti: on 2026-10-12, on Windows with the pinned intel-llvm toolchain, 2 207 tests ran and none failed. [The GPU validation design](../design/GPU_VALIDATION/README.md) lists what is unverified.

### Building with CMake directly

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/clang \
  -DSushiRuntime_DIR=/path/to/sushiruntime/lib/cmake/SushiRuntime \
  -DSushiLog_DIR=/path/to/sushilog/lib/cmake/SushiLog \
  -DSushiProfiler_DIR=/path/to/sushiprofiler/lib/cmake/SushiProfiler
cmake --build build --config Release
```

`CMAKE_C_COMPILER` must point at the toolchain's plain `clang` binary. A `clang++` binary compiles every file as C++ and breaks CMake's C probe; the project declares `LANGUAGES CXX C`. The three `_DIR` variables name the config folders of the installed packages; `sb build` passes them for you. `SushiProfiler_DIR` is needed only with `SB_ENABLE_PROFILER` on, which is the default.

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
sb run sushiblas_blas_tests -- --gtest_filter='Medium_GEMMTest.*'
```
