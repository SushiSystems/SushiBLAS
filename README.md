# SushiBLAS

[![CI](https://github.com/SushiSystems/SushiBLAS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SushiSystems/SushiBLAS/actions/workflows/ci.yml)

SushiBLAS is a C++17 tensor and BLAS library for programs that run on a CPU or a GPU through
SYCL: operations are recorded on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s
task graph and run by its kernels, with no vendor math library behind them.

```cpp
sr::Runtime runtime = sr::Runtime::create().value();
sb::Engine engine(runtime, sb::Core::Layout::ColumnMajor);
sr::Graph graph = runtime.graph();
sb::LateCountHold late_counts(runtime);
sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                         engine.mode());

auto A = engine.create_tensor({K_SIZE, K_SIZE});
auto B = engine.create_tensor({K_SIZE, K_SIZE});
auto C = engine.create_tensor({K_SIZE, K_SIZE});

const size_t count = static_cast<size_t>(K_SIZE * K_SIZE);
SushiBLAS::IO().from_vector(A, std::vector<float>(count, K_A_VALUE));
SushiBLAS::IO().from_vector(B, std::vector<float>(count, K_B_VALUE));

SushiBLAS::BLASOps(recorder).gemm(A, B, C);
graph.run().value();

const std::vector<float> result = SushiBLAS::IO().to_vector<float>(C);
```

Source: `applications/examples/quickstart_demo.cpp`

`gemm` records a task and runs nothing; `graph.run()` runs it, and `to_vector` copies `C` to the
host. [A first program](docs/getting_started/FIRST_PROGRAM.md) goes through it line by line.

## Build

```bash
hub install-cli sushiblas   # puts sb on the PATH, editable against this checkout
sb setup                    # provisions the toolchain and the libraries the build needs
sb build                    # release build
sb test                     # the small and medium tests
```

[Building](docs/getting_started/BUILDING.md) gives the requirements, a CMake build without the
CLI and an NVIDIA GPU target.

## Status

SushiBLAS has no tagged release; it is at 0.1.0, below 1.0.0, so a MINOR release may break its
public interface. Dense BLAS levels 1 to 3, sparse BLAS over CSR and the elementwise, non-linear,
logic, reduction, random and spatial operations work, each with a reference path and a SYCL
kernel. GPU coverage and performance are not yet measured, and
[`KNOWN_ISSUES.md`](docs/reference/KNOWN_ISSUES.md) lists the limits. What is left is in
[`REMAINING_WORK.md`](docs/design/REMAINING_WORK.md).

## Documentation

The manual starts at [docs/README.md](docs/README.md). Contributors read
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

## Licence

SushiBLAS is free for non-commercial use under the PolyForm Noncommercial License 1.0.0
([LICENSE](LICENSE)); commercial use needs a licence from Sushi Systems
([COMMERCIAL.md](COMMERCIAL.md)). The GEMM and SYRK tiling is ported from portBLAS under
Apache-2.0; [NOTICE.md](NOTICE.md) lists it with the other third-party parts.
