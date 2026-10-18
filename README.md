# SushiBLAS

[![CI](https://github.com/SushiSystems/SushiBLAS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SushiSystems/SushiBLAS/actions/workflows/ci.yml)

SushiBLAS is a C++17 tensor and BLAS library for programs that run on a CPU or a GPU through
SYCL: operations are recorded on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s
task graph and run by its kernels, with no vendor math library behind them.

```cpp
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
```

Source: `applications/examples/quickstart_demo.cpp`

`gemm` records a task and runs nothing; `to_vector` runs the graph and copies `C` to the host.
[A first program](docs/getting_started/FIRST_PROGRAM.md) goes through it line by line.

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

- No release is tagged; the version in `CMakeLists.txt` is 0.1.0, and a MINOR release may break
  the public interface. [The integration guide](docs/guides/INTEGRATION.md#stability) says how
  to depend on it.
- Dense BLAS levels 1 to 3, sparse BLAS over CSR, elementwise, non-linear, logic, reduction,
  random and spatial operations each have a reference path and a SYCL kernel.
- The tests run on CPU devices in CI. On a GPU the suite has run on one card, an RTX 3080 Ti, and
  no speed figure is measured. [The known issues](docs/reference/KNOWN_ISSUES.md) list the rest.
- What is left is in [the backlog](docs/design/REMAINING_WORK.md).

## Documentation

The manual starts at [docs/README.md](docs/README.md). Contributors read
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

## Licence

SushiBLAS is free for non-commercial use under the PolyForm Noncommercial License 1.0.0
([LICENSE](LICENSE)); commercial use needs a licence from Sushi Systems
([COMMERCIAL.md](COMMERCIAL.md)). The GEMM and SYRK tiling is ported from portBLAS under
Apache-2.0; [NOTICE.md](NOTICE.md) lists it with the other third-party parts.
