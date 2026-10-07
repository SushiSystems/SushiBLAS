# Frequently asked questions

Short answers for a developer meeting SushiBLAS for the first time: what it is, what it needs,
how to build and consume it, where it runs, what is unfinished, how it is licensed and where to
report a problem. Each answer links to the manual page that holds the detail.

## What is SushiBLAS?

SushiBLAS is a C++17 tensor and BLAS library built on SushiRuntime's task graph. You create
tensors and queue operations on them through an `Engine`; the runtime tracks the dependencies
between the queued tasks and runs them on a CPU or a GPU through SYCL. No kernel calls a vendor
math library: there is no oneMKL and no cuBLAS behind it.

Operations are recorded, then run. A call such as `SushiBLAS::BLASOps(engine).gemm(A, B, C)`
records a task and runs nothing; `engine.execute()` compiles and runs the graph. `Engine` has no
accessor per operation family. A family is a view class built from the engine: `BLASOps`,
`SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`,
`SpatialOps` and `IO`. The [root README](../../README.md#how-it-works) has a complete example, and
[the architecture](../architecture/ARCHITECTURE.md) describes each layer.

## What do I need to build SushiBLAS?

The [README's requirements](../../README.md#requirements) list five things:

- A SYCL 2020 compiler: the intel-llvm `clang++ -fsycl` that `sb setup` or SushiStack provisions.
- CMake 3.20 or newer, and 3.22 for the test suite.
- GoogleTest for the test suite.
- A SushiRuntime checkout or installed package.
- C++17 or later. The library is built at C++17; a C++20 consumer is supported.

SushiBLAS selects no toolchain of its own. It builds with the one in the workspace
`dependencies/` tree or under `~/.sushisystems`.

## How do I install SushiBLAS and its toolchain?

SushiStack's bootstrap script installs Python and Git, installs `hub`, then runs `hub install` to
fetch every toolchain and library the workspace's modules declare. SushiBLAS declares its own in
`cli/sushistack.deps.toml`.

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

On Windows, run `irm https://sushisystems.io/install.ps1 | iex` in PowerShell. In PowerShell
`curl` is an alias for `Invoke-WebRequest` and does not pipe a script the same way.

To use an existing checkout of the repository, link it into a workspace with `hub link`.
`hub doctor` and `hub status` show what is installed. See the
[README's quick start](../../README.md#quick-start).

## How do I build SushiBLAS and run its tests?

Install the CLI through `hub`, run `sb setup` once before the first build, then build and test:

```bash
hub install-cli sushiblas
sb setup
sb build
sb test
```

`hub install-cli sushiblas` puts two equivalent commands on the PATH, `sb` and `sushiblas`.
`sb build` makes a release build by default; `--type debug` and `--type relwithdebinfo` select
the others, and `--clean` wipes the build tree first. `sb test` runs the functional suite by
default, and `--suite` selects `unit`, `integration`, `regression`, `all` or `package`.

Inside a SushiStack workspace, or with a SushiRuntime checkout beside this one, no local
configuration is needed. The [README](../../README.md#building) also shows a build with CMake
directly, and [the CLI's README](../../cli/README.md) describes every command.

## How do I use SushiBLAS from another CMake project?

Call `find_package(SushiBLAS REQUIRED)` and link `SushiBLAS::sushiblas`:

```cmake
find_package(SushiBLAS REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SushiBLAS::sushiblas)
```

`CMAKE_PREFIX_PATH` must name both install prefixes, SushiBLAS's and SushiRuntime's, unless both
were installed to the same prefix. You must also supply the same SYCL compiler SushiBLAS was
built with, C++17 or later, and SushiRuntime's shared library on the loader path at run time.
`add_subdirectory(path/to/sushiblas)` gives the same `SushiBLAS::sushiblas` target.

While the project is pre-1.0 the package uses `SameMinorVersion` compatibility:
`find_package(SushiBLAS 0.1 REQUIRED)` accepts 0.1.x and rejects 0.2.0. `tests/package/` in the
repository is a complete consumer. The [integration guide](INTEGRATION.md) covers what a consumer
inherits, what it must supply and how to read common build errors.

## Does my own code need a SYCL compiler if I write no kernel?

Yes. The kernels behind the view methods are compiled into the library, so a call such as
`BLASOps::gemm` produces no device code in your translation unit. A call to a
`SushiBLAS::Dispatch` seam does, because the seam inlines your functor into a kernel. The public
headers include `<sycl/sycl.hpp>` and use `sycl::event` and `sycl::device` in their declarations
in either case, so every translation unit that
includes `<SushiBLAS/SushiBLAS.h>` needs a compiler that can parse those types. Linking
`SushiBLAS::sushiblas` carries `-fsycl` as a `PUBLIC` compile and link option for that reason.

A parse error inside `<sycl/sycl.hpp>` or `<SushiBLAS/tensor.hpp>` means the file is being
compiled by a non-SYCL compiler. See
[section 1 of the integration guide](INTEGRATION.md#1-the-fact-that-shapes-everything).

## Which devices does SushiBLAS run on?

By default the device image is SPIR-V, compiled by the runtime for the device it finds. That
reaches every OpenCL and Level Zero device and no NVIDIA GPU, because NVIDIA's OpenCL driver
cannot ingest SPIR-V. For an NVIDIA GPU the CUDA image is compiled ahead of time:

```bash
sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
```

Keep `spir64` in the list to keep the CPU device reachable from the same build. The compiler must
be an intel-llvm built with `--cuda`; a stock oneAPI release has no CUDA backend and rejects the
triple. intel-llvm documents support for `sm_75` and newer.

None of the GPU path has run on hardware yet. See
[Targeting an NVIDIA GPU](../../README.md#targeting-an-nvidia-gpu) in the README.

## How does SushiBLAS relate to SushiRuntime and SushiStack?

SushiBLAS owns no execution engine, no scheduler and no memory pool; all three are
SushiRuntime's. SushiBLAS adds a domain layer on top: the `Tensor` abstraction, the `Engine`
that owns one graph, the operation views that record tasks on it, and the kernels. It does not vendor
SushiRuntime. The build looks for it in three places, in order: a target a superproject already
brought in, an installed package through `find_package(SushiRuntime)`, then a sibling
`../sushiruntime` checkout. [Section 8 of the architecture](../architecture/ARCHITECTURE.md#8-the-sushiruntime-dependency-d3)
describes the resolution.

SushiStack is the workspace: a folder that holds several module checkouts and one `dependencies/`
tree. Its CLI, `hub`, provisions the toolchains and libraries of a workspace. `sb` is this
repository's own CLI. The [glossary](../reference/GLOSSARY.md) defines these words.

## What does SushiBLAS not do yet?

- No release is tagged. The version in the build files is 0.1.0, and the project is pre-1.0.
- None of the GPU path has run on hardware.
- The v0.1.0 interface has been written and not yet built or run as a whole.
- `trmm` is not implemented. Sparse matrices are CSR only.
- `LinalgOps` (SVD, QR, eigendecomposition) and `TransformsOps` (FFT, DFT) are declared under
  `include/SushiBLAS/experimental/` and have no implementation. The umbrella header does not
  include them.

Defects that are recorded and not yet fixed are in [known issues](../reference/KNOWN_ISSUES.md).
The [changelog](../reference/CHANGELOG.md) lists what has changed.

## How is SushiBLAS licensed?

SushiBLAS is source-available, free for non-commercial use under the PolyForm Noncommercial
License 1.0.0. [`LICENSE`](../../LICENSE) is the binding text and lists the permitted purposes.

Any commercial purpose needs a separate, paid licence from Sushi Systems. That includes use
inside a company, use in paid client work, and use in a product or service that is sold. To ask
for one, write to hello@sushisystems.io with what you want to build and who will use it; see
[`COMMERCIAL.md`](../../COMMERCIAL.md).

The GEMM and SYRK tiling is ported from portBLAS and stays under Apache-2.0.
[`NOTICE.md`](../../NOTICE.md) lists it with the other third-party parts.

## Where do I report a problem?

For anything that is not sensitive, open a GitHub issue or Discussion.

Do not open a public issue for a security problem. Report it privately by email to
mustafagarip@sushisystems.io, to the maintainer on the community Discord server, or through
GitHub's private vulnerability reporting under the repository's Security tab. The
[security policy](../reference/SECURITY.md) says what to include and what is in scope; it
promises an acknowledgement within 72 hours and an initial assessment within 7 days. Security
fixes go to the latest `main`, so test against the current `main` before reporting.

Conduct in project spaces is covered by the
[code of conduct](../reference/CODE_OF_CONDUCT.md), which names the same email address and
Discord server for reports.
