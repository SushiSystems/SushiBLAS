# Integrating SushiBLAS

How to consume SushiBLAS from another project, and the one fact that decides what you need to
set up.

## 1. Where the kernels are compiled

SushiBLAS's kernels are compiled into the library. They are not header templates instantiated
in your translation units, which is the opposite of SushiRuntime. When you write

```cpp
SushiBLAS::BLASOps(engine).gemm(A, B, C);
```

the call resolves to a function already compiled into `libsushiblas.a`. Your compiler does not
see the GEMM kernel's source, instantiates no device lambda and produces no device code for it.
`Level3::gemm` is an ordinary library call that records a task and returns a `TaskHandle`; the
task runs when you call `engine.execute()`.

The exception is the `SushiBLAS::Dispatch` seams: `execute_binary`, `execute_gemm` and the
others listed in
[the architecture, section 6](../architecture/DISPATCH_SEAMS.md).
They are templates that inline your functor into a kernel, so a call to one does instantiate
device code in your translation unit. The GEMM kernels are installed under
`include/SushiBLAS/kernels/` for that reason.

You still need a SYCL-capable compiler, because the public headers include `<sycl/sycl.hpp>`
and use `sycl::queue`, `sycl::event` and `sycl::device` in their own declarations:
`Tensor::get_device()` and the task bodies in `engine/task_recorder.hpp` are two places. A
translation unit that includes `<SushiBLAS/SushiBLAS.h>` needs a compiler that parses those
types, even though it submits no kernel of its own. `sushiblas_apply_sycl()` in
`cmake/SyclTarget.cmake` therefore puts `-fsycl` on `SushiBLAS::sushiblas` as a `PUBLIC` compile
and link option.

You do not need the two-lane build split that SushiRuntime's own
[INTEGRATION.md](https://github.com/SushiSystems/SushiRuntime/blob/main/docs/guides/INTEGRATION.md)
describes. Every translation unit that touches `SushiBLAS::sushiblas` needs the SYCL compiler,
whether or not it writes a kernel.

## 2. Consuming the package

### From an installed prefix

```bash
# In SushiBLAS's tree
sb build
cmake --install build --prefix /opt/sushiblas

# In your project (CMAKE_PREFIX_PATH needs BOTH SushiBLAS and its
# SushiRuntime dependency's install prefixes)
cmake -S . -B build \
    -DCMAKE_PREFIX_PATH="/opt/sushiblas;/opt/sushiruntime" \
    -DCMAKE_CXX_COMPILER=<the same SYCL compiler SushiBLAS was built with>
```

```cmake
find_package(SushiBLAS REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SushiBLAS::sushiblas)
```

`tests/package/` in this repository is a working example: a two-file consumer with no access to
SushiBLAS's source tree. It also runs the flag assertion of
[section 3](#3-what-you-inherit-and-what-you-must-supply).

### From a sibling checkout

`add_subdirectory(path/to/sushiblas)` gives you the same `SushiBLAS::sushiblas` target. The
[D3 layered resolution](../architecture/RUNTIME_DEPENDENCY.md) that
SushiBLAS uses for its SushiRuntime dependency applies one level up in the same way.

### Version compatibility

The package is written with `SameMinorVersion` compatibility while the project is below 1.0:
`find_package(SushiBLAS 0.1 REQUIRED)` accepts 0.1.x and rejects 0.2.0. At the v1.0.0 freeze
this becomes `SameMajorVersion`.

## 3. What you inherit, and what you must supply

Linking `SushiBLAS::sushiblas` gives you:

| Inherited | Why |
| --- | --- |
| Include directories | The target exports them |
| The library itself | It is the target |
| `-fsycl`, on the intel-llvm toolchain | The public headers need a SYCL-aware compiler to parse `sycl::event` and `sycl::device`; see section 1 |
| SushiRuntime's floating-point contract, transitively | SushiBLAS links `SushiRuntime::SushiRuntime` `PUBLIC`, so its `sushiruntime_deterministic_fp` interface (`-ffp-contract=off -fno-fast-math`, or `/fp:precise`) reaches you too |

You do not inherit SushiBLAS's warnings, `-Werror`, sanitizer selection or `-march`. Those are
this project's development choices, and a dependency that imposed them would fail your build
for a reason your own source tree does not show. `tests/package/CMakeLists.txt` asserts this
from the consumer side: it checks `INTERFACE_COMPILE_OPTIONS` for `-Werror`, `-Wall`, `-march=`
and `-fsanitize`, and `INTERFACE_LINK_LIBRARIES` for the names of the in-tree policy targets.

You must supply:

- The SYCL compiler SushiBLAS was built with. Mixing toolchains between the library and a
  consumer is not supported.
- C++17 or later. SushiBLAS is built at C++17, as SushiRuntime is; a consumer on C++20 is
  supported, and `tests/package/` compiles as one. A consumer whose standard disagrees with the
  SushiRuntime it links is not supported: `SushiRuntime::span` is `std::span` at C++20 and
  SushiRuntime's own type below it, and the difference shows in the mangled names of
  SushiRuntime's compiled ABI.
- A `CMAKE_PREFIX_PATH` that names both install prefixes, SushiBLAS's and SushiRuntime's,
  unless both were installed to one prefix.
- SushiRuntime's shared library on the loader path at run time, and whatever your SYCL
  implementation needs. SushiRuntime's README has the notes on redistribution.

## 4. Checking the binary you linked

```cpp
#include <SushiBLAS/version.hpp>

std::printf("Built against SushiBLAS headers %s\n", SUSHIBLAS_VERSION_STRING);
```

SushiRuntime is a `SHARED` library, so the headers a program compiled against and the library
it loads can disagree. SushiBLAS is `STATIC`: no separately loaded file exists whose version
could differ from `SUSHIBLAS_VERSION_STRING` at run time, so the library has no runtime version
check and needs none.

You can also branch at compile time:

```cpp
#if SUSHIBLAS_VERSION >= SUSHIBLAS_VERSION_ENCODE(0, 2, 0)
    // use something introduced in 0.2.0
#endif
```

## 5. Troubleshooting

**A parse error inside `<sycl/sycl.hpp>` or `<SushiBLAS/tensor.hpp>`.** A compiler without SYCL
support is compiling the translation unit. Every file that includes a SushiBLAS header needs
the SYCL-aware compiler; see section 1.

**`find_package(SushiBLAS)` succeeds and the configure then fails on SushiRuntime.**
`CMAKE_PREFIX_PATH` must name both install prefixes. `SushiBLASConfig.cmake` calls
`find_dependency(SushiRuntime)`, so a missing SushiRuntime prefix shows up as a SushiRuntime
configure error.

**A build failure that mentions `-Werror`, `-march` or `sushiblas_warnings`.** SushiBLAS's
private development flags leaked into the export set, which is a defect in SushiBLAS.
`tests/package/` asserts against it; please report it.

**Link errors on a BLAS symbol.** Confirm that you link `SushiBLAS::sushiblas`, the namespaced
alias, and not a bare `sushiblas` target from an `add_subdirectory` older than the packaging
work of SB-6.
