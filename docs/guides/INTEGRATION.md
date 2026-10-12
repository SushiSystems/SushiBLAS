# Integrating SushiBLAS

How to consume SushiBLAS from another project, and the one fact that decides what you need to
set up. SushiBLAS is below 1.0.0: read [the stability notice](../../README.md#stability) before
you depend on it, and [section 5](#5-write-one-adapter) for how to keep a break in one place.

## 1. Where the kernels are compiled

SushiBLAS's kernels are compiled into the library. They are not header templates instantiated
in your translation units, which is the opposite of SushiRuntime. When you write

```cpp
SushiBLAS::BLASOps(engine).gemm(A, B, C);
```

the call resolves to a function already compiled into the `sushiblas` static library. Your
compiler does not
see the GEMM kernel's source, instantiates no device lambda and produces no device code for it.
`Level3::gemm` is an ordinary library call that records a task and returns a `TaskHandle`; the
task runs when you call `engine.execute()`.

The exception is the `SushiBLAS::Dispatch` seams: `execute_binary`, `execute_gemm` and the
others listed in
[the architecture, section 6](../architecture/DISPATCH_SEAMS.md).
They are templates that inline your functor into a kernel, so a call to one does instantiate
device code in your translation unit. The GEMM kernels are installed under
`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/` for that reason.

You still need a SYCL-capable compiler, because the public headers include `<sycl/sycl.hpp>`
and use `sycl::queue`, `sycl::event` and `sycl::device` in their own declarations:
`Tensor::device()` and the task bodies in `engine/task_recorder.hpp` are two places. A
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

# In your project (CMAKE_PREFIX_PATH names the prefix of SushiBLAS and the
# prefix of SushiRuntime)
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

When SushiBLAS built SushiRuntime as a subdirectory, `cmake --install` on SushiBLAS's build tree
installs SushiRuntime into the same prefix, and one prefix is enough. `sb test --suite package`
installs that way.

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
- C++17 or later. SushiBLAS is built at C++17, as SushiRuntime is, and `tests/package/` sets
  its consumer to C++20. The spans in the public headers are `SushiRuntime::span`, which is one
  class template at every standard; at C++20 it also converts from and to `std::span`.
- A `CMAKE_PREFIX_PATH` that names both install prefixes, SushiBLAS's and SushiRuntime's,
  unless both were installed to one prefix.
- SushiRuntime's shared library on the loader path at run time, and whatever your SYCL
  implementation needs. SushiRuntime's README has the notes on redistribution.

## 4. Checking the binary you linked

```cpp
#include <SushiBLAS/core/version.hpp>

std::printf("Built against SushiBLAS headers %s\n", SUSHIBLAS_VERSION_STRING);
```

`tests/package/main.cpp` prints the same macro in its last line.

SushiRuntime is a `SHARED` library, so the headers a program compiled against and the library
it loads can disagree. SushiBLAS is `STATIC`: no separately loaded file exists whose version
could differ from `SUSHIBLAS_VERSION_STRING` at run time, so the library has no runtime version
check and needs none.

You can also branch at compile time:

```cpp
#if SUSHIBLAS_VERSION >= SUSHIBLAS_VERSION_ENCODE(0, 2, 0)
    // code that needs 0.2.0
#endif
```

Both macros are in `modules/foundation/core/include/SushiBLAS/core/version.hpp`.

## 5. Write one adapter

Keep every include of a SushiBLAS header in one module of your own, and give the rest of your
program functions in your own types. A release that changes a name, a signature or an error
channel is then an edit to that module. SushiAI does this for the failure channels in
`sushiai/modules/core/core/`, in `seam.hpp` and `seam.cpp`; the blocks below are from those two
files.

**What it includes.** `seam.hpp` includes `<SushiBLAS/engine/engine.hpp>` and
`<SushiRuntime/sushiruntime.hpp>`; `seam.cpp` adds `<SushiBLAS/core/errors.hpp>` for
`UnsupportedExecution`. No other SushiAI file reads a `SushiRuntime::Error` or catches a
SushiBLAS exception.

**What it exposes.** Functions that return the host's own types. In SushiAI they return its
`Status` or its `Result`, and take a `SeamPhase` that says what the call was doing:

```cpp
[[nodiscard]] Status status_of_error(const SushiRuntime::Error& error,
                                     SeamPhase phase) noexcept;

[[nodiscard]] Status status_of_current_exception(SeamPhase phase) noexcept;

template<typename Call>
[[nodiscard]] Status guard(SeamPhase phase, Call&& call) noexcept;

[[nodiscard]] Status run_recorded(SushiBLAS::Engine& engine) noexcept;
```

Source: `sushiai/modules/core/core/include/SushiAI/core/seam.hpp`, without the comments.

**How it turns a failure into the host's type.** SushiBLAS has two channels the adapter must
close, and [the architecture, section 11](../architecture/EXCEPTIONS.md) lists what arrives on
each. A record call and a creation throw. The adapter catches the three types most derived first:

```cpp
Status status_of_current_exception(SeamPhase phase) noexcept
{
    try
    {
        throw;
    }
    catch (const SushiBLAS::UnsupportedExecution& error)
    {
        return failure_of(StatusCode::Unsupported, error.what());
    }
    catch (const std::invalid_argument& error)
    {
        return failure_of(StatusCode::InvalidDescription, error.what());
    }
    catch (const std::runtime_error& error)
    {
        return status_of_runtime_error(error, phase);
    }
    catch (const std::bad_alloc& error)
    {
        return failure_of(StatusCode::OutOfMemory, error.what());
    }
    catch (const std::exception& error)
    {
        return failure_of(StatusCode::DeviceFailed, error.what());
    }
    catch (...)
    {
        return failure_of(StatusCode::DeviceFailed, K_UNKNOWN_EXCEPTION_MESSAGE);
    }
}
```

A run returns a `SushiRuntime::Result<void>`. The adapter reads the `Error` out of it and maps
the `ErrorCode`, inside the same guard, so a run has one return type whichever channel failed:

```cpp
Status run_recorded(SushiBLAS::Engine& engine) noexcept
{
    return guard(SeamPhase::Execute, [&]()
    {
        const SushiRuntime::Result<void> ran = engine.execute();
        if (!ran)
            return status_of_error(ran.error(), SeamPhase::Execute);

        return Status();
    });
}
```

Source of both: `sushiai/modules/core/core/source/seam.cpp`. `status_of_error` in the same file
is a `switch` over the `SushiRuntime::ErrorCode` values with no `default`, so `-Wswitch` reports
a code the function does not name.

An adapter for the rest of the surface follows the same rule: it names `Engine`, the views and
`Tensor` inside, and hands out handles and functions of the host.

## 6. Troubleshooting

**A parse error inside `<sycl/sycl.hpp>` or `<SushiBLAS/tensor/tensor.hpp>`.** A compiler without SYCL
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
alias, and not a bare `sushiblas` target.
