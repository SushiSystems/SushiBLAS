# Integrating SushiBLAS

How to consume SushiBLAS from another project, and the one fact that decides what you need to
set up. SushiBLAS is below 1.0.0: read [the stability notice](#stability) before
you depend on it, and [section 5](#5-write-one-adapter) for how to keep a break in one place.

## 1. Where the kernels are compiled

SushiBLAS's kernels are compiled into the library. They are not header templates instantiated
in your translation units, which is the opposite of SushiRuntime. When you write

```cpp
    SushiBLAS::BLASOps(recorder).gemm(A, B, C);
```

Source: `applications/examples/quickstart_demo.cpp`

the call resolves to a function already compiled into the `sushiblas` static library. Your
compiler does not
see the GEMM kernel's source, instantiates no device lambda and produces no device code for it.
`Level3::gemm` is an ordinary library call that records a task and returns a `TaskHandle`; the
task runs when you run the graph the recorder records into, `graph.run()`.

The exception is the `SushiBLAS::Dispatch` seams: `execute_binary`, `execute_gemm` and the
others listed in
[the architecture, section 6](../architecture/DISPATCH_SEAMS.md).
They are templates that inline your functor into a kernel, so a call to one does instantiate
device code in your translation unit. The GEMM kernels are installed under
`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/` for that reason.

You still need the SYCL toolchain's compiler and headers, because the public headers include
`<sycl/sycl.hpp>` and use `sycl::queue`, `sycl::event` and `sycl::device` in their own
declarations: `Tensor::device()` and the task bodies in `engine/task_recorder.hpp` are two
places. A translation unit that includes `<SushiBLAS/SushiBLAS.hpp>` parses those types. It
compiles for the device only when it instantiates a kernel.

`cmake/SyclTarget.cmake` draws that line on the intel-llvm toolchain. `SushiBLAS::sushiblas`
gives you the SYCL headers, through `SushiBLAS::sycl_headers`, and `-fsycl` with the device
targets as link options, so the link of your executable pulls the device images out of the
archive. It gives you no device compile option. A translation unit that calls a dispatch seam or
a kernel template compiles with the options of `SushiBLAS::device_code`: link that target to a
target whose every unit instantiates a kernel, or put its options on the units that do.

```cmake
add_executable(my_app main.cpp fused_kernels.cpp)
target_link_libraries(my_app PRIVATE SushiBLAS::sushiblas)
set_property(SOURCE fused_kernels.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<TARGET_PROPERTY:SushiBLAS::device_code,INTERFACE_COMPILE_OPTIONS>")
```

Source: not compiled

A unit that instantiates a kernel without those options gets a warning that the
`handler::parallel_for` it calls is deprecated, an error under `-Werror`. Do not silence it: the
unit has no device code, so its kernel is missing from the device image.

You do not need the two-lane build split that SushiRuntime's own
[INTEGRATION.md](https://github.com/SushiSystems/SushiRuntime/blob/main/docs/guides/INTEGRATION.md)
describes. Every translation unit that touches `SushiBLAS::sushiblas` is compiled by the SYCL
toolchain's compiler, with `-fsycl` or without it.

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
installs SushiRuntime into the same prefix, and one prefix is enough. `sb test --package`
installs that way.

### From a sibling checkout

`add_subdirectory(path/to/sushiblas)` gives you the same `SushiBLAS::sushiblas` target. The
[D3 layered resolution](../architecture/RUNTIME_DEPENDENCY.md) that
SushiBLAS uses for its SushiRuntime dependency applies one level up in the same way.

### Version compatibility

The package is written with `SameMinorVersion` compatibility while the project is below 1.0:
`find_package(SushiBLAS 0.1 REQUIRED)` accepts 0.1.x and rejects 0.2.0. At the v1.0.0 freeze
this becomes `SameMajorVersion`.

### Stability

SushiBLAS is below 1.0.0, and its public interface is not stable.

- A MINOR release (0.2, 0.3) may change or remove any public name, signature, header path or binary layout. A PATCH release does not.
- Pin an exact tag. Do not follow a branch. No tag exists yet; the version in `CMakeLists.txt` is 0.1.0.
- Reach SushiBLAS through one adapter of your own: a single module that includes `<SushiBLAS/SushiBLAS.hpp>` and exposes only what your code needs, in your own types. When a release breaks the interface, that module is the only place you edit.
- SushiBLAS's public headers include `<SushiRuntime/sushiruntime.hpp>` and name `SushiRuntime::Runtime`, `Result` and `Residency`, so a SushiRuntime release that changes those changes this interface too. The build names no SushiRuntime version: `cmake/Runtime.cmake` takes the package or the checkout it finds.

`applications/examples/adapter/` is such an adapter in a program this repository builds: its two adapter files, `sushiblas_adapter.cpp` and `status_mapping.hpp`, are the only ones that include the library, and they turn every exception SushiBLAS throws and every `SushiRuntime::Error` a run returns into the program's own `Status`. [Section 5](#5-write-one-adapter) walks through it. SushiAI's seam (`sushiai/modules/core/core/source/seam.cpp`) does the same for the failure channels of a larger consumer.

## 3. What you inherit, and what you must supply

Linking `SushiBLAS::sushiblas` gives you:

| Inherited | Why |
| --- | --- |
| Include directories | The target exports them |
| The library itself | It is the target |
| The SYCL headers as a system include, with `SYCL_DISABLE_FSYCL_SYCLHPP_WARNING`, on the intel-llvm toolchain | A unit compiled without `-fsycl` includes `<sycl/sycl.hpp>` through the public headers; see section 1 |
| `-fsycl` and the device targets as link options, on the intel-llvm toolchain | The link of an executable pulls the device images out of the archive |
| SushiRuntime's floating-point contract, transitively | SushiBLAS links `SushiRuntime::SushiRuntime` `PUBLIC`, so its `sushiruntime_deterministic_fp` interface (`-ffp-contract=off -fno-fast-math`, or `/fp:precise`) reaches you too |

You do not inherit `-fsycl` as a compile option: a unit that instantiates a kernel takes the
options of `SushiBLAS::device_code`, as section 1 shows.

You do not inherit SushiBLAS's warnings, `-Werror`, sanitizer selection or `-march`. Those are
this project's development choices, and a dependency that imposed them would fail your build
for a reason your own source tree does not show. `tests/package/CMakeLists.txt` asserts this
from the consumer side: it checks `INTERFACE_COMPILE_OPTIONS` for `-Werror`, `-Wall`, `-march=`
and `-fsanitize`, and `INTERFACE_LINK_LIBRARIES` for the names of the in-tree policy targets. It
also checks that `-fsycl` is on `SushiBLAS::device_code` and not on `SushiBLAS::sushiblas`.

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

`<SushiBLAS/core/version.hpp>`, which the umbrella header includes, defines
`SUSHIBLAS_VERSION_STRING`. The package consumer prints it:

```cpp
std::printf("OK: SushiBLAS %s ran AXPY, a consumer-authored fused kernel, a "
            "fused in-place update and a GEMM with a consumer-authored epilogue "
            "against an installed package\n",
            SUSHIBLAS_VERSION_STRING);
```

Source: `tests/package/main.cpp`

SushiRuntime is a `SHARED` library, so the headers a program compiled against and the library
it loads can disagree. SushiBLAS is `STATIC`: no separately loaded file exists whose version
could differ from `SUSHIBLAS_VERSION_STRING` at run time, so the library has no runtime version
check and needs none.

You can also branch at compile time, with a guard such as
`#if SUSHIBLAS_VERSION >= SUSHIBLAS_VERSION_ENCODE(0, 2, 0)`. Both macros are in
`modules/foundation/core/include/SushiBLAS/core/version.hpp`.

## 5. Write one adapter

Keep every include of a SushiBLAS header in one file of your own, and give the rest of your
program functions in your own types. A release that changes a name, a signature or an error
channel is then an edit to that file. `applications/examples/adapter/` is a program built this
way, and the blocks below are from its four files;
[the examples page](../../applications/examples/README.md) says what it prints.

**What the program sees.** A header that includes the standard library alone. It declares the
program's own value type and status, a backend it names and does not define, and the operations
the program needs:

```cpp
class Backend;

// ...

[[nodiscard]] Status create_backend(BackendHandle& backend);

// ...

[[nodiscard]] Status multiply_add(Backend& backend, const Matrix& a, const Matrix& b,
                                  const Matrix& offset, Matrix& result, float& result_sum);
```

Source: `applications/examples/adapter/host_program.hpp`

`BackendHandle` is a `std::unique_ptr` with a deleter the adapter defines, so the program owns
a backend whose size it does not know.

**What the adapter includes.** The two adapter files, `sushiblas_adapter.cpp` and
`status_mapping.hpp`, include `<SushiBLAS/SushiBLAS.hpp>`, and no other file of the program
does. The first holds the operations and defines the backend as the runtime and an engine
built on it:

```cpp
class Backend final
{
    public:
        /** @brief Takes the runtime and builds the engine on it. */
        explicit Backend(SushiRuntime::Runtime&& created)
            : runtime(std::move(created)), engine(runtime)
        {

        }

        /// @brief The runtime the engine is built on; declared first, destroyed last.
        SushiRuntime::Runtime runtime;
        /// @brief The engine every operation of the host records on.
        SushiBLAS::Engine engine;
};
```

Source: `applications/examples/adapter/sushiblas_adapter.cpp`

**How it turns a failure into the host's type.** SushiBLAS has two channels the adapter must
close, and [the architecture, section 11](../architecture/EXCEPTIONS.md) lists what arrives on
each. `status_mapping.hpp` closes both. A record call and a creation throw, so the adapter runs
every operation inside one function that catches the types most derived first:

```cpp
template<typename Work>
[[nodiscard]] Status guarded(const Work& work)
{
    try
    {
        return work();
    }
    catch (const SushiBLAS::UnsupportedExecution& error)
    {
        return Status{StatusCode::Unsupported, error.what()};
    }
    catch (const std::invalid_argument& error)
    {
        return Status{StatusCode::InvalidRequest, error.what()};
    }
    catch (const std::bad_alloc& error)
    {
        return Status{StatusCode::OutOfMemory, error.what()};
    }
    catch (const std::exception& error)
    {
        return Status{StatusCode::BackendFailed, error.what()};
    }
}
```

Source: `applications/examples/adapter/status_mapping.hpp`

The last handler takes `std::runtime_error`, which a creation and a refused recording throw.

A run returns a `SushiRuntime::Result<void>`, and so does `Runtime::create()` with a `Runtime`
in it. The adapter maps the `ErrorCode` of the `Error` in a `switch` that names every value and
has no `default`, so `-Wswitch` reports a code the function does not name:

```cpp
[[nodiscard]] inline Status status_of_error(const SushiRuntime::Error& error)
{
    switch (error.code)
    {
        case SushiRuntime::ErrorCode::InvalidGraph:
        case SushiRuntime::ErrorCode::InvalidAccess:
        case SushiRuntime::ErrorCode::DanglingResource:
        case SushiRuntime::ErrorCode::InvalidState:
        case SushiRuntime::ErrorCode::InvalidArgument:
            return Status{StatusCode::InvalidRequest, error.message};
        case SushiRuntime::ErrorCode::Unsupported:
            return Status{StatusCode::Unsupported, error.message};
        case SushiRuntime::ErrorCode::AllocationFailed:
            return Status{StatusCode::OutOfMemory, error.message};
        case SushiRuntime::ErrorCode::CompileFailed:
        case SushiRuntime::ErrorCode::KernelFailed:
        case SushiRuntime::ErrorCode::DeviceLost:
        case SushiRuntime::ErrorCode::ProtocolMismatch:
            return Status{StatusCode::BackendFailed, error.message};
    }

    return Status{StatusCode::BackendFailed, error.message};
}
```

Source: `applications/examples/adapter/status_mapping.hpp`

**An operation.** The work is a lambda handed to `guarded`: it records, runs, tests the result
of the run and reads the tensors back into the host's types. Both channels leave it as one
`Status`:

```cpp
const Status status = guarded([&]()
{
    // The graph lives for this call, so a record call that throws leaves nothing behind.
    SushiRuntime::Graph graph = backend.runtime.graph();
    SushiBLAS::LateCountHold late_counts(backend.runtime);
    SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(),
                                     late_counts, engine.mode());
    // ...
    SushiBLAS::BLASOps(recorder).gemm(left, right, product);
    SushiBLAS::ElementwiseOps(recorder).add(product, shift, shifted);
    SushiBLAS::ReductionOps(recorder).sum(shifted, total);

    const SushiRuntime::Result<SushiRuntime::RunReport> ran = graph.run();
    if (!ran)
        return status_of_error(ran.error());

    SushiBLAS::IO io;
    result = Matrix{a.rows, b.columns, io.to_vector<float>(shifted)};
    result_sum = io.to_vector<float>(total)[0];
    return Status{};
});
```

Source: `applications/examples/adapter/sushiblas_adapter.cpp`

The graph and the recorder belong to the call, so a refused request leaves no recorded task
behind and the program goes on with the same backend.

**The program.** `adapter_demo.cpp` includes `host_program.hpp` alone and tests a `Status`:

```cpp
HostProgram::Matrix result;
float sum = 0.0f;
const HostProgram::Status ran = HostProgram::multiply_add(*backend, a, b, offset, result, sum);
if (ran.code != HostProgram::StatusCode::Ok)
{
    std::cerr << "multiply_add: " << HostProgram::to_string(ran.code) << ": " << ran.message
              << "\n";
    return 1;
}
```

Source: `applications/examples/adapter/adapter_demo.cpp`

An adapter for the rest of the surface follows the same rule: it names `Engine`, the views and
`Tensor` inside, and hands out handles and functions of the host.

## 6. Troubleshooting

**A parse error inside `<sycl/sycl.hpp>` or `<SushiBLAS/tensor/tensor.hpp>`.** A compiler without SYCL
support is compiling the translation unit. Every file that includes a SushiBLAS header needs
the SYCL toolchain's compiler; see section 1.

**`'parallel_for<...>' is deprecated` in a unit that calls a dispatch seam.** The unit
instantiates a kernel and compiles without `-fsycl`. Give it the options of
`SushiBLAS::device_code`; see section 1.

**`find_package(SushiBLAS)` succeeds and the configure then fails on SushiRuntime.**
`CMAKE_PREFIX_PATH` must name both install prefixes. `SushiBLASConfig.cmake` calls
`find_dependency(SushiRuntime)`, so a missing SushiRuntime prefix shows up as a SushiRuntime
configure error.

**A build failure that mentions `-Werror`, `-march` or `sushiblas_warnings`.** SushiBLAS's
private development flags leaked into the export set, which is a defect in SushiBLAS.
`tests/package/` asserts against it; please report it.

**Link errors on a BLAS symbol.** Confirm that you link `SushiBLAS::sushiblas`, the namespaced
alias, and not a bare `sushiblas` target.
