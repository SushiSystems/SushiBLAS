# Integrating SushiBLAS

How to consume SushiBLAS from another project, and the one fact that decides
what you do and don't need to set up.

---

## 1. The fact that shapes everything

**SushiBLAS's kernels are compiled into the library. They are not header
templates instantiated in your translation units.**

This is the opposite of SushiRuntime's integration story, and it matters:
when you write

```cpp
engine.blas().gemm(A, B, C);
```

that call resolves to a function already compiled into `libsushiblas.a` —
your compiler does not see the GEMM kernel's source, does not instantiate a
device lambda, and produces no device code of its own for it. `Level3::gemm`
is an ordinary (if asynchronous) library call.

What you *do* still need a SYCL-capable compiler for: **the public headers
include `<sycl/sycl.hpp>` directly** and use `sycl::event`/`sycl::device` in
their own signatures (`Tensor::get_device()`, every `BLASOps`/`ElementwiseOps`
method returns `sycl::event`). A translation unit that includes
`<SushiBLAS/SushiBLAS.h>` needs a compiler that can parse those types, even
though it never submits a kernel of its own. This is why linking
`SushiBLAS::sushiblas` still carries `-fsycl` as a `PUBLIC` compile/link
option (`cmake/SyclTarget.cmake`'s `sushiblas_apply_sycl()`) — not because
your code will be compiled *as* a kernel, but because the headers require a
SYCL-aware front end to compile *at all*.

The practical consequence: **you do not need the two-lane build split
SushiRuntime's own [INTEGRATION.md](https://github.com/SushiSystems/SushiRuntime/blob/main/docs/INTEGRATION.md)
describes.** There is no "SYCL lane vs. ordinary lane" decision to make for
SushiBLAS itself — every translation unit that touches `SushiBLAS::sushiblas`
needs the SYCL compiler regardless of whether it writes a kernel, because the
headers alone require it.

---

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

`tests/package/` in this repository is a complete, working example of
exactly this — a two-file consumer with no access to SushiBLAS's source
tree. It also runs the flag-hygiene assertion of [§3](#3-what-you-inherit-and-what-you-must-supply).

### From a sibling checkout

`add_subdirectory(path/to/sushiblas)` also works and gives you the same
`SushiBLAS::sushiblas` target — the [D3 layered resolution](ARCHITECTURE.md#8-the-sushiruntime-dependency-d3)
that SushiBLAS itself uses for its own SushiRuntime dependency applies
identically one level up.

### Version compatibility

The package is written with `SameMinorVersion` compatibility while the
project is pre-1.0: `find_package(SushiBLAS 0.1 REQUIRED)` accepts 0.1.x and
rejects 0.2.0. At the v1.0.0 freeze this becomes `SameMajorVersion`.

---

## 3. What you inherit, and what you must supply

Linking `SushiBLAS::sushiblas` gives you:

| Inherited | Why |
|---|---|
| Include directories | Obviously. |
| The library itself | Obviously. |
| `-fsycl` (on the intel-llvm toolchain) | The public headers need a SYCL-aware compiler to parse `sycl::event`/`sycl::device`, even if you write no kernel of your own — see §1. |
| SushiRuntime's floating-point contract, transitively | SushiBLAS links `SushiRuntime::SushiRuntime` `PUBLIC`, so its `sushiruntime_deterministic_fp` interface (`-ffp-contract=off -fno-fast-math` / `/fp:precise`) reaches you too. |

You do **not** inherit SushiBLAS's own warnings, `-Werror`, sanitizer
selection, or `-march` — those are this project's development choices, and a
dependency silently imposing them on you is a build failure whose cause is
invisible from your own source tree. `tests/package/CMakeLists.txt` asserts
this from the consumer side (checking `INTERFACE_COMPILE_OPTIONS` for
`-Werror`/`-Wall`/`-march=`/`-fsanitize` and `INTERFACE_LINK_LIBRARIES` for
the in-tree policy target names), which is the only side it is observable
from.

You must supply:

- **The same SYCL compiler** SushiBLAS itself was built with — mixing
  toolchains between the library and a consumer is not a supported
  configuration.
- **C++17 or later.** SushiBLAS is built at C++17, matching SushiRuntime; a
  consumer on C++20 is equally supported and is what `tests/package/` compiles
  as. What is *not* supported is a consumer whose standard disagrees with the
  SushiRuntime it links: `SushiRuntime::span` is `std::span` at C++20 and
  SushiRuntime's own shim below, and that difference is visible in mangled
  names across SushiRuntime's compiled ABI.
- **`CMAKE_PREFIX_PATH` naming both install prefixes** — SushiBLAS's and its
  SushiRuntime dependency's — unless both were installed to the same prefix.
- **SushiRuntime's shared library on the loader path** at run time (see
  SushiRuntime's own README for the redistributable notes), plus whatever
  your SYCL implementation needs.

---

## 4. Checking the binary you actually linked

```cpp
#include <SushiBLAS/version.hpp>

std::printf("Built against SushiBLAS headers %s\n", SUSHIBLAS_VERSION_STRING);
```

Unlike SushiRuntime (a `SHARED` library, where "the headers compiled against"
and "the `.dll` actually loaded" can legitimately disagree), SushiBLAS is
`STATIC`: there is no separately loaded artifact whose version could drift
from `SUSHIBLAS_VERSION_STRING` at run time, so the compile-time macro is the
whole story — there is no `SushiBLAS::version_matches()` runtime check to
make, because there is nothing for it to check against.

You can also branch at compile time:

```cpp
#if SUSHIBLAS_VERSION >= SUSHIBLAS_VERSION_ENCODE(0, 2, 0)
    // use something introduced in 0.2.0
#endif
```

---

## 5. Troubleshooting

**A parse error inside `<sycl/sycl.hpp>` or `<SushiBLAS/tensor.hpp>`.** Your
translation unit is being compiled by a non-SYCL compiler. Every file that
includes any SushiBLAS header needs the SYCL-aware compiler — see §1.

**`find_package(SushiBLAS)` succeeds but `find_package` for SushiRuntime
fails downstream.** `CMAKE_PREFIX_PATH` must name *both* install prefixes;
`SushiBLASConfig.cmake` calls `find_dependency(SushiRuntime)` internally, so
a missing SushiRuntime prefix surfaces as a SushiRuntime configure error, not
a SushiBLAS one.

**A build failure mentioning `-Werror`, `-march`, or `sushiblas_warnings`.**
That would be a leak of SushiBLAS's private development flags into the
export set — a defect on our side, not yours. `tests/package/` asserts
against it; please report it.

**Link errors on a BLAS symbol.** Confirm you're linking
`SushiBLAS::sushiblas` (the namespaced alias), not a bare `sushiblas` target
name from an `add_subdirectory` that predates SB-6's packaging work.
