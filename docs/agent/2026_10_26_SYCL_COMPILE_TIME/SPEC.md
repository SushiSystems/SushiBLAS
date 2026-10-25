# Device compilation only where a unit instantiates a kernel

**Status:** Open — written 2026-10-26; the owner approved dropping `-fsycl` and the device targets from the compile options `sushiblas` publishes. Builds are forbidden on this machine while a SushiAI build runs, so the owner's build measures the result.

## 1. The problem

`sushiblas_apply_sycl()` in `cmake/SyclTarget.cmake` puts `-fsycl` and
`-fsycl-targets=spir64,nvidia_gpu_sm_86` on `sushiblas` and on every module object library as
`PUBLIC` compile options. Every unit of this tree, and every unit of every consumer that links
`SushiBLAS::sushiblas`, therefore compiles three times: once per device target and once for the
host. SushiAI measured 997 of its 1024 units compiled that way, of which 5 instantiate a kernel.
SushiEngine and SushiAI each work around it by linking `$<LINK_ONLY:SushiBLAS::sushiblas>` and
copying its include directories and definitions.

## 2. What was measured in this tree

`build/compile_commands.json` of the last build (24 to 25 October) lists 657 SushiBLAS
compilations of 606 files, all with `-fsycl`. For each object, `clang-offload-bundler` extracted
the `spir64` slice and the slice was searched for a SYCL kernel name (`_ZTS`):

| | Files | Compilations |
| --- | --- | --- |
| A kernel in the device slice | 406 | 407 |
| No kernel; the slice is the 2.3 to 2.4 KB empty module | 200 | 250 |

No slice fell between: every slice with a kernel name is above 3.5 KB and every slice without one
is below 2.5 KB. The commits after the build touch 13 files and add no kernel launch.

A single-file syntax check of the kernel-free `tests/operations/blas/test_level1.cpp` took 17.4 s
with the device options and 6.9 s without them, with the toolchain's include directory as
`-isystem`.

Of the library's module sources, the operation modules (`blas`, `blend`, `elementwise`, `logic`,
`nonlinear`, `random`, `reductions`, `signal`, `sparse`, `spatial`) carry 141 kernel units; `cluster`,
`engine`, `index`, `io`, `sparse_storage` and `random/set_seed.cpp` carry 13 that have none.

### Where SYCL reaches a unit through the public headers

94 public headers include `<sycl/sycl.hpp>`. Five of them name nothing from SYCL:
`blas/sparse.hpp`, `elementwise/dispatch.hpp`, `io/io.hpp`, `logic/dispatch.hpp` and
`nonlinear/dispatch.hpp`. The rest name `sycl::half`, `sycl::queue`, `sycl::event` or
`sycl::device` in a declaration, or are kernel templates.

`tensor/tensor.hpp` includes SYCL for `Tensor::device()`, which returns `sycl::device`. Dropping
that include changes nothing a consumer parses: `tensor/storage.hpp`, which `tensor.hpp` includes,
includes the SushiRuntime umbrella `<SushiRuntime/sushiruntime.hpp>` and names `sycl::queue`, and
SushiRuntime's headers include `<sycl/sycl.hpp>`. The same holds for `engine/task_recorder.hpp`,
whose task bodies take `sycl::queue&` and return `sycl::event`. A unit that includes a SushiBLAS
header that reaches SushiRuntime parses SYCL whatever SushiBLAS does, so the gain here is the
device pass, not the parse.

## 3. The design

### One policy home

`cmake/SyclTarget.cmake` holds the whole SYCL policy. On the intel-llvm toolchain it:

- computes the device options once, `-fsycl` plus the device targets and `SB_SYCL_DEVICE_FLAGS`,
  from `SB_SYCL_TARGETS` or else from SushiRuntime's `sushiruntime_gpu_offload`, as today;
- defines `sushiblas_sycl_headers` (`SushiBLAS::sycl_headers`): the toolchain's include directory
  as a system include and `SYCL_DISABLE_FSYCL_SYCLHPP_WARNING`, so a unit compiled without
  `-fsycl` finds `<sycl/sycl.hpp>` and the header stays quiet;
- defines `sushiblas_device_code` (`SushiBLAS::device_code`): the device options as interface
  compile options, for a unit that instantiates a SushiBLAS kernel template;
- `sushiblas_apply_sycl(<target> [DEVICE_SOURCES <src>...])`: every unit of `<target>` gets the
  headers, and only the listed units get the device options;
- `sushiblas_publish_sycl(<library>)`: the library publishes the headers target and the device
  options as interface link options, so an executable's link pulls the device images out of the
  archive.

The toolchain's include directory is found by `sushiblas_find_sycl_include_dir()` in
`cmake/SushiBLASSyclHeaders.cmake`, beside the compiler (`<root>/bin/clang++` gives
`<root>/include`). That file is installed with the package, and `SushiBLASConfig.cmake` calls it
with the consumer's compiler, because an absolute path of the build machine cannot go into the
export. It is a lookup with no policy in it; `SyclTarget.cmake` stays the only place that decides
which unit gets which option.

On the other toolchains (`adaptivecpp`, `oneapi`) `add_sycl_to_target()` is the toolchain's own
function and stays the whole policy, as today.

### Naming the device units

A unit that instantiates a kernel is listed under a `DEVICE_` keyword: `DEVICE_SOURCES` in
`sushiblas_add_module()` and `sushiblas_add_application_test_program()`, `DEVICE_TESTS` and
`DEVICE_BENCHMARKS` in `sushiblas_add_test_program()`, and `DEVICE_SOURCES` of
`sushiblas_apply_sycl()` for an example. The keywords match SushiAI's `DEVICE_SOURCES`. A unit
left off a device list that launches a kernel fails when its test runs, with SYCL's
"No kernel named ... was found"; a unit listed that launches none costs a device pass.

### What a consumer links

| Consumer unit | Links |
| --- | --- |
| Calls the library, the umbrella header, `Engine`, the operation views | `SushiBLAS::sushiblas` |
| Instantiates a dispatch seam or a kernel template, as `tests/package/main.cpp` does with `Dispatch::execute_binary` | `SushiBLAS::sushiblas` and, for that unit, the options of `SushiBLAS::device_code` |

A target whose every unit instantiates kernels links `SushiBLAS::device_code`. A target that mixes
puts the options on the device units:

```cmake
set_property(SOURCE kernels.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<TARGET_PROPERTY:SushiBLAS::device_code,INTERFACE_COMPILE_OPTIONS>")
```

Source: not compiled

## 4. Not verified

- That a unit compiled without `-fsycl` links with device objects into one executable under
  `-fsycl` on intel-llvm and runs as today. SushiEngine builds that way; this tree has not.
- The time saved. The syntax check above is one unit; the owner's clean build gives the total.
- AdaptiveCpp and oneAPI, whose paths this work leaves as they were.
