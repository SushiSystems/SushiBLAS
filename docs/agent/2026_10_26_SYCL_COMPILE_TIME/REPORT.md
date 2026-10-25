# Device compilation only where a unit instantiates a kernel: report

**Status:** Open — written 2026-10-26 by the SushiAI orchestrator from the worker's reply; S1 to S3 are committed, S4 waits on the owner's build.

## Dispatches

dispatch: SushiBLAS SYCL compile plan and S1 to S3 — opus medium, 160 tool calls, passed, rung 1

## 1. What was measured

The spir64 slice of every object of the build of 2026-10-24 and 25, read with
`clang-offload-bundler`: 406 of 606 files carry a kernel and 200 do not, 250 of the 657
compilations. `test_level1.cpp`, a unit with no kernel, compiles in 6.9 s for the host alone and
in 17.4 s with the device options.

## 2. What was done

| Commit | Change |
| --- | --- |
| `86eef1c` | The plan and `SPEC.md` |
| `a4a3abd` | The units that instantiate a kernel are listed under `DEVICE_` keywords |
| `bb51e60` | Only those units compile for the device. `SushiBLAS::sushiblas` publishes `SushiBLAS::sycl_headers` and the device link options and no device compile option; a consumer's kernel unit uses `SushiBLAS::device_code`. The export, the Config file and the package test have the same shape |
| `f47ae6a` | Five operation headers no longer include SYCL they do not use |
| `6750fd3` | The plan records S1 to S3 |

Each changed unit passed `clang++ -fsyntax-only` with the flags of `build/compile_commands.json`,
host units without `-fsycl` and device units with it. `sycl/level3/test_gemm.cpp` and
`test_blas.cpp`, compiled for the host on purpose, are refused: `parallel_for` is deprecated
there, and `-Werror` makes it an error, so a kernel unit left off a list does not compile.

## 3. Found by the first SushiAI build against this tree

`graph_primitives/counting_sort.hpp` defines `counting_sort` as an `inline` function whose body
launches kernels, and `SushiBLAS/SushiBLAS.hpp` includes it, so every unit that includes the
umbrella instantiates those kernels. SushiRuntime's new compile-time guard refused SushiAI's
`training_program.cpp`, which included the umbrella and compiled for the host. SushiAI now
includes the headers it uses; a consumer that includes the umbrella links `SushiBLAS::device_code`.
Moving the primitive's body into a source of this library would remove the trap; it is not done.

## 4. Proposals for the owner

Each changes a public API.

1. `Tensor::device()` returns a `DeviceIndex`. Alone it gains nothing, because `storage.hpp`
   still brings in SYCL.
2. `storage.hpp`, `tensor.hpp` and `task_recorder.hpp` include the SushiRuntime headers they use
   instead of `SushiRuntime/sushiruntime.hpp`, and `Storage::queue()` leaves the public header.
3. SushiRuntime keeps SYCL out of `Runtime`, `Buffer` and `Result`; only this takes SYCL parsing
   out of host units.
4. Task bodies of the task recorder stop taking `sycl::queue&` and returning `sycl::event`.

## 5. What the owner's build must confirm

1. The configure succeeds and `compile_commands.json` shows `-fsycl` on the 406 listed files.
2. Host and device objects in one archive link into executables.
3. `sb test` passes, so no kernel is missing from an image.
4. `sb test --package` passes.
5. The clean build time before and after.

AdaptiveCpp and oneAPI are untested: they call `add_sycl_to_target` on every target given to
`sushiblas_apply_sycl`.
