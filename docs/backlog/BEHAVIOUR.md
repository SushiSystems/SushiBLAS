# Behaviour

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-1 | sushiruntime KI-1 | A print, a save or a reference run can fault on Windows with CUDA | finish the queue before a print, a save or a `ForceReference` run |  |  | 2026-10-19 |
| KI-2 | own | The reference path of the elementwise, nonlinear and logic seams checks launches, not… | test a formula against an independent value | none, the shared functor is what keeps the two paths one text |  | 2026-10-19 |
| KI-3 | toolchain | Host `HALF` truncates below 2^-13 where a device rounds | compare `HALF` below that magnitude to one unit in the last place |  |  | 2026-10-19 |
| KI-4 | own | Some random families differ between the two paths | compare them within the allowances of `test_random_reference.cpp` | none, each path calls its own math library |  | 2026-10-19 |
| KI-5 | toolchain | The OpenCL CPU runtime crashes compiling `TrmvRow<std::complex<float>>` | none known; taking the diagonal out of the row loop did not avoid it |  |  | 2026-10-19 |
| KI-8 | own | `spmv` and "packed" each name two things | qualify the name by its class or module | RW-1 |  | 2026-10-19 |
| KI-9 | own | `blas/sparse.hpp` is installed and used by nothing but the umbrella and one test | use `SparseOps` | RW-2 |  | 2026-10-19 |
| KI-11 | own | A recorded tensor stays allocated until its graph node dies | drop the compiled plan to release its operands | none, a recorded node owns what it reads and writes |  | 2026-10-19 |
| KI-12 | own | Two binaries that link the static library can share a recorder identity | link the shared library | RW-67 |  | 2026-10-19 |
| KI-14 | own | Backward banded and packed solves differ from the dense solve in the last bit | compare them within a tolerance | none, the two solves sum in different orders |  | 2026-10-19 |
| KI-15 | own | `sptrsv` solves on one work-item | run long solves on a device without a display | RW-55 |  | 2026-10-19 |
| KI-16 | own | A SYCL sum that overflows midway gives an infinity | sum in a wider type | none, the device accumulates in the element type |  | 2026-10-19 |
| KI-17 | own | Eight spatial operations accept an output that overlaps an input | pass distinct tensors | RW-73 |  | 2026-10-19 |
| KI-18 | own | A rank-0 tensor prints as `[]` | read the value with `to_vector` | RW-64 |  | 2026-10-19 |
| KI-20 | toolchain | A generic lambda calling a SYCL task body fails to link on Windows | write the dtype arm as a named class, as `execute_random` and `execute_typed_task` do |  |  | 2026-10-19 |
| KI-22 | own | `sb test --package` ends in a traceback when a DLL is locked | close the program holding the DLL | RW-84 |  | 2026-10-19 |
| KI-23 | own | `sb run --sort` keeps a name the sibling command lines renamed | spell `--sort` | RW-80 |  | 2026-10-19 |
