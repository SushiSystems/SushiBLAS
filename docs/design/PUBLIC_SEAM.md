# The public seam

**Status:** Shipped — 2026-10-09

## Problem

Three properties of the public interface depart from the house C++ style: SYCL types in the
interface, exceptions where the style asks for statuses, and kernel headers that are installed.
The estate audit of 2026-10-05 asked for each to be recorded with its reason (findings C15 and
C21 of `docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`).

## Decision

The owner decided each case on 2026-10-09:

- SYCL types stay part of the interface, because the dispatch seams inline a caller's functor
  into a SYCL kernel.
- A failure is an exception, except for a file, which is a returned `IOStatus`.
- A kernel is installed only when a dispatch seam inlines a caller's functor into it.
- `Tensor::storage` stays a public field.

## Design

### SYCL types are part of the interface

| Where | SYCL type |
| --- | --- |
| `Tensor::device()` | `sycl::device` |
| The task bodies a `TaskRecorder` records | `sycl::queue&` and `std::vector<sycl::event>` in, `sycl::event` out |
| The `SushiBLAS::Dispatch` seams | The caller's functor is inlined into a SYCL kernel |

A record call on a view returns a `TaskHandle`, not a `sycl::event`.

The reason is the dispatch seams. A caller that fuses its own arithmetic into a kernel hands the
library a functor, and that functor has to be compiled as device code in the caller's
translation unit. A seam that hid SYCL could not take it.

What it costs a consumer is in [the integration guide](../guides/INTEGRATION.md), section 1:
every translation unit that includes a SushiBLAS header needs the SYCL compiler, and `-fsycl` is
a `PUBLIC` option of `SushiBLAS::sushiblas`.

### A failure is an exception, except for a file

A record call throws before a task reaches the graph. The four types and what throws each are
in [the architecture, section 11](../architecture/EXCEPTIONS.md).

The house style asks for a `[[nodiscard]]` status where an operation fails for a reason outside
the program, and for no exception across a module's public surface:

| Failure | Mechanism | Reason |
| --- | --- | --- |
| A wrong operand, flag or scalar | `std::invalid_argument` | A programmer error the style would assert on. A record call cannot abort the caller's process over an operand it was handed, and the exception names the operation and the operand |
| A file that is missing, damaged or of another shape | `IOStatus`, returned | A failure outside the program, which a caller handles on every call |
| Tensor creation: a rule of the shape, or a failed allocation | `std::runtime_error` | Every record call already throws, so a status here would give one caller two mechanisms, and each of the tree's 1 120 creation calls would unwrap a result to catch the one outside failure, an exhausted device |

### Some kernels are installed and the rest are private

| Folder | Installed | Holds |
| --- | --- | --- |
| `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/` | yes | The GEMM kernels, reference and SYCL, their epilogues, the packed product, and the access bricks of the BLAS operands |
| `modules/kernels/` | no | Every other kernel |

`Dispatch::execute_gemm` takes an epilogue, a functor applied to each element of the product
inside the kernel. The GEMM kernel is therefore a template the consumer's compiler has to see,
and so is every header it includes. A kernel that takes no caller functor is compiled into the
library and stays private.

The rule for a new kernel follows: it goes under `modules/kernels/` unless a dispatch seam inlines a
caller's functor into it.

`tools/layering/check_tier_contents.py` holds the include direction: a kernel includes neither
`Engine`, `engine/` nor the tensor header.

### Members that stay public

`Tensor::storage`, a `SushiRuntime::sushi_ptr<Storage>`, stays a public field: `Tensor` is a
value type that carries plain data, and the kernels and SushiAI read the field directly.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| P1 | Record the three properties and their reasons | built |

## Open questions

None.
