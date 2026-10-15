# nonlinear

`NonLinearOps`, the activation functions and their backward passes.

It depends on `core`, `support`, `tensor`, `engine` and `functor_map`. Its headers are included as
`<SushiBLAS/nonlinear/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `dispatch.hpp` | The activation executors of the dispatch seam |
| `nonlinear.hpp` | NonLinearOps, the activation functions and their backward passes |

`source/` holds 8 translation units.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to.

## `nonlinear/dispatch.hpp`

### `execute_nonlinear_forward`

The generic seam behind every `NonLinearOps` forward entry point. Like the elementwise executors it
defaults to `Domain::Real`; `sigmoid` and `tanh` opt into `Domain::RealAndComplex`. A consumer
passes `Domain::RealAndComplex` only when its functor is written against the complex-tolerant
`safe_*` helpers.

Warning: This is a low-level generic seam. `op_func` is inlined into a SYCL device kernel: it must
be device-copyable, capture by value only, hold no host pointers, allocate nothing and throw
nothing. A functor instantiated for the complex dtypes is written against
`<SushiBLAS/arithmetic/safe_math.hpp>` rather than the raw `sycl::` math builtins, or it fails to
compile for them.

### `execute_nonlinear_backward`

The generic seam behind every `NonLinearOps` backward entry point. The second operand is whatever
the activation's derivative needs, the saved input for GELU and ELU, the saved output for sigmoid
and tanh, which is why it is named generically here and documented per op at the call site.

Warning: Same device-kernel constraints as `execute_nonlinear_forward`.
