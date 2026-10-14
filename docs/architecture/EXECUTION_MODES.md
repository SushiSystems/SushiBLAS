# 3. Execution modes

`ExecutionMode` (`engine/execution_mode.hpp`) has three enumerators, and `Engine::set_mode` sets
the mode of every operation recorded from then on.

| Mode | What a task recorded under it runs |
| --- | --- |
| `Auto` | The library's choice, which is the SYCL kernel. The default. |
| `ForceSYCL` | The SYCL kernel. |
| `ForceReference` | The reference path, on the host thread that executes the task. |

`set_mode` is `noexcept` and accepts all three values. No thread may be recording on the engine
while it is called.

**The mode belongs to the task.** `TaskRecorder::record_pair` builds an
`KernelPairDispatcher` from the recorder's mode at the moment of the call and stores it
in the task. Changing the mode afterwards changes later record calls only, so one graph can hold
tasks of different modes:

```cpp
engine->set_mode(ExecutionMode::ForceSYCL);
SushiBLAS::BLASOps(*engine).gemm(A, B, C_actual, transpose_of(transA),
                                 transpose_of(transB), alpha, beta);
engine->execute().value();

engine->set_mode(ExecutionMode::ForceReference);
SushiBLAS::BLASOps(*engine).gemm(A, B, C_expected, transpose_of(transA),
                                 transpose_of(transB), alpha, beta);
engine->execute().value();

SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, tolerance_for(dtype, k), T);
```

Source: `modules/operations/blas/tests/test_gemm.cpp`

`engine` is the fixture's pointer. This is how the tests compare the two paths of an operation: the same call under each mode into
two tensors, then a comparison of the tensors. The test runs the graph between the two calls; the
two tasks could also wait in one graph.

`KernelPairDispatcher::run(q, deps, reference_fn, sycl_fn)` is the single mode switch. Under
`ForceReference` it waits on the dependency events, calls `reference_fn` on the calling thread
and returns an empty event. Otherwise it returns `sycl_fn(q, deps)`.

**Every family has a reference path**, and what that path is differs by family:

| Family | The reference path | Shared with the SYCL path |
| --- | --- | --- |
| Dense BLAS, sparse, spatial | A reference kernel in the `reference/` folder of the family's kernel module, written separately from the SYCL kernel | The record-time checks |
| Elementwise, non-linear, logic | `Serial::map` or `Serial::zip_inplace` (`reference/serial_1d.hpp`): one loop in index order over the operation's functor | The functor, its narrowed scalars, its declared read pointers, the record-time checks |
| Reductions | `Serial::reduce`, `reduce_indexed` or `reduce_axis` (`reference/serial_reduce.hpp`): one fold in index order | The transform and finalize functors, and the accumulator class of every fold that is not a sum |
| Random | The same `draw_block` functor in a loop over the blocks, through `Dispatch::RandomFill` | Philox, the mapping from words to draws, the `{seed, offset}` reservation |

For the last three rows the two paths differ in the loop and, where the functor calls a math
builtin, in the math library. A disagreement between the modes there points at a launch, indexing
or chunking error, a race between work-items, a dropped dependency, or two math libraries rounding
differently. It does not point at the formula: if the functor of `silu_backward` is wrong, both
paths compute the same wrong number and agree. For those families the reference path is a second
execution of one formula, and only the kernel pairs of the first row are two derivations.

A sum is the one fold whose accumulator also differs: the host folds it in
`Kernels::ReferenceSum` (`double`, compensated for FLOAT64), the device in `Reduce::CompensatedSum`
over the element's accumulation type. The two therefore add in different orders and types and are
not expected to agree to the bit.

**`ForceReference` needs host-reachable operands.** A reference path dereferences its operands
on the host, which `Residency::Device` storage does not allow. `OperandSet` notes a device-only
operand as it is declared, and `TaskRecorder::require_host_reachable` throws
`UnsupportedExecution` at record time, before any task or scratch exists. A tensor an epilogue or
a functor declares through `reads` is part of the same operand set and is refused the same way.
