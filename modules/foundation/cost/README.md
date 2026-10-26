# cost

What a recorded task says about its work: floating-point operations, bytes and the dtype class they are judged in. One table holds a row for every operation id the library records, and one function, `cost_of`, applies the rule of a shape family to a row. Its headers are included as `<SushiBLAS/cost/<name>.hpp>`.

It depends on [`core`](../core/README.md) and on SushiRuntime for `OperationID`. It includes no SYCL, no engine and no SushiProfiler, so it builds without device passes.

| File | Declares |
| --- | --- |
| `cost.hpp` | `Cost`, `UncostedReason`, `real_class_of` |
| `cost_shape.hpp` | `CostShape`, a `std::variant` of `GemmShape`, `ConvolutionShape`, `ElementwiseShape`, `ReductionShape`, `WindowShape` and `CopyShape` |
| `cost_table.hpp` | `CostRow`, `cost_rows`, `find_cost_row`, `cost_of` |
| `composite_cost.hpp` | `softmax_flops`, the cost of an operation a caller composes from several tasks |

## How a cost is stated

`TaskRecorder::insert` is the one place that states it. It calls `cost_of(description.operation, description.shape)`, adds the bytes the operands declared (`OperandSet::declared_bytes`) and hands the result to the runtime as `NodeOptions::cost`. A call site states only facts it already holds: the extents, in the shape struct of its family. The knowledge of what an operation costs is in `cost_table.cpp`.

A task that states no cost carries no `NodeCost` at all; the runtime then writes no `flops` or `bytes` attribute on its span. A stated zero is a real zero: an `im2col` moves data and does no arithmetic, and its row says so. The reason a task has no cost is one of:

| `UncostedReason` | Meaning |
| --- | --- |
| `NoRule` | The operation id has no row |
| `NoShape` | The row exists and the call site stated no shape |
| `DataDependent` | The work depends on data or on a functor the recorder cannot see: `index.for_each`, and every task that takes a late count |
| `Unaudited` | The row is declared and no rule is written for it yet |

`tools/profiling/check_operation_costs.py` fails when a `"..."_op` literal under `modules/` has no row, or a row has no literal.

## Conventions

A rule counts the operations of the operation's formula, one per arithmetic operation and one per transcendental. The count is not instruction-accurate: `exp` is 1, as `add` is 1. A comparison counts 1 and a select counts 0. A complex element is counted as one element in an elementwise rule; only the product families take the complex factor.

| Family | Rule |
| --- | --- |
| Gemm | `2 * m * n * k * batch` for a real type, `8 * m * n * k * batch` for a complex one, plus `epilogue_flops_per_element * m * n * batch` |
| Convolution | `2 * N * Ho * Wo * Cout * (Cin / groups) * Kh * Kw`, times 4 for a complex type, plus `epilogue_flops_per_output * N * Ho * Wo * Cout`. A depthwise convolution has `groups` equal to both channel counts, so `window_dot` states this family |
| Elementwise | `per_element * count` |
| Reduction | `per_element * inputs + per_combine * (inputs - outputs) + per_output * outputs` |
| Window | `per_combine * outputs * taps + per_output * outputs`; the taps are the nominal window, so a window cut by padding is counted whole |
| Copy | 0 |

A negative extent counts as 0, and a product that does not fit 64 bits saturates.

A GEMM epilogue states its count per element of C as `static constexpr std::uint32_t K_FLOPS_PER_ELEMENT` on the functor; an epilogue that states none counts 0, which is the identity epilogue's cost and a lower bound for any other. A bias counts 1 and an activation counts by the transcendental rule above.

`softmax_flops(rows, cols)` is `rows * (5 * cols - 2)`: the maximum (`cols - 1`), the shift (`cols`), the exponential (`cols`), the sum (`cols - 1`) and the division (`cols`).

### Bytes

The bytes of a task are the lengths of the regions it declared, an updated operand counted twice and a region with no length of its own counted 0. For a dense operand this is a lower bound of the traffic the task causes. For a strided view it is the span of the view, which may be above or below the traffic: no direction is guaranteed. A task that declares a scratch lease counts it as the operand it is.

### Weights of the elementwise and nonlinear rows

| Operation | Per element | Formula counted |
| --- | --- | --- |
| `math.ew.<op>` for every unary and binary function | 1 | the function; `clamp` is 2 (a compare per bound) |
| `math.nonlinear.relu` | 1 | the comparison |
| `math.nonlinear.leaky_relu`, `elu` | 2, 4 | comparison, multiplication; comparison, exponential, subtraction, multiplication |
| `math.nonlinear.sigmoid`, `silu` | 4, 4 | negation, exponential, addition, division |
| `math.nonlinear.softplus`, `tanh` | 6, 1 | `max`, `abs`, negation, exponential, `log1p`, addition; the function |
| `math.nonlinear.gelu` | 9 | the tanh form: the cube, the inner polynomial, the tanh, the outer product |
| the backward of each | 2, 2, 4, 3, 9, 5, 3, 19 | the same counting over `d(op)/dx` times `dy`, in the order relu, leaky_relu, elu, sigmoid, silu, softplus, tanh, gelu |

The values are pinned in `tests/foundation/cost/test_cost_table.cpp`, so a change shows as a diff.

## What each row says about its bodies

The runtime times the event a body returns. A body that submits several kernels therefore hands every event to the runtime, through `Device::SubmittedEvents` and `SushiRuntime::note_dispatch_event`, so the span runs from the first kernel's start to the last kernel's end and its FLOPs are not divided by the last stage alone. The audit of every body that submits more than once:

| Operation or kernel | Kernels submitted | Handed over |
| --- | --- | --- |
| `reduce_indexed` and `reduce`: `math.sum`, `prod`, `max`, `min`, `mean`, `asum`, `nrm2`, `argmax`, `argmin`, `count_nonfinite`, `blas.lvl1.asum`, `nrm2`, `dot`, `dotc`, `dotu`, `iamax`, `sparse.doti`, `dotui`, `dotci` | 1 sequential, 2 staged | yes |
| `std`, `var`, `logsumexp` (centred reductions) | 2 to 4 | yes, every stage |
| `triangular_product`: `trmv`, `tbmv`, `tpmv` | 2 | yes |
| `triangular_solve`: `trsv`, `tbsv`, `tpsv`, and `trsm` | 1 open step plus 1 or 2 per block | yes, every launch |
| `sparse.spmv` transposed, `sparse.spsymv` | 2 | yes |
| `random.orthogonal` | one per projection plus 1 | yes |
| `signal.fft`, `signal.inverse_fft` | the table copies, then 1 to 7 per pass over rows and columns, and more for Bluestein lines | yes, every launch |
| `blas.lvl3.gemm` | 1 per variant: local tiled, depth split, joint matrix, no local, packed | not needed |
| `gemv`, `gbmv`, `symv`, `symmetric_product`, `lu_factor`, `lu_solve`, `spmm`, `logic.all`, `logic.any`, `map_task` | 1 in each branch | not needed |

A `sycl::reduction` inside one `parallel_for` (`logic.all`, `logic.any`) may launch more than one kernel inside SYCL; that is not visible from here.

## Rows

Every operation id the library records has a row; `cost_rows()` lists them sorted by id. Costed rows are `blas.lvl3.gemm`, `blas.lvl1.copy`, `swap`, `scal`, `axpy`, the reductions of the table above, every `math.ew.*` but `ordered_sum`, every `math.nonlinear.*`, the comparisons and logic of `logic.*` except `all` and `any`, and the spatial operations `im2col`, `col2im`, `window_dot`, `window_dot_backward`, `pool_max`, `pool_avg`, `pool_max_backward`, `pool_avg_backward`, `upsample_nearest`, `upsample_nearest_backward`, `window_dot_weights_backward`, `broadcast_spatial`, `channel_affine`, `layout_convert`, `concat_channels` and `slice_channels`. The transforms `signal.fft` and `signal.inverse_fft` are costed from a `TransformShape`: `5 * n * log2(n)` per power-of-two line, `10 * m * log2(m) + 6 * m + 14 * n` per Bluestein line of padded size m, and `5 * n` more to untangle a real row of even length. `index.for_each` is `DataDependent`. Every other row is `Unaudited`; the backlog entry RW-121 names each one.

The operations of `graph_primitives` carry an operation id, the hash of their label, and no row: the module is a foundation module that a consumer builds without the cost table, so its nodes state no cost.
