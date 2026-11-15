# broadcast

`BroadcastOps`, the broadcast seam and its four arithmetic operations: a matrix meets up to four
vectors, each holding one value per row or one value per column, and a functor combines them in one
task.

It depends on `core`, `tensor`, `engine` and `functor_map`. Its headers are included as
`<SushiBLAS/broadcast/<name>.hpp>`. The task body that runs the functor is `broadcast_task.hpp` in
[`functor_map`](../../engine/functor_map/README.md).

| File | Declares |
| --- | --- |
| `broadcast_operand.hpp` | BroadcastAxis, which names the index that selects from a vector, and BroadcastOperand, which holds one vector with its axis |
| `dispatch.hpp` | `Dispatch::execute_broadcast`, the executor that checks the operands, declares them and records the task |
| `broadcast.hpp` | BroadcastOps, whose `add`, `sub`, `mul` and `div` broadcast one vector |

`source/` holds one file per operation: `add.cpp`, `sub.cpp`, `mul.cpp` and `div.cpp`. Their
operation identities are `"broadcast.add"_op` to `"broadcast.div"_op`.

## The call

```cpp
SushiBLAS::BroadcastOps broadcast(recorder);
broadcast.sub(x, mean, centred, SushiBLAS::BroadcastAxis::Leading);
broadcast.mul(centred, gamma, y, SushiBLAS::BroadcastAxis::Trailing);
```

A matrix operand `A` of `outer * inner` elements meets up to four vectors. A leading vector holds
one value per outer index: `v[o]` meets every element of row `o`. A trailing vector holds one value
per inner index: `v[n]` meets element `n` of every row. The functor receives the element of `A` and
the selected value of each vector, in the order the vectors are given:

```cpp
const std::array<BroadcastOperand, 3> vectors{{{&mean, BroadcastAxis::Leading},
                                               {&inv, BroadcastAxis::Leading},
                                               {&gamma, BroadcastAxis::Trailing}}};
Dispatch::execute_broadcast(recorder, x, vectors, y, "layer_norm.apply", "layer_norm.apply"_op,
                            [](auto v, auto m, auto s, auto g) { return (v - m) * s * g; });
```

The functor may be a bare callable, a `bind_scalars` bundle or a `bind_reads` bundle, as for the
other seams. A `bind_writes` or `bind_updates` bundle is refused at compile time.

## What the call checks

Every refusal is a `std::invalid_argument` thrown at record time that names the operation, the
operand and the rule.

- **Geometry.** Every vector of one kind has one element count. When both kinds are present,
  `outer * inner` equals `A.num_elements`. When one kind is absent its count is `A.num_elements`
  divided by the other, and the division must be exact. `outer` must equal a product of leading
  extents of `A`'s shape, the empty product included, so `[B, T, D]` takes a leading count of `B`
  or `B * T` and a trailing count of `D` or `T * D`. The message gives the shape and the count.
- **Rank.** `A` and `C` have any rank and one shape. A vector has any rank: only its element count
  and its flat order matter, so a `[T, D]` table is a trailing vector of `[B, T, D]`.
- **Layout.** `A`, `C` and the vectors are `RowMajor` and dense. A `ColumnMajor` operand is refused
  by name at any rank, because its flat order is not prefix by suffix above rank 1.
- **Element type.** One dtype for every operand, from `Domain::Real` unless the caller widens it.
  The number of vectors is one to four, by `static_assert`.
- **Aliasing.** `C` may be the same view as `A`: each work-item reads its element before it writes
  it. `C` may not partly overlap `A`, and may not overlap a vector. A vector may overlap `A`.
- **Empty.** An `A` with no element records nothing and returns an empty handle, as
  `execute_binary` does.

## Paths

The device path launches `sycl::range<2>{outer, inner}`, the shape `ger` uses, so no work-item
divides. The reference path is two nested host loops over the same functor object, so the paths
agree bit for bit where the functor does. Nothing accumulates, and the task states no determinism
class. There is no `sycl::vec` path.

Per element the task loads one element of `A`, one element per vector and stores one.

## What this module does not do

- It does not broadcast between two arbitrary shapes. A broadcast over a middle axis needs an
  index decomposition per element over the rank.
- It does not route `ElementwiseOps` through the seam: `ElementwiseOps` keeps its same-shape
  contract, its kernels and its vector path. `SpatialOps::channel_affine` keeps its own kernel.
- It has no constructor from an `Engine`. An `Engine` records no graph; a view is built on the
  `TaskRecorder` of a `RecordingGraph`, as every other operation view is.

The reasoning is item B3 of the GRAPH_CAPABILITIES design topic of SushiAI.
