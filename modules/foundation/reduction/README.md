# reduction

The fixed-order reductions on a SYCL device: the whole-tensor fold with its accumulators, and the segmented fold that runs many folds side by side. Its headers are included as `<SushiBLAS/reduction/<name>.hpp>`.

It depends on [`arithmetic`](../arithmetic/README.md), [`launch`](../launch/README.md), SYCL and the standard library.

| File | Declares |
| --- | --- |
| `reduce_1d.hpp` | The deterministic whole-tensor reduction and the accumulators it folds with |
| `reduce_axis.hpp` | The segmented reduction: many independent folds, one result each |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `reduce_1d.hpp`

The deterministic whole-tensor fold and its accumulators. The reasoning behind the header and
behind each of its symbols (`plan`, `reduce`, `reduce_indexed`, `CompensatedSum`,
`ScaledSumOfSquares`, `RunningProduct`, `NaNPropagatingMax`, `NaNPropagatingMin`, `ExactSum`,
`RunningArgMax`, `RunningArgMin`, `AccumulationOf`) is in
[the architecture, section 12](../../../docs/architecture/FIXED_ORDER_FOLD.md), one heading per
symbol.

## `reduce_axis.hpp`

`reduce_1d.hpp` folds a whole tensor to one scalar and `launch_1d.hpp` maps every element to its own
output. This is the third shape, and the one neither of those can express: N folds running side by
side, each over its own slice of the same buffer, producing a vector.

**Why it exists.** A log-sum-exp shift has to be *per row*. Shifting a whole batch by the batch's
global maximum is exact in real arithmetic, log-softmax is invariant to subtracting any constant
from a row, and wrong in binary32, because the invariance is destroyed by the exponential
underflowing: a row sitting 104 below the batch maximum has every `exp(x - global_max)` flush to
zero, its row sum is exactly zero, `log(0)` is `-inf`, and the row comes back `+inf`. That is a
silently wrong answer for gaps around 90-104 and a catastrophically wrong one past it. The only fix
is a maximum computed per row, and a per-row maximum is this shape.

**The layout is a stride pair, not an axis number.** `AxisSpan` describes "where do the segments
start, and how do I walk one" in element counts, which makes the axis, the `Core::Layout` and any
non-unit stride a *caller's* problem, resolved once on the host. Reducing along either axis of
either layout is the same kernel with different numbers, so there is no layout branch in the inner
loop and no second instantiation.

**Two shapes, for the two regimes.** A segmented reduction has two independent sources of
parallelism and which one is available is a property of the input:

1. **Item per segment** (`AxisPlan::cooperative` false). Each work-item
owns one whole segment and folds it serially. The parallelism is the
segment count, so this is what a tall, narrow reduction wants, a
row-wise reduction over a batch, which is the case this file was written
for. Goes through `Device::plan_1d`, so it inherits the CPU-flat /
GPU-grid-stride split `launch_1d.hpp` measured.
2. **Work-group per segment** (`AxisPlan::cooperative` true). Each
work-group owns one segment, grid-strides across it and folds through a
halving tree over local memory. The parallelism is the segment *length*,
which is what recovers the machine when there are fewer segments than it
has execution units: a `[2, 10M]` row-wise sum would otherwise run on
two work-items.

Both are correct at every shape; the choice is only ever about speed. In particular neither requires
the segment length to relate to the work-group width in any way: see `reduce_axis` for what the tree
does with a partial group.

**Determinism.** Same standard, and the same argument, as `reduce_1d.hpp`: `sycl::reduction` and
`sycl::reduce_over_group` are both unusable because the specification fixes no combination order for
them and floating-point addition is not associative. Shape 1 has *no* cross-work-item combination at
all: one item folds one segment in ascending index order. Shape 2 combines through the same
hand-written fixed-offset halving tree stage one of `Reduce::reduce` uses. In both, which elements a
given lane folds and the order it folds them in are pure functions of the launch geometry, and the
geometry is a pure function of the device profile and the span. Two runs of the same binary on the
same device therefore perform the same operations on the same values in the same order. A second
segment axis (`AxisSpan::segments2`) leaves that argument alone: it only changes which address a
fold starts at, so the fold order *within* a segment did not move and no combination across segments
was introduced.

That is an argument from what the code does, not a measurement of a GPU: the same limit
`reduce_1d.hpp` records for itself. See `docs/design/GPU_VALIDATION.md` §5.

**No scratch buffer, and that is structural rather than lucky.** A whole-tensor reduction needs one
because its work-groups all contribute to a single answer and something has to hold the partials
between the two kernels. Here a work-group's tree *is* its segment's answer, so there is nothing to
combine across groups, nothing to stage, and one kernel submission either way.

### `AxisSpan`

Everything the kernel needs to address memory, and nothing about *why*: an axis index, a
`Core::Layout` and a strided view all collapse into these five numbers on the host, so the device
never sees any of them. The element at position `k` of segment `s` is `in[s * segment_stride + k *
element_stride]`, and its result goes to `out[s * out_stride]`.

All strides are element counts and all are non-negative, which `Tensor`'s stride-explicit
constructor already guarantees for any view this can be built from.

### `AxisSpan::segments2`

The second **segment** axis, not a second folded one, and which of the two it is decides which loop
changes. Folding two axes at once needs nothing new: `[N*H*W, C]` is a zero-copy rank-2 view and the
existing five numbers fold axis 0 to `[C]`. What the five numbers cannot express is a segment
identified by a *pair*: `[N,H,W,C] -> [N,C]` has one result per `(n, c)`, whose input offset is `n *
H*W*C + c`, and no single `segment_stride` produces that from a single index.

Segment `(s2, s)` starts at `in[s2 * segment_stride2 + s * segment_stride]` and its result goes to
`out[(s2 * segments + s) * out_stride]`.

### `plan_axis`

The rule is a parallelism argument and nothing else, because both shapes are correct at every size:

- **Enough segments to fill the machine?** Then one item per segment,
which is strictly cheaper: no local memory, no barriers, one
coalesced grid-stride launch, and the fold order is trivially fixed.
"Enough" is one work-item per hardware thread on a CPU, and
`K_WAVES_PER_COMPUTE_UNIT` resident groups' worth on a GPU, matching
what `Device::plan_1d` targets for a flat launch.
- **Otherwise, is a segment long enough to be worth a work-group?** A
group whose segment is shorter than the group is mostly idle lanes
paying log2 of `local` pairs of barriers, so the tree only earns its
keep once every lane has at least one element to fold. One full
group-width pass is that threshold.
- **Otherwise** there is nothing to recover; take the cheap shape.

Warning: The GPU side of this is **derived, not measured**: no GPU run has measured the crossover
(see `docs/design/GPU_VALIDATION.md`). The threshold is a floor below which the cooperative shape is
certainly wrong, not a tuned crossover, and it is the first knob to reach for if a row-wise
reduction underperforms on real hardware. It cannot make an answer wrong: the shape is a pure
function of the device and the span, so it is the same on every run, and both shapes compute the
same thing.

Returns: The geometry; `AxisPlan::cooperative` says which shape.

### `reduce_segments`

The segmented fold with the reads and the store left to the caller. It reduces each of `segments`
index ranges `[0, extent)` to one value: `open(segment)` returns a cursor, the cursor returns
term `k` of that segment in the type `W` the accumulator folds in, and `store(segment, value)` is
called once per segment. `reduce_axis` is this with a cursor that walks an `AxisSpan` and a store
that writes `out[s]`. `Kernels::SYCL::window_dot_weights_backward` calls it directly, because a
tap's terms are products of two tensors and not elements of one.

### `reduce_axis`

Computes `out[s] = finalize(fold(transform(in[s][k])), extent)` for every segment `s`, where `fold`
is `Accum`'s operation applied in an order fixed entirely by the launch geometry. One kernel
submission, whichever shape `plan_axis` chose.

A segment shorter than the work-group is still correct on the cooperative shape: a lane with no
element of its own contributes `Accum<AccumulationOf<T>>{}.value()`, which is the accumulator's
identity by construction, and combining the identity changes nothing. `plan_axis` declines that
shape below one full pass for cost, not correctness.

- `Accum`: The accumulation policy (`CompensatedSum`, `NaNPropagatingMax`, `NaNPropagatingMin`,
  `RunningProduct`), a class template on the accumulation type supplying `identity()`,
  `combine(a, b)`, `accumulate(x)` and `value()`.
- `span`: The segment layout, in element counts.
- `in`: The input array; fold `s` starts at `in + (s / segments) * segment_stride2 + (s % segments)
  * segment_stride`, which is `in + s * segment_stride` whenever there is no second segment axis.
- `out`: Where the per-segment results go.
- `geometry`: Which shape to launch, normally `plan_axis`'s answer. Taken rather than decided here
  so that the planner and the launcher stay separable: a test can then drive both shapes over one
  input and check they agree, which is the only way the cooperative tree gets exercised at a size
  the planner would not send to it.
- `transform`: Callable `(T) -> auto`, applied to each element before it is folded in. Pass the
  identity when there is nothing to do.
- `finalize`: Callable `(AccumulationOf<T> value, std::size_t n) -> auto`, applied once per
  segment, with `n` the segment's own element count, so a per-axis mean is this seam plus `v / n`
  rather than a second kernel.
- `deps`: Events this submission must depend on.

Returns: The event for the submitted kernel.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel: they must be
device-copyable, capture by value, hold no host pointers, allocate nothing and throw nothing.

At `const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), segments);`: one item
per segment, over the same flat/grid-stride launch shape every elementwise op takes, so a CPU covers
the segments in one pass of a wide group and a GPU walks a capped, resident grid. The stride loop's
`s < segments` test is the bounds check.

At `for (std::size_t off = local / 2; off > 0; off >>= 1)`: the same hand-written halving tree as
stage one of Reduce::reduce, and NOT sycl::reduce_over_group, for the reason recorded there: the
spec lets that function combine in any order it likes, which for a non-associative operator means
different bits on different launches of one kernel. `local` is a power of two (see
Reduce::group_width), which is what makes the halving visit every lane exactly once.

The flat branch of `reduce_segments` folds each segment in ascending index order, and no other
work-item touches it. There is nothing in that loop whose order the SYCL specification leaves free.

`const std::size_t s2 = s / segments_;`, in `Detail::AxisOpen`, for both branches: one integer
division per fold, against an inner loop of `extent` elements. It is not worth a specialised
`segments2 == 1` kernel, which would be a second instantiation of every reduction in the library.

The lane loop in the work-group branch of `reduce_segments` walks a stride-`local` slice of one
segment: lane `l` owns positions `l`, `l + local`, `l + 2*local`, and so on. Which positions those
are comes only from the geometry, so it is the same on every run, and consecutive lanes read
consecutive addresses at every step.
