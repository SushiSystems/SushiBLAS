# graph_primitives

Four numeric primitives that record their work on a SushiRuntime task graph: a fixed-order
reduction, a segmented reduction, an exclusive prefix scan and a stable counting sort. Each is a
free function whose first parameter is a `SushiRuntime::Recorder&`, so it records on anything
that has a `recorder()`: a `Graph`, a `Region` of a `DynamicGraph`. They work on
`SushiRuntime::Buffer<T>`, not on tensors.

The module is header-only. Its headers are included as `<SushiBLAS/graph_primitives/<name>.hpp>`,
and everything is in namespace `SushiBLAS::GraphPrimitives`; what a caller of a primitive does not
name is in `SushiBLAS::GraphPrimitives::Detail`.

```cpp
auto graph = runtime.graph();
const SushiRuntime::Result<void> recorded = SushiBLAS::GraphPrimitives::reduce(
    graph.recorder(), values, total, count, SushiBLAS::GraphPrimitives::Sum<float>{});
if (!recorded)
    return SushiRuntime::make_error(recorded.error());

auto report = graph.run();
if (!report)
    return SushiRuntime::make_error(std::move(report.error()));
```

## What it depends on

| Header | Includes |
| --- | --- |
| `combiners.hpp`, `fixed_order_fold.hpp`, `fixed_order_reduce.hpp`, `fixed_order_scan.hpp`, `radix_plan.hpp` | The standard library and each other |
| `invalid_graph.hpp` | `<SushiRuntime/core/error.hpp>`, `<SushiRuntime/core/result.hpp>` and `<SushiRuntime/api/graph/recorder.hpp>` |
| `element_window.hpp` | `invalid_graph.hpp` and SushiRuntime's `Buffer`, `Reads`, `Writes`, `BufferRegion` and `Result` |
| `reduce.hpp`, `segmented_reduce.hpp`, `prefix_scan.hpp`, `counting_sort.hpp` | The headers above and SushiRuntime's public graph headers: `Recorder`, `Access`, `DeviceSized`, `NodeOptions`, `Reads`, `Writes`, `LateCount`, `Buffer`, `BufferRegion`, `Result` |

No header includes another SushiBLAS module, and none names `SushiRuntime::Detail`.

`combiners.hpp`, `fixed_order_fold.hpp`, `fixed_order_reduce.hpp` and `fixed_order_scan.hpp` compile
with a plain C++17 compiler that has neither SYCL nor SushiRuntime on its include path. A consumer
with a build lane of that kind takes the tile size, the host folds and the combiners from them, so
both of its lanes fold in the one order `fixed_order_fold.hpp` defines. `K_REDUCE_TILE` is the only
place the tile size is written.

## Entry points

| Header | Declares |
| --- | --- |
| `reduce.hpp` | `reduce(recorder, input, output, n, op[, identity][, window])`, with `n` a `std::size_t` or a `LateCount`; `ReduceWindow` |
| `segmented_reduce.hpp` | `segmented_reduce(recorder, values, segment_offsets, output, segment_count, op[, identity][, window])`, with the count a `std::size_t` or a `LateCount`; `SegmentedReduceWindow` |
| `prefix_scan.hpp` | `prefix_scan(recorder, counts, offsets, total, n[, window])`, with `n` a `LateCount`; `PrefixScanWindow` |
| `counting_sort.hpp` | `counting_sort(recorder, keys, indices, offsets, total, n, key_range[, counts][, window])`, with both counts a `LateCount`; `CountingSortWindow` |
| `element_window.hpp` | `Detail::ElementSpan`, `Detail::elements`, `Detail::require_window`, `Detail::access_of`, `Detail::region_of` |
| `combiners.hpp` | `Sum`, `Product`, `Minimum`, `Maximum`: binary functors with a static `identity()` |
| `fixed_order_fold.hpp` | `K_REDUCE_TILE`, `fold_tile`, `fold_indexed_tile`, `fold_fixed_order` (a host fold, level by level) and `fold_fixed_order_streamed` (the same order on the stack, reading each element once) |
| `fixed_order_reduce.hpp` | `Detail::ReduceLevel`, `Detail::plan_reduction`, `Detail::reduce_scratch_count` |
| `fixed_order_scan.hpp` | `Detail::LiveTileLevel`, `Detail::plan_tiled_levels`, `Detail::next_live_tile_count`, `Detail::fill_live_tile_levels`, `Detail::fill_live_level_words`, `Detail::live_tile_levels` |
| `radix_plan.hpp` | `Detail::K_RADIX_*`, `Detail::radix_digit_count`, `Detail::radix_digit`, `Detail::fill_radix_plan` |
| `invalid_graph.hpp` | `Detail::invalid_graph`, `Detail::recording_outcome` |

A `LateCount` carries a capacity, which the primitive plans for when it records, and the source
of the live count, which the runtime resolves once per run: `LateCount::fixed`,
`LateCount::from_host` or `LateCount::from_device`.

## Errors

Every primitive returns `SushiRuntime::Result<void>`, and nothing in this module throws. The
result is `[[nodiscard]]`; a caller with no way to go on writes `reduce(...).value();`, which
stops the process on an error.

| Error | When |
| --- | --- |
| `ErrorCode::InvalidGraph` | A size or device rule is broken, or a window does not fit its buffer; that message names the operand. The call returns before any node or scratch is added, and the message starts with the name of the primitive. |
| `ErrorCode::AllocationFailed` | `Recorder::scratch()` could not allocate. The nodes recorded before it stay on the recorder; `Recorder::clear()` drops them. |
| The recorder's refusal | The recorder holds a refusal. The primitive returns the first recording the recorder refused since its last `clear()`, which may be older than the call: a refusal an earlier call left on the recorder is returned by every primitive recorded after it, until `clear()`. It is the error the next run returns as well. |

`Detail::invalid_graph` builds the first and `Detail::recording_outcome` returns the third or
success. A live count above its capacity fails the run, not the recording: `Graph::run()` returns
that error.

Nothing in this module logs: a primitive records nodes and returns its result.

## Windows

Every primitive takes a last argument, a window struct with one element offset per `Buffer`
operand: `ReduceWindow`, `SegmentedReduceWindow`, `PrefixScanWindow`, `CountingSortWindow`. The
primitive reads and writes its operand from that element on, as it would a buffer that started
there, so a result over a window is bit-identical to the result over a buffer holding only the
window. A default window has every offset at zero and is the call without a window.

The length of each operand's window is what the primitive touches: `n` or its capacity for an
input, one element for `output` and `total`, the segment count (plus one for the offsets) and the
key range (plus one for the offsets). `segment_offsets` index the values window, whose length is
not fixed by the call, so `SegmentedReduceWindow::values_count` states it; its default,
`K_TO_END`, runs the window to the end of `values`. The indices `counting_sort` writes are
positions in the keys window.

Each node declares its access as the byte interval of its windows in the whole buffer, keyed as
`Reads` keys that buffer, so the runtime orders it against a node over an overlapping part of the
same buffer and not against one over a disjoint part. A borrowed buffer over the sub-range would
carry another key, which no node over the parent overlaps. A window that does not fit its buffer
returns `ErrorCode::InvalidGraph` before anything is recorded. Scratch does not depend on the
windows.

## What stays fixed

The results are bit-identical across runs, worker counts, devices and steal patterns. The tests
pin the node count of each call, the rule it forms, the width each node launches at and the
order values are combined in.

Every node one call records sits in one rule, so the nodes share one registration index and one
name. Scratch comes from `Recorder::scratch()`, on the device and with the residency of the input
buffer, and lives as long as the recorder.

## `fixed_order_reduce.hpp`

### `K_REDUCE_TILE`

One work-item folds a contiguous run of at most `K_REDUCE_TILE` values left to right. The level
plan, the live level widths, the scan and the tiles of the counting sort all derive from this one
constant; no other number in the module repeats its value.

### `plan_reduction`

A reduction over `n` values is a tree of levels. Level 0 folds the input into
`ceil(n / K_REDUCE_TILE)` partials, the next level folds those, and the last level produces one
value. Which values meet which depends on `n` and the tile size alone, never on how the work was
scheduled. That is the property a replayable simulation needs from every accumulation that feeds
its state, and a schedule-driven parallel reduce cannot give it.

Each level but the last writes its partials to its own window of one scratch block; the last
writes the output. A reduction of one level takes no scratch.

### `fold_tile`

The fold is seeded with the first element of the tile, so the identity never enters the
arithmetic of a non-empty reduction.

## `fixed_order_scan.hpp`

A late-bound primitive plans its levels for the capacity and runs them at the live count. The
live width of a level follows the same recurrence as the plan, so the fold order for a live count
is the order `plan_reduction` gives that count. Every level past the one where the live chain
reaches one value passes that single value through, which is why a late-bound result is
byte-identical to the fixed-count result for the same count.

`fill_live_level_words` writes those widths as `2 * level_count` words that the level kernels
read on the device: the live `in_count` of level `i` at `i`, its `out_count` at
`level_count + i`.

## `reduce.hpp`

### `reduce(recorder, input, output, n, op, identity)`

Reduces the first `n` elements of `input` into `output[0]`. It records one node per level, each
launched over the `out_count` of its level. `identity` is the result when `n` is zero. In that
case the call still records one node that writes it, so a reader of `output` orders after that
node as it would after any other writer.

`Combine` is a binary functor and must be trivially copyable, because a device kernel captures
it.

### `reduce(recorder, input, output, n, op)`

Takes the identity from `Combine::identity()`. The combiners of `combiners.hpp` have one; a
lambda does not and uses the overload above.

### `reduce(recorder, input, output, LateCount, op, identity)`

Plans the tree for `n.capacity()` and records a resolver node, one node per level and a closing
node. The resolver writes the live widths. Each level launches at the capacity of its level and
reads its live width on the device, so a live count below the capacity runs fewer work-items and
the graph does not recompile when the count changes. A live count of zero folds no level; the
closing node is then the one writer of `identity`.

## `segmented_reduce.hpp`

### `segmented_reduce(recorder, values, segment_offsets, output, segment_count, op, identity)`

`segment_offsets` is a CSR-style boundary array of `segment_count + 1` entries: segment `s`
covers `values[offsets[s], offsets[s + 1])` and `output[s]` receives its fold. This is the
scatter-accumulate shape of per-body contact impulses or per-vertex contributions.

One work-item folds one segment left to right, so there are no atomics and no combining across
work-items. The cost is load balance: a segment of ten thousand values and a segment of one take
one work-item each. The caller owns the layout and determinism is the contract, so the imbalance
stays visible; a split that evened it out would break the order.

| Argument | Rule |
| --- | --- |
| `Combine` | A binary functor; trivially copyable |
| `segment_offsets` | `segment_count + 1` non-decreasing entries, the last no larger than the length of the values window |
| `output` | `segment_count` entries |
| `identity` | The result of an empty segment |

### `segmented_reduce(recorder, values, segment_offsets, output, LateCount, op, identity)`

Records a resolver and one node launched at `segment_count.capacity()`. A live count below the
capacity runs fewer work-items; the fold of each live segment is the one the fixed-count overload
produces, and an output entry at or past the live count keeps its value.

## `prefix_scan.hpp`

`prefix_scan` plans the tree of `reduce` over `n.capacity()` and sweeps it twice.

The up-sweep is the tile fold of `reduce` over every level but the top, and it keeps the tile sums
of each level in scratch.

The down-sweep runs from the top level to level 0. The top level has no level above it, so one
work-item scans its tile, at most `K_REDUCE_TILE` values, from a base of zero. That scan yields
the grand sum, so it always runs: a live count of zero still has to publish `total = 0`. Below
the top, each tile starts from the exclusive prefix the level above wrote for it and walks its
values left to right. Level 0 writes `offsets` directly.

Level 0 stops at `offsets[live n - 1]`. A closing node copies the settled `total` into
`offsets[live n]`, so a CSR consumer reads the sentinel from `offsets` and needs no second
buffer. When the plan has one level, the top pass writes the sentinel itself and no closing node
is recorded. No host step runs between the sweeps.

| Argument | Rule |
| --- | --- |
| `counts` | At least `n.capacity()` elements |
| `offsets` | At least `n.capacity() + 1` elements, on the device of `counts` |
| `total` | At least one element; receives the grand sum |

A plan of one level (a capacity of 1 to `K_REDUCE_TILE`) allocates no sweep scratch. A zero-size
USM allocation returns null and the allocator treats null as a failure, so the allocation is
skipped, as `reduce` skips its own.

## `counting_sort.hpp`

`counting_sort` is a stable least-significant-digit radix argsort of `uint32` keys. It writes the
order to `indices`, the exclusive start of each bucket to `offsets`, the live element count to
`total` and, when `counts` is given, the size of each bucket.

| Argument | Rule |
| --- | --- |
| `keys` | At least `n.capacity()` elements; each live key is below the live key range |
| `indices` | At least `n.capacity()` elements; receives positions in the keys window |
| `offsets` | At least `key_range.capacity() + 1` elements |
| `total` | At least one element |
| `n` | The element capacity and the source of the live element count |
| `key_range` | The bucket capacity, not zero, and the source of the live bucket count |
| `counts` | Null, or at least `key_range.capacity()` elements |

`keys`, `indices`, `offsets` and `counts` sit on one device.

One resolver node resolves both counts, publishes `total` and fills the radix plan. Each digit
then records four nodes:

1. Histogram. One work-item per tile of `K_REDUCE_TILE` elements counts its digits into its own
   row, so no run has to clear the histogram.
2. Column scan. One work-item per bin replaces the counts of its column with their exclusive
   prefix over the tiles, in place, and writes the bin total.
3. Bin base. One work-item turns the bin totals into the start of each bin.
4. Placement. One work-item per tile writes its elements at the bin start plus its column
   prefix. Tiles cover ascending index ranges, so equal digits keep their order.

The offsets node runs one work-item per bucket plus the sentinel, each a lower bound over the
sorted keys. It reads no live bucket count: a search for a key past the live range lands on the
live element count, which is the sentinel. A count is the difference of two settled offsets, so
the counts node reads no live scalar either; past the live range both neighbours are the live
element count and the difference is zero.

With a capacity of zero elements the call records the resolver and one node that zeroes the
offsets and the counts.

## `radix_plan.hpp`

The digit count is fixed by `key_range.capacity()` when the call records, so the nodes of every
digit are in the graph. `fill_radix_plan` decides per run which of them do work. A digit above
the live key range gets a live width of zero for its four nodes and is skipped.

Placement alternates between `indices` and one scratch buffer. The plan gives each digit the flag
that says which way it writes, counted back from the last live digit, so the last digit that
runs always lands in `indices` whatever the live key range is. A sort planned for one digit never
needs the scratch buffer and does not allocate it.
