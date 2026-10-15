# SushiBLAS modules

The library is one tree per module, `modules/<tier>/<module>/`, and a header is included as
`<SushiBLAS/<module>/<name>.hpp>`. A tier depends only on the tiers above it in the table;
`cmake/SushiBLASLayers.cmake` holds the order and the few edges allowed inside a tier.
`SushiBLAS.hpp`, in `library/umbrella`, includes the header of every module that is not internal
in one line.

| Tier | Module | Holds | Reasoning |
| --- | --- | --- | --- |
| foundation | `core` | Element types, type traits, the format stand-in, the logger, the version macros | [README](foundation/core/README.md) |
| foundation | `arithmetic` | The accumulator trait, the complex arithmetic, the real and complex safe math, the host and device scalar functions | [README](foundation/arithmetic/README.md) |
| foundation | `launch` | The device policy and the per-element launch shapes | [README](foundation/launch/README.md) |
| foundation | `reduction` | The fixed-order whole-tensor and segmented reductions and their accumulators | [README](foundation/reduction/README.md) |
| foundation | `reference` | The reference accumulator, the serial host loops and folds behind `ForceReference` | [README](foundation/reference/README.md) |
| foundation | `substitution` | `RunningSolution`, the unrounded solution of a triangular substitution | [README](foundation/substitution/README.md) |
| foundation | `orthonormal` | The vector layout and constants of the orthonormalisation sweep | [README](foundation/orthonormal/README.md) |
| foundation | `sparse_assembly` | The host assembly of CSR arrays from triplets and its input checks | [README](foundation/sparse_assembly/README.md) |
| foundation | `geometry` | `Vector3T`, `QuaternionT`, `Matrix4T`, the blends, the transform builders and the window geometry, usable in a kernel and without SYCL | [README](foundation/geometry/README.md) |
| foundation | `blas_test_oracle` | Test-only: the double-precision BLAS truth, the flag mappings, the band and packed conversions | [README](foundation/blas_test_oracle/README.md) |
| foundation | `serial_reductions` | Test-only: serial host reductions a test takes its expected values from | [README](foundation/serial_reductions/README.md) |
| foundation | `test_runner` | Test-only: the environment that owns the test runtime, and the `main` of every test program | [README](foundation/test_runner/README.md) |
| foundation | `sparse_fixtures` | Test-only: the sample sparse matrices and their host products | [README](foundation/sparse_fixtures/README.md) |
| foundation | `typed_suite_naming` | Test-only: the name generator of a typed suite | [README](foundation/typed_suite_naming/README.md) |
| foundation | `graph_primitives` | The fixed fold order and the four primitives that record on a SushiRuntime `Recorder`: reduce, segmented reduce, prefix scan, counting sort | [README](foundation/graph_primitives/README.md) |
| data | `tensor` | `Storage`, `Tensor`, `TensorFactory` and `HostDataType` | [README](data/tensor/README.md) |
| data | `sparse_storage` | `SparseMatrix`, `SparseVector` and `SparseBuilder` | [README](data/sparse_storage/README.md) |
| data | `tensor_expectations` | Test-only: filling, reading back and comparing a dense tensor | [README](data/tensor_expectations/README.md) |
| kernels | `blas_seam_kernels` | The GEMM kernels and epilogues a public seam instantiates, and the access bricks the BLAS kernels share | [README](kernels/blas_seam_kernels/README.md) |
| kernels | `blas_kernels`, `random_kernels`, `sparse_kernels`, `spatial_kernels` | The internal kernels of each family | [README](kernels/README.md) |
| kernels | `packed_product_harness` | Test-only: the problem, the data and the blockings of the packed GEMM tests | [README](kernels/packed_product_harness/README.md) |
| engine | `engine` | `Engine`, the task recorder, the operand set and checks, the scratch pool, the dispatch seam | [README](engine/engine/README.md) |
| engine | `operands` | The operand checks and views `blas` and `sparse` share | [README](engine/operands/README.md) |
| engine | `functor_map` | The per-element seam: the map task and its bindings | [README](engine/functor_map/README.md) |
| engine | `engine_fixture` | Test-only: `SushiBLASTest`, a fresh `Engine` per test | [README](engine/engine_fixture/README.md) |
| engine | `execution_mode_sweep` | Test-only: the execution modes a routine's tests run it in | [README](engine/execution_mode_sweep/README.md) |
| engine | `graph_probe` | Test-only: `GraphProbe`, a graph a test reads its recorded tasks from | [README](engine/graph_probe/README.md) |
| operations | `blas` | BLAS levels 1 to 3 and the GEMM executor | [README](operations/blas/README.md) |
| operations | `elementwise`, `nonlinear`, `random`, `reductions` | The four math families | [elementwise](operations/elementwise/README.md), [nonlinear](operations/nonlinear/README.md), [random](operations/random/README.md), [reductions](operations/reductions/README.md) |
| operations | `logic` | Comparison and logical operations | [README](operations/logic/README.md) |
| operations | `sparse` | The sparse operations, their record-time checks and operand describers | [README](operations/sparse/README.md) |
| operations | `spatial` | The window, pooling, channel and broadcast operations | [README](operations/spatial/README.md) |
| operations | `io` | `IO` and `IOStatus` | [README](operations/io/README.md) |
| operations | `experimental` | LAPACK and signal declarations that have no implementation | [README](operations/experimental/README.md) |
| operations | `benchmark_harness` | Benchmark-only: the size arguments, environment lines, fill, timing loop and table of the benchmark programs | [README](operations/benchmark_harness/README.md) |
| library | `umbrella` | `SushiBLAS.hpp`, the project umbrella: every public header in one line | [README](library/umbrella/README.md) |
| library | `sushiblas` | The one library a consumer links, composed from every module | [README](library/sushiblas/README.md) |

The sections below are about the headers of `tensor`, `sparse_storage`, `engine` and `io` that
a consumer meets first. Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `engine.hpp`

### `Engine`

`Engine` is a composition root. It is built on a `SushiRuntime::Runtime`, which `runtime()`
returns, and owns five bricks: a `ScratchPool`, a `RecordedGraph`, a `TaskRecorder` bound to
that graph's recorder and to the pool, a `TensorFactory` and an `RNGStream`. It declares no
operation and has no accessor for the graph. A caller that needs a device queue or a wait takes
it from the runtime: `runtime().advanced().queue(SushiRuntime::DeviceIndex{0})`,
`runtime().wait()`. An operation view
(`BLASOps`, `SparseOps`, `SpatialOps`, `ElementwiseOps`, `NonLinearOps`, `ReductionOps`,
`RandomOps`, `LogicOps`) is built from an `Engine&` or from the bricks it needs: `BLASOps
blas(engine)` and `BLASOps blas(engine.recorder())` record on the same graph. Every operation
of a view returns a `TaskHandle`, and the work runs in `Engine::execute()`.

Record calls from several threads on one engine are serialised by the mutex of its
`TaskRecorder`; `execute()` and `set_mode` are not, and neither may be called while a thread
records. See [the engine headers](engine/engine/README.md).

`set_mode` and `mode` forward to the recorder, and `layout` to the factory. `set_mode` is
`noexcept`: `ExecutionMode` has the three enumerators `Auto`, `ForceSYCL` and `ForceReference`,
and none of them is refused.

### `Engine::execute`

Returns a `SushiRuntime::Result<void>` once every recorded task has finished: success, or the
`SushiRuntime::Error` that ended the run. The result is `[[nodiscard]]`; a caller with no way to go
on writes `engine.execute().value();`, which stops the process on an error. A call with nothing
recorded returns success without running: several io.cpp paths call `execute()` before a host-side
read even when no task is pending, and `SushiRuntime::Graph::run()` refuses a graph with no task.
A recording the runtime's recorder refused throws `std::runtime_error` from the record call, and
the refusal stays with the graph: the next `execute()` returns its error unless
`discard_recorded()` was called first.

`execute()` is `RecordedGraph::run()`. The graph is empty when it returns, whether the run
succeeded or failed: the tasks are destroyed with what their bodies captured, so operand storage
and scratch leases are released. One failure keeps the tasks: the runtime answers
`ErrorCode::InvalidState` when another graph's run is in flight on the same runtime, no recorded
task has started, and the caller can call `execute()` again once that run has settled.

### `Engine::discard_recorded`

Drops every task that was recorded through `recorder()` and has not run. A record call that
throws leaves the tasks recorded before it on the graph, and the next `execute()` would run them;
a caller that turns such a throw into a status calls `discard_recorded()` and goes on with the
same engine. The tasks are destroyed with what their bodies captured, so the storage of their
operands is held only by the tensors that still name it, and each scratch lease returns to the
pool. Tensors are not changed: a task that has not run has written nothing. The call is
`RecordedGraph::discard()` on the engine's own graph, the one no accessor reaches; a caller that
owns a `SushiRuntime::Graph` empties it with `SushiRuntime::Recorder::clear()`.

### The scratch pool

`Engine::scratch_pool()` returns the `ScratchPool` operations lease device scratch from; the
recorder hands the same pool out through `TaskRecorder::scratch_pool()`. The engine declares
the pool before the graph, so the graph and the tasks it holds are destroyed before the pool is.
[The engine module](engine/engine/README.md) describes the pool, the shelf and the lease.

### `RNGStream::reseed`

Rewinds the counter offset to zero as well as setting the seed. Every RNG kernel mixes `(seed,
offset)` into Philox, so the seed alone does not name a stream position: setting the seed while
leaving the offset where the previous draws left it meant two identically-seeded generators on one
engine produced different data, and re-seeding could not reproduce anything. Reproducibility is the
only thing a seed is for, so the reset is part of what seeding means here rather than a separate
call a caller has to remember.

No caller in this repository, and none in the SushiAI tree that consumes it, relies on the offset
surviving a re-seed: SushiAI's dataset shuffling explicitly constructs a fresh Engine to work around
the old behaviour. Nothing is therefore exposed to opt out of the reset; if a use for one appears,
it belongs as a distinct named method, never as a side effect of setting a
seed.

### `RNGStream::reserve`

Every distribution draw takes a fresh position so two tasks never reuse a Philox (key, counter)
pair, even under concurrent submission. The sequence restarts at zero on `reseed`.

### `Engine::create_tensor`

Each form takes its dimensions either as a braced list or as a `SushiRuntime::span<const int64_t>`.
Both forward to `TensorFactory::create`, so both check the rank, negative dimensions, byte overflow
and the device index before they allocate. `Tensor::reshape` takes a braced list or a
`std::vector<int64_t>`.

## `tensor_factory.hpp`

`TensorFactory` holds a `SushiRuntime::Runtime` and a default layout, and `create` allocates a dense
tensor. `Engine::create_tensor` forwards to it, and `SparseBuilder` takes a factory, so it builds
sparse operands without an `Engine`. `allocation_bytes` is the size check alone: it returns the
byte count a shape and dtype need, and throws `std::runtime_error` on a rank above
`Core::K_MAX_TENSOR_RANK`, a negative dimension or an overflowing count. A shape with a zero
dimension needs 0 bytes, and `create` then returns a tensor without storage.

## `sparse_matrix.hpp`, `sparse_vector.hpp`

`SparseMatrix` is a CSR matrix: `rows`, `cols`, an `index_base`, and the three rank-1 tensors
`row_offsets` (`rows + 1` entries), `column_indices` and `values`. `SparseVector` is a `size`, an
`index_base` and the two rank-1 tensors `indices` and `values`. The index tensors are INT32 or
INT64. Both are structs of public tensors and check nothing themselves: `SparseOps` checks an
operand when a call is recorded, through the describers under `engine/sparse/`.

## `sparse_builder.hpp`

`SparseBuilder` builds both containers on one device queue from host data. `matrix_from_triplets`
takes COO triplets, adds the values of a repeated (row, column) pair and orders each row by
ascending column. `matrix_from_dense` stores every element of a contiguous rank-2 tensor that is
not equal to zero, NaN included; its precondition is that no recorded task that writes the tensor
is waiting to run. `vector_from_entries` checks its positions with
`Sparse::require_distinct_positions`. The assembly is in `sparse_assembly/sparse_assembly.hpp`; the
builder allocates through the factory and fills the tensors from the assembled arrays. Every
refusal here is `std::runtime_error`.

## `io.hpp`

### `IO::read`, `IO::write`, `IO::to_vector`, `IO::from_vector`

These four calls move a tensor's elements between host memory and its storage. Each one first runs
the recorded graph with `Engine::execute()`, then copies through the storage's queue, so a
`Residency::Device` tensor works the same way as a shared one. The tensor must have storage and be
contiguous, and the host byte count must equal `num_elements * bytes_per_element(dtype)`. The typed
helpers also check that `T` maps to the tensor's dtype through `Detail::HostDataType`. Every
violation throws `std::invalid_argument`.

### File operations return an `IOStatus`

`save`, `load`, `save_bin`, `load_bin`, `save_npy` and `load_npy` return an `IOStatus` and are
`[[nodiscard]]`. A wrong operand still throws `std::invalid_argument`; a file that is missing,
short, malformed or of another shape, dtype or layout is reported through the status, and the
reason is logged once at error level. Each of the six runs the recorded graph first, and a run
that fails returns `RunFailed` before the file is opened. `read`, `write` and `to_string` have no
status to return and throw `std::runtime_error` with the runtime's message for the same failure. `to_string(IOStatus)` gives the enumerator's name.
[The architecture, section 11](../docs/architecture/EXCEPTIONS.md) lists the values.

### Loads run the recorded graph first

`load`, `load_bin` and `load_npy` call `Engine::execute()` before they write, so a task recorded
earlier that writes the same tensor finishes before the file's data arrives and cannot overwrite it.

### `IO::load_npy`

The reader accepts format versions 1.0, 2.0 and 3.0. Version 1.0 stores the header length in two
little-endian bytes; 2.0 and 3.0 store it in four. A header longer than 1 MiB is `MalformedFile`. It parses
the whole header dictionary and requires the `descr`, `fortran_order` and `shape` keys, no others.
`descr` must equal the descriptor `save_npy` writes for the tensor's dtype, and `shape` must equal
the tensor's shape exactly. `fortran_order` must be `True` for a COLUMN_MAJOR tensor of rank 2 and
`False` for a ROW_MAJOR tensor of rank 2 or more. At rank 0 and 1 either value is accepted, because both orders give the
same memory. A COLUMN_MAJOR tensor of rank 3 or more is rejected, as it is in `save_npy`.

## `storage.hpp`

### `Storage::queue`, `residency`, `device_index`

A `Storage` owns the `SushiRuntime::Buffer<std::byte>` the runtime allocated for it, and the
buffer frees the memory when the last reference to the `Storage` goes. The residency and the
device index are fixed at construction and readable only through these accessors, and `queue()`
returns the queue of that device. `size_bytes()` returns the bytes allocated, which is the
requested size rounded up to `K_STORAGE_ALIGNMENT`. The buffer shares ownership of the runtime's
state, so a `Storage` may outlive the `Runtime` object it was allocated from.

### `K_STORAGE_ALIGNMENT`

Every allocation base is a multiple of 128 bytes, and so is every allocation size. `Storage` passes
the constant to `Runtime::buffer` as the alignment it asks for, and the runtime allocates at it;
an `SB_ASSERT` checks the address in a debug build. `storage.hpp` states the constant itself and
takes it from no runtime header. `Tensor::is_aligned()` reports whether a view's first element sits
on that boundary: a view at `storage_offset` 0 does, and a view at another offset is aligned to
its element size only.

### `Storage::allocation_base`

Returns: The base address of the runtime's buffer.

Warning: This is the *allocation* base, never a tensor's data. Handing this pointer to a kernel
silently ignores `Tensor::storage_offset`, so a tensor that is a view at a non-zero offset reads or
writes the start of the allocation instead of its own elements, which is exactly the defect that
made this member private. Use `Tensor::data()` / `Tensor::data_as<T>()` to address a tensor's
elements, and `Tensor::region()` to describe them to the dependency tracker. The only legitimate
uses of the base are identity ("which allocation is this?") and deallocation.

## `tensor.hpp`

### `Tensor::storage_offset`

Elements, never bytes. `data()` scales it by `bytes_per_element(dtype)`, so a view is by
construction at a whole number of elements from a 128-byte-aligned allocation base and therefore
always naturally aligned for its own element type: an fp32 view cannot land on an odd byte. Anything
that computes an offset (`slice()`, a memory planner packing an arena) must count in elements too,
and a planner packing mixed dtypes into one allocation has to round each tensor's start up to a
multiple of its own element width itself; `Tensor` cannot do it for the planner, because by the time
a Tensor exists the offset has already been chosen.

### `Tensor::data`, `data_as`, `data_or_null`

Each accessor has two overloads, picked by the constness of the Tensor. A non-const Tensor returns
`void*` / `T*`; a const Tensor, or a `const Tensor&`, returns `const void*` / `const T*`. Both point
at the same address, `storage_offset` elements past the allocation base. A pointer a task writes
through must therefore come from a non-const Tensor, and a pointer read from a const Tensor goes to
a kernel as an input. `data_or_null()` returns nullptr instead of throwing when there is no storage.

### `Tensor::layout_strides`

The single definition of "contiguous" for the whole library: the constructor lays tensors out with
it and `is_contiguous()` checks against it, so the two cannot drift apart.

**`layout` describes the memory order inside one matrix, not the whole tensor.** For rank <= 2 that
is the entire tensor and this is the familiar rule: row-major strides grow from the last dimension
inward, column-major from the first. For rank >= 3 the trailing two axes are the matrix and take the
layout's order, while every leading (batch) axis is outermost and slowest-varying **whatever the
layout is**, so a rank-3 `{b, m, n}` tensor has batch stride `m * n` in both layouts.

That asymmetry is deliberate. It is the only reading consistent with how the batch stride is
consumed (`Dispatch::batch_stride` hands the GEMM/SYRK/TRSM drivers `rows * cols` to step between
consecutive batch slices) and with every batched BLAS API, which takes a batch stride alongside a
per-slice `lda` rather than folding the batch into the layout. Treating the batch axis as just
another column-major dimension, which is what the old `for (i = 0; i < rank; ++i)` loop did, gave a
rank-3 column-major tensor `strides[0] == 1`, so the driver read slice 0 for every batch element and
a batched column-major GEMM silently computed the same product `b` times.

- `dims`: The extent of each dimension. `SushiRuntime::span`, not `std::span`: the two are the same
  type wherever `std::span` exists (SushiRuntime's span.hpp aliases it at C++20 and supplies its own
  subset below), so this signature is unchanged for a C++20 caller while still compiling on the
  C++17 baseline the rest of the stack is built to. Every span in this header is spelled this way
  for that reason, and none of them crosses the compiled ABI: they are all inline.

Returns: The stride of each dimension, in elements; entries past `dims.size()` are 0.

`const int first_matrix_dim = r - std::min(r, 2);` in `layout_strides`: the matrix inside a batch
element is the trailing two axes, or the whole tensor when there are fewer than two of them.

### `Tensor` with explicit strides

The construction path for a view that is **not contiguous**: a slice along an inner axis keeps its
parent's strides, because that is where its elements actually are. The shape-derived constructor
cannot express that: it lays the dimensions out afresh, which for such a slice describes memory the
slice does not occupy. `slice()` goes through the explicit-stride constructor for that reason.

Prefer the shape-derived constructor for anything being allocated; this one is for describing memory
that already has a layout.

- `strides_in`: The stride of each dimension, in elements, in the same fixed-width form
  `layout_strides` returns and `strides` holds: entries past `dims.size()` are unused, exactly as
  they are in `shape`. A fixed-width array rather than a span so that a stride count and a dimension
  count cannot disagree; every entry within the rank must be non-negative, since a negative one
  would make `element_span`, and therefore `region`, describe the wrong interval.
- `offset`: The starting position in the storage, **in elements** (see `storage_offset`).
- `l`: The memory layout this view is *interpreted* under. It does not decide the strides; it
  tells a BLAS driver how to read the view.

Throws `std::runtime_error` if the rank is too high, a stride is negative, or the described elements
reach past the end of `s`.

The bounds check in the explicit-stride constructor uses the element span, not `num_elements`: a
strided view spreads its elements over a wider interval than it has elements, and it is the wider
interval that has to fit inside the allocation.

### `Tensor::data`

This calculates the address using the storage pointer and the offset.

**This, not `storage->allocation_base()`, is the pointer an operation hands to a kernel.** The two
differ by `storage_offset` elements, which is zero for a freshly allocated tensor and non-zero for
every view: a slice, a reshape, or a tensor a memory planner placed inside a shared arena.

### `Tensor::data_or_null`

The operation drivers treat a tensor with no storage as "nothing to address" rather than as an
error. A driver that tested the storage and took the allocation base itself would drop
`storage_offset`. This accessor is that test, offset-correct, in one place.

### `Tensor::element_span`

Not `num_elements`. For a contiguous tensor the two agree, which is why the distinction went
unnoticed, but a strided view (a slice along an inner axis, or a transpose of one) holds
`num_elements` elements scattered across a *wider* interval, and it is the wider interval that says
which bytes the view can reach. `1 + sum((shape[i] - 1) * strides[i])` is the distance from the
first element to the last, which bounds that interval exactly.

`region` uses it and the capacity check computes the same span; both are questions about reachable
memory rather than about element count.

### `Tensor::region`

Keyed on the allocation base (the tracker's identity for "same buffer") and narrowed to the view's
own byte interval, so two disjoint views of one allocation, what a memory planner produces when it
packs an arena, are correctly seen as *not* conflicting, while two overlapping views still are.
A bare base pointer passed instead makes every tensor in an arena alias every other one.

The length is `element_span` rather than `num_elements`, so the bound stays true for a strided view
as well as a contiguous one. Stopping at the element count would under-declare what the view
reaches, and the tracker would then emit no edge against whatever sits in the gaps between its
elements.

### `Tensor::is_contiguous`

"Continuous" means "laid out exactly as this tensor's layout would lay it out", so the answer is a
comparison against `layout_strides` rather than a second, independently written walk over the
dimensions: with two walks, the rank-3 column-major batch stride can be changed in one and not
in the other.

Axes of extent 1 are exempt: their stride is unobservable, so a tensor is not declared
non-contiguous over a stride nothing can ever use.

### `Tensor::slice`

A view: it moves the offset and shortens one axis, and **keeps the parent's strides**, because
narrowing an axis does not move any of the elements that remain.

Keeping the parent's strides is the whole content of `slice`. A slice built through the
shape-derived constructor, which recomputes contiguous strides from the new shape, is wrong:
for `dim == 0` the answer happens to come out right (shortening the outermost axis of a
contiguous tensor leaves every inner stride unchanged) and for every other `dim` it came out wrong.
A 4x4 row-major tensor sliced to columns [1, 3) reported strides {2, 1}, claiming its four elements
sit at offsets 1, 2, 3, 4 when they really sit at 1, 2, 5, 6. Nothing in this tree called `slice`
with `dim > 0`, which is why it survived; the first caller would have read the wrong memory with no
diagnostic anywhere.

The result of an inner-axis slice is therefore **not contiguous**, and `is_contiguous()` now says
so. This library's operations expect contiguous operands, so such a slice is a description of memory
rather than something to hand straight to `gemm`, which is accurate, and better than a tensor that
lies about where its elements are.

## `version.hpp`

Unlike SushiRuntime (a SHARED library, where "the headers compiled against" and "the .dll actually
loaded" can legitimately disagree), SushiBLAS is STATIC: there is no separate loaded artifact whose
version could drift from these macros at runtime, so a single compile-time version is the whole
story here.

The literals in `version.hpp` are checked against the project's CMake version at configure time (see
the root CMakeLists.txt), so `version.hpp` and the build cannot drift apart.
