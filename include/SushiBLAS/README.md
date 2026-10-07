# SushiBLAS public headers

`include/SushiBLAS` holds every header a consumer of the library includes. `SushiBLAS.h` includes
the core, storage, engine, tensor and version headers in one line.

| Folder | Holds | Reasoning |
| --- | --- | --- |
| `core/` | Element types, type traits, the format stand-in, the logger | [core/README.md](core/README.md) |
| `engine/` | The dispatch seam, the operand checks, the task recorder | [engine/README.md](engine/README.md) |
| `engine/blas/` | BLAS levels 1 to 3 and the GEMM executor | [engine/blas/README.md](engine/blas/README.md) |
| `engine/logic/` | Comparison and logical operations | [engine/logic/README.md](engine/logic/README.md) |
| `engine/math/` | Elementwise, non-linear, random and reduction operations | [engine/math/README.md](engine/math/README.md) |
| `engine/spatial/` | Window span builders of the spatial operations | [engine/spatial/README.md](engine/spatial/README.md) |
| `experimental/` | LAPACK and signal declarations that have no implementation | [experimental/README.md](experimental/README.md) |
| `kernels/` | The GEMM kernels and epilogues a public seam instantiates | [kernels/README.md](kernels/README.md) |
| `support/` | The accumulator trait, the device policy, the launch and reduction shapes, the window geometry | [support/README.md](support/README.md) |

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `engine.hpp`

### `Engine::execute`

At `if (compiled.roots.empty()) return sycl::event();`: several io.cpp paths call execute()
defensively (flush any pending device write before a host-side read) even when nothing was queued:
e.g. a tensor filled directly through its host-visible pointer, no add_task involved.
IExecutionEngine::execute() throws on a graph with no root nodes, so treat empty as an
already-complete no-op instead.

### `RngStream::reseed`

Rewinds the counter offset to zero as well as setting the seed. Every RNG kernel mixes `(seed,
offset)` into Philox, so the seed alone does not name a stream position: setting the seed while
leaving the offset where the previous draws left it meant two identically-seeded generators on one
engine produced different data, and re-seeding could not reproduce anything. Reproducibility is the
only thing a seed is for, so the reset is part of what seeding means here rather than a separate
call a caller has to remember.

No caller in this repository, and none in the SushiAI tree that consumes it, relies on the offset
surviving a re-seed: SushiAI's dataset shuffling explicitly constructs a fresh Engine to work around
the old behaviour. Nothing is therefore exposed to opt out of the reset; if a use for one appears,
it belongs as a distinct named method (`advance_rng_stream()`), never as a side effect of setting a
seed.

### `RngStream::reserve`

Every distribution draw takes a fresh position so two tasks never reuse a Philox (key, counter)
pair, even under concurrent submission. The sequence restarts at zero on `reseed`.

### `Engine::create_tensor`

Each form takes its dimensions either as a braced list or as a `std::vector<int64_t>`. The braced
list forwards to the vector form, so both check the rank, negative dimensions and byte overflow
before they allocate. `Tensor::reshape` has the same two forms.

## `io.hpp`

### `IO::read`, `IO::write`, `IO::to_vector`, `IO::from_vector`

These four calls move a tensor's elements between host memory and its storage. Each one first runs
the recorded graph with `Engine::execute()`, then copies through the context's queue, so a
`Residency::Device` tensor works the same way as a shared one. The tensor must have storage and be
contiguous, and the host byte count must equal `num_elements * bytes_per_element(dtype)`. The typed
helpers also check that `T` maps to the tensor's dtype through `Detail::HostDataType`. Every
violation throws `std::invalid_argument`.

### Loads run the recorded graph first

`load`, `load_bin` and `load_npy` call `Engine::execute()` before they write, so a task recorded
earlier that writes the same tensor finishes before the file's data arrives and cannot overwrite it.

### `IO::load_npy`

The reader accepts format versions 1.0, 2.0 and 3.0. Version 1.0 stores the header length in two
little-endian bytes; 2.0 and 3.0 store it in four. It rejects a header longer than 1 MiB. It parses
the whole header dictionary and requires the `descr`, `fortran_order` and `shape` keys, no others.
`descr` must equal the descriptor `save_npy` writes for the tensor's dtype, and `shape` must equal
the tensor's shape exactly. `fortran_order` must be `True` for a COLUMN_MAJOR tensor of rank 2 and
`False` for a ROW_MAJOR tensor of rank 2 or more. At rank 0 and 1 either value is accepted, because both orders give the
same memory. A COLUMN_MAJOR tensor of rank 3 or more is rejected, as it is in `save_npy`.

## `storage.hpp`

### `Storage::get_context`, `get_residency`, `get_device_index`

The context, residency and device index are fixed at construction and readable only through these
accessors. `size_bytes` and `requested_bytes` remain public fields.

### `Storage::allocation_base`

Returns: The pointer the runtime context handed back, or nullptr before construction completed.

Warning: This is the *allocation* base, never a tensor's data. Handing this pointer to a kernel
silently ignores `Tensor::storage_offset`, so a tensor that is a view at a non-zero offset reads or
writes the start of the allocation instead of its own elements, which is exactly the defect that
made this member private. Use `Tensor::data()` / `Tensor::data_as<T>()` to address a tensor's
elements, and `Tensor::region()` to describe them to the dependency tracker. The only legitimate
uses of the base are identity ("which allocation is this?") and deallocation.

## `tensor.hpp`

### `Tensor::storage_offset`

Elements, never bytes. `data()` scales it by `bytes_per_element(dtype)`, so a view is by
construction at a whole number of elements from a 64-byte-aligned allocation base and therefore
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
consumed (`A.strides[rank - 3]` is handed to the GEMM/SYRK/TRSM drivers as the distance between
consecutive batch slices) and with every batched BLAS API, which takes a `stride_a` alongside a
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
slice does not occupy, and `slice()` used to go through it, which is the defect the explicit-stride
constructor exists to make unspellable.

Prefer the shape-derived constructor for anything being allocated; this one is for describing memory
that already has a layout.

- `strides_in`: The stride of each dimension, in elements, in the same fixed-width form
  `layout_strides` returns and `strides` holds: entries past `dims.size()` are unused, exactly as
  they are in `shape`. A fixed-width array rather than a span so that a stride count and a dimension
  count cannot disagree; every entry within the rank must be non-negative, since a negative one
  would make `element_span`, and therefore `region`, describe the wrong interval.
- `offset`: The starting position in the storage, **in elements** (see `storage_offset`).
- `l`: The memory layout this view is *interpreted* under. It no longer decides the strides, but it
  still tells a BLAS driver how to read the view.

Throws `std::runtime_error` If the rank is too high, a stride is negative, or the described elements
reach past the end of `s`.

The bounds check in the explicit-stride constructor uses `element_span()`, not `num_elements`: a
strided view spreads its elements over a wider interval than it has elements, and it is the wider
interval that has to fit inside the allocation.

### `Tensor::data`

This calculates the address using the storage pointer and the offset.

**This, not `storage->allocation_base()`, is the pointer an operation hands to a kernel.** The two
differ by `storage_offset` elements, which is zero for a freshly allocated tensor and non-zero for
every view: a slice, a reshape, or a tensor a memory planner placed inside a shared arena.

### `Tensor::data_or_null`

The operation drivers treat a tensor with no storage as "nothing to address" rather than as an
error, and every one of them used to spell that as `t.storage ? t.storage->data_ptr : nullptr`: the
exact expression that dropped `storage_offset`. This is that idiom, offset-correct, in one place.

### `Tensor::element_span`

Not `num_elements`. For a contiguous tensor the two agree, which is why the distinction went
unnoticed, but a strided view (a slice along an inner axis, or a transpose of one) holds
`num_elements` elements scattered across a *wider* interval, and it is the wider interval that says
which bytes the view can reach. `1 + sum((shape[i] - 1) * strides[i])` is the distance from the
first element to the last, which bounds that interval exactly.

Used by the capacity check and by `region`, both of which are questions about reachable memory
rather than about element count.

### `Tensor::region`

Keyed on the allocation base (the tracker's identity for "same buffer") and narrowed to the view's
own byte interval, so two disjoint views of one allocation, what a memory planner produces when it
packs an arena, are correctly seen as *not* conflicting, while two overlapping views still are.
Passing the bare base pointer instead, as every driver used to, made every tensor in an arena alias
every other one.

The length is `element_span` rather than `num_elements`, so the bound stays true for a strided view
as well as a contiguous one. Stopping at the element count would under-declare what the view
reaches, and the tracker would then emit no edge against whatever sits in the gaps between its
elements.

### `Tensor::is_contiguous`

"Continuous" means "laid out exactly as this tensor's layout would lay it out", so the answer is a
comparison against `layout_strides` rather than a second, independently written walk over the
dimensions: the two used to be written separately, which is how the rank-3 column-major batch stride
could be changed in one and not the other.

Axes of extent 1 are exempt: their stride is unobservable, so a tensor is not declared
non-contiguous over a stride nothing can ever use.

### `Tensor::slice`

A view: it moves the offset and shortens one axis, and **keeps the parent's strides**, because
narrowing an axis does not move any of the elements that remain.

Keeping the parent's strides is the whole content of `slice`, and it used to be wrong. The slice was
built through the shape-derived constructor, which recomputes contiguous strides from the new shape,
so for `dim == 0` the answer happened to come out right (shortening the outermost axis of a
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
