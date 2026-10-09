# Tests of tensor

The unit tests of the `tensor` module, built into `sushiblas_tensor_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_storage_offset.cpp` | Tests every operation family on views at a non-zero storage offset |
| `test_tensor_const_access.cpp` | Tests that Tensor's const and non-const data accessors return matching pointers |
| `test_tensor_factory.cpp` | Tests TensorFactory and Engine::create_tensor: shapes, layout, device index and size |
| `test_tensor_view.cpp` | Tests that a tensor view's offset and strides lead to the elements it occupies |

## test_storage_offset.cpp

Every operation family runs against tensors that are views at a non-zero `storage_offset`, not
at the base of their allocation. This is the shape a memory planner produces: one arena
allocation with many tensors packed into it, each handed to an operation as a view at an
offset. Most families once captured `tensor.storage->data_ptr` and dropped the offset, so a
view silently aliased the start of its arena, reading the wrong inputs and writing over
whatever else lived there.

Each test asserts two things, and the second carries the weight. That the result is correct
often passes by accident: with a single tensor per allocation, offset 0 and "ignore the offset"
are the same address. That the poison bytes before and after the view survive untouched cannot
pass by accident, because an operation that ignores the offset writes over them.

The tests run in family order: BLAS levels 1 to 3, elementwise, nonlinear, logic, reductions,
random, the two tests about what an offset counts, and last three about alignment and one
about `get_device`.

`OffsetIsCountedInElementsNotBytes`: the unit of `storage_offset` is a decision, not an
accident, and it is the one that keeps every view naturally aligned. An element count scaled by
the element width can never land an fp64 view on a 4-byte boundary. The test pins it so that
"offsets are bytes" cannot be introduced later and pass the family tests, which use one dtype
at a time and would not notice.

`DisjointViewsOfOneAllocationDoNotAlias` is what the memory planner in the layer above needs:
two tensors packed into one arena must not be reported as touching the same memory, or every
operation in the graph serialises behind every other one.

## test_tensor_view.cpp

The file asks whether a view's metadata describes the memory it occupies. The question it asks
of `slice` is not "are the values right": a slice does not move anything, so there is nothing
to get right in that sense. It is "do the strides the view reports lead to its elements".
`slice` once answered no for every `dim > 0`, because it rebuilt the view through the
shape-derived Tensor constructor, which lays contiguous strides out afresh from the new shape.
For `dim == 0` that coincides with the truth; for an inner axis it describes a compacted tensor
that does not exist. Nothing in the tree called it that way, so nothing caught it.

Each test therefore addresses the view's elements through its own reported strides and checks
what it finds against the value the parent put there. That is the only formulation that fails
when the strides are wrong, and it is what a downstream kernel would do.

`SliceAlongEachAxisAddressesItsElementsRowMajor`: dim 0 always worked; dim 1 is the case that
reported contiguous strides for non-contiguous memory. In column-major the strides run the
other way and a dim-0 slice is the non-contiguous one. The rank-3 test adds an axis that is
neither the outermost nor the innermost, which a two-dimensional test cannot distinguish.

`InnerAxisSliceReportsItselfNonContiguous`: a slice keeps its parent's strides, so it reports
non-contiguous whenever narrowing an inner axis leaves gaps. `reshape`, which requires
contiguity, refuses it and does not produce a second wrong view.

`FullWidthSliceIsTheIdentity`: a whole-axis slice changes nothing, so it must leave a tensor
that compares equal to its parent in offset, strides and shape.

`RegionCoversEveryByteAStridedViewReaches`: `region()` is what SushiRuntime's dependency
tracker orders on, so it has to cover every byte the view can reach, not merely as many bytes
as the view has elements. For a strided view those are different numbers, and the smaller one
would leave the tail of the view unguarded.

`ContiguousTensorSpansExactlyItsElements`: the contiguous case must not have moved.
`element_span` and `num_elements` agree there, which is why the distinction went unnoticed for
so long.

`TransposeSpansItsWholeMatrix`: a transpose swaps strides without moving data, so it spans its
whole matrix, and it always did, which makes it a useful control on `element_span`.

`StrideExplicitConstructorRejectsIncoherentMetadata`: the stride-explicit constructor is the
seam `slice` now goes through, so its guards are the last thing between a mis-described view
and a kernel reading past the end of an allocation. In the last case, 4 rows at a stride of 64
reach element 195 of a 16-element allocation. The element count alone (16) would have fitted,
which is why the capacity check has to be asked in terms of the span.
