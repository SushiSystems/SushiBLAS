# Tests of engine

The tests of the `engine` module, built into `sushiblas_engine_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_dispatch.cpp` | Tests the element domains, the dtype switch and the task parameters of the dispatch seam |
| `test_engine/test_reduce_plans.cpp` | Tests the plan-taking forms of Reduce::reduce and Reduce::reduce_indexed |
| `test_engine/test_run_failure.cpp` | Tests what Engine::execute returns and leaves behind when a task body throws |
| `test_engine/test_runtime_shapes.cpp` | Tests create_tensor and reshape with dimensions held in a std::vector |
| `test_engine/test_tensor_const_access.cpp` | Tests that Tensor's const and non-const data accessors return matching pointers |
| `test_engine/test_tensor_factory.cpp` | Tests TensorFactory and Engine::create_tensor: shapes, layout, device index and size |
| `test_engine/test_tensor_view.cpp` | Tests that a tensor view's offset and strides lead to the elements it occupies |
| `test_engine/test_unsupported_execution.cpp` | Tests which record-time refusals throw UnsupportedExecution and which do not |
| `test_execution_mode.cpp` | Tests how the engine's execution mode selects and refuses kernels |
| `test_kernel_pair_dispatcher.cpp` | Tests that the dispatcher picks the reference body under ForceReference only |
| `test_operand_check.cpp` | Tests the operand checks, the scalar binding and the Tensor constructor and transpose guards |
| `test_operand_set.cpp` | Tests the regions an OperandSet declares, the storages it keeps alive and its merge |
| `test_recorded_graph.cpp` | Tests that a RecordedGraph counts, runs and discards what its recorder recorded |
| `test_rng_stream.cpp` | Tests RNGStream's reseed, offset reservation, thread safety and seed determinism |
| `test_scratch.cpp` | Tests the lease a ScratchBuffer holds: its bytes, its region, its copies and its return |
| `test_scratch_pool.cpp` | Tests that a ScratchPool keeps returned blocks idle, reuses them and trims them |
| `test_scratch_shelf.cpp` | Tests the shelf of idle device scratch blocks and its size classes |
| `test_task.cpp` | Tests that a typed task records for an admitted dtype, runs its arm and refuses the rest |
| `test_task_description.cpp` | Tests the defaults of a TaskDescription and the round trip of its parameter slots |
| `test_task_handle.cpp` | Tests TaskHandle: validity, equality and the handle of a call that records nothing |
| `test_task_recorder.cpp` | Tests TaskRecorder: the graph and pool it is bound to, and what each mode records |

## test_engine/test_tensor_view.cpp

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
