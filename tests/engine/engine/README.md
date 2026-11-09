# Tests of engine

The tests of the `engine` module, built into `sushiblas_engine_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_dispatch.cpp` | Tests the element domains, the dtype switch and the task parameters of the dispatch seam |
| `test_engine/test_engine_contract.cpp` | Tests that a host engine and an engine on a runtime keep the Engine contract, and what only a host engine refuses |
| `test_engine/test_reduce_plans.cpp` | Tests the plan-taking forms of Reduce::reduce and Reduce::reduce_indexed |
| `test_engine/test_run_failure.cpp` | Tests what a run of a test-owned graph returns and leaves behind when a body throws |
| `test_engine/test_runtime_shapes.cpp` | Tests create_tensor and reshape with dimensions held in a std::vector |
| `test_engine/test_tensor_const_access.cpp` | Tests that Tensor's const and non-const data accessors return matching pointers |
| `test_engine/test_tensor_factory.cpp` | Tests TensorFactory and Engine::create_tensor: the shape and strides each overload gives and the device-index refusal |
| `test_engine/test_tensor_view.cpp` | Tests that a tensor view's offset and strides lead to the elements it occupies, on a host engine |
| `test_engine/test_unsupported_execution.cpp` | Tests which record-time refusals throw UnsupportedExecution and which do not |
| `test_execution_mode.cpp` | Tests how the engine's execution mode selects and refuses kernels |
| `test_late_count_hold.cpp` | Tests that a late count's token holds every storage and dies at release |
| `test_kernel_pair_dispatcher.cpp` | Tests that the dispatcher picks the reference body under ForceReference only |
| `test_operand_check.cpp` | Tests the operand checks, the scalar binding and the Tensor constructor and transpose guards |
| `test_operand_set.cpp` | Tests the regions an OperandSet declares, the storages it keeps alive and its merge |
| `test_operation_options.cpp` | Tests that OperationOptions names a node and states no determinism class unless told to, and the record-time check every family shares |
| `test_profile_hooks.cpp` | Tests that the profiling hooks run their body once, under their lock, and let an exception out |
| `test_rng_stream.cpp` | Tests RNGStream's reseed, offset reservation, thread safety and seed determinism |
| `test_scratch.cpp` | Tests the lease a ScratchBuffer holds: its bytes, its region, its copies and its return |
| `test_host_scratch_source.cpp` | Tests what HostScratchSource counts and the memory it serves |
| `test_engine_log_category.cpp` | Checks at compile time that the log category names its module and the SushiBLAS project; no test runner lists it |
| `test_graph_task_sink.cpp` | Tests that GraphTaskSink adds one node to its recorder for each task |
| `test_host_task_sink.cpp` | Tests the parts of HostTaskSink that the TaskSink contract leaves out |
| `test_task_cost.cpp` | Tests the access and the cost a recorded node states for an operation, a shape and its operands |
| `test_task_sink.cpp` | Tests that the host and graph adapters of TaskSink keep the TaskSink contract |
| `test_tensor_profile.cpp` | Tests that profile_of() gives a host tensor the host Profile and refuses a tensor with no storage |
| `test_mode_admission.cpp` | Tests that both ModeAdmission adapters keep its contract |
| `test_host_mode_admission.cpp` | Tests that the host adapter admits no mode that needs a SYCL queue |
| `test_runtime_mode_admission.cpp` | Tests that the runtime adapter admits the modes that run the SYCL kernel |
| `compile_pair_run.cpp` | Checks at compile time that a TaskSink body takes a PairRun; no test runner lists it |
| `test_late_count_sink.cpp` | Tests the LateCountSink contract on the graph adapter: one node per late record, and a refused record names the task |
| `test_runtime_scratch_source.cpp` | Tests what RuntimeScratchSource serves beyond the BlockSource contract |
| `test_scratch_block_source.cpp` | Tests that the host and runtime adapters of ScratchBlockSource keep the BlockSource contract, and what the host adapter counts |
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

`Small_TensorViewSlice` has one row per axis of a row-major matrix, a column-major matrix and a
rank-3 tensor, all on a host engine. Each row asserts the parent's strides against the literal
values its layout gives ({4, 1}, {1, 4}, {20, 5, 1}) before it checks the view, so the oracle
is not the code under test. dim 0 always worked; dim 1 is the case that reported contiguous
strides for non-contiguous memory. In column-major the strides run the other way and a dim-0
slice is the non-contiguous one. The rank-3 rows add an axis that is neither the outermost nor
the innermost, which a two-dimensional test cannot distinguish.

`ReshapeRefusesAnInnerAxisSlice`: a slice keeps its parent's strides, so it reports
non-contiguous whenever narrowing an inner axis leaves gaps. `reshape`, which requires
contiguity, refuses it and does not produce a second wrong view. That an outer-axis slice stays
contiguous is held by `tests/data/tensor/test_tensor/test_tensor.cpp`.

`Small_TensorViewIdentity`: a whole-axis slice changes nothing, so it must leave a tensor
that compares equal to its parent in offset, strides and shape, one row per axis.

`Small_TensorViewRegion`: `region()` is what SushiRuntime's dependency tracker orders on, so it
has to cover every byte the view can reach, not merely as many bytes as the view has elements.
For the inner-axis slice those are different numbers (8 elements over a span of 14, a region of
56 bytes from byte 4), and the smaller one would leave the tail of the view unguarded. The
contiguous row is the control: `element_span` and `num_elements` agree there, which is why the
distinction went unnoticed for so long.

`TransposeSpansItsWholeMatrix`: a transpose swaps strides without moving data, so it spans its
whole matrix, and it always did, which makes it a useful control on `element_span`.

The stride-explicit constructor tests: the stride-explicit constructor is the seam `slice` now
goes through, so its guards are the last thing between a mis-described view and a kernel
reading past the end of an allocation. A coherent set of strides is kept, a negative stride is
refused, and so is a span past the capacity: 4 rows at a stride of 64 reach element 195 of a
16-element allocation. The element count alone (16) would have fitted, which is why the
capacity check has to be asked in terms of the span.
