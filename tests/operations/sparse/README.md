# Tests of sparse

The tests of the `sparse` module, built into `sushiblas_sparse_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_sparse.cpp` | Times spmv, spsymv, spmm, sptrsv, axpyi and doti and prints GB/s per matrix order |
| `test_accumulator_size.cpp` | Tests IndexBase, the two sparse operands, the kernel index views and accumulator_bytes |
| `test_sparse/test_sparse_level1.cpp` | Tests the sparse-dense level 1 routines of SparseOps against values worked out by hand |
| `test_sparse/test_sparse_recording.cpp` | Tests that a recorded sparse task computes its result, keeps its operands and is ordered behind the task that wrote them |
| `test_sparse/test_sparse_spmm.cpp` | Tests SparseOps::spmm column by column against host products, on both layouts |
| `test_sparse/test_sparse_spmv.cpp` | Tests SparseOps::spmv and the transposed SYCL kernel against host products |
| `test_sparse/test_sparse_spsymv.cpp` | Tests SparseOps::spsymv, sphemv and their SYCL kernel against mirrored host products |
| `test_sparse/test_sparse_sptrsv.cpp` | Tests SparseOps::sptrsv against triangular systems whose solution is known |
| `test_sparse/test_sparse_validation.cpp` | Tests the record-time operand checks of the sparse operations |

## test_sparse/test_sparse_recording.cpp

Three properties of a sparse task, whose operand is three tensors and not one:
`ARecordedTaskComputesItsResult`, `AMatrixDestroyedBeforeExecuteIsKept`, and
`AProductIsOrderedBehindTheTaskThatWroteItsValues`, which asserts on the recorded graph that a
product depends on the task that wrote the matrix's `values` and on no task it reads nothing
from. `TheSameCallsReplayIntoASecondGraph`
records the same calls through a second recorder.
