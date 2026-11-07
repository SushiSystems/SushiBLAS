# Tests of sparse

The tests of the `sparse` module, built into `sushiblas_sparse_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_sparse.cpp` | Times spmv, spsymv, spmm, sptrsv, axpyi and doti and prints GB/s per matrix order |
| `test_accumulator_size.cpp` | Tests that accumulator_bytes gives the size of the accumulator of each dtype |
| `test_index_check.cpp` | Tests the checks on an index base, a compressed array and an index array, on arrays built by hand |
| `test_sparse/test_sparse_level1.cpp` | Tests the sparse-dense level 1 routines of SparseOps against values worked out by hand |
| `test_sparse/test_sparse_recording.cpp` | Tests that a recorded sparse product is ordered behind the task that wrote the values of its matrix and behind no other |
| `test_sparse/test_sparse_spmm.cpp` | Tests SparseOps::spmm column by column against host products, on both layouts |
| `test_sparse/test_sparse_spmv.cpp` | Tests SparseOps::spmv against host products, and the transposed SYCL kernel on a matrix whose heaviest row spans several row blocks |
| `test_sparse/test_sparse_spsymv.cpp` | Tests SparseOps::spsymv and sphemv against mirrored host products |
| `test_sparse/test_sparse_sptrsv.cpp` | Tests SparseOps::sptrsv against triangular systems whose solution is known, and on the host that the order of the triplets does not change the bits |
| `test_sparse/test_sparse_validation.cpp` | Tests the record-time operand checks of the sparse operations |
| `test_sparse_matrix_operand.cpp` | Tests how a CSR matrix is described and the shapes of its arrays that are refused, on arrays built by hand |
| `test_sparse_vector_operand.cpp` | Tests how a compressed vector is described and the sizes that are refused, on arrays built by hand |

## Small tests of the operand checks

`test_accumulator_size.cpp`, `test_index_check.cpp`, `test_sparse_matrix_operand.cpp` and
`test_sparse_vector_operand.cpp` are `Small_`: they build the arrays of an operand with
`HandBuiltOperand` (`tests/support/hand_built_operand.hpp`), pair them with the facts of a storage
that does not exist, and call the overloads of `require_compressed_array`, `require_index_array`,
`describe_sparse_matrix` and `describe_sparse_vector` that take those facts
(`Dispatch::PlacedTensor`, `Dispatch::PlacedSparseMatrix`, `Dispatch::PlacedSparseVector`). The
overloads that take a `Tensor` or a `SparseMatrix` read the facts from the storage and call the same
code.

## Sweeps

A test that runs a routine in both forced modes, on both index bases, under every transposition or
on either triangle does it through `in_each_forced_mode` (`tests/support/forced_mode_checks.hpp`)
and `in_each_index_base`, `in_each_mode_and_base`, `in_each_transpose`, `in_plain_and_transposed`
and `in_each_triangle` (`tests/support/sparse_test_sweeps.hpp`), so the body of a test is a straight
line and a failure names the mode, the base and the flag it ran under. A routine defined only for real or only
for complex elements has its own fixture beside the common one; it skips the other types.

## test_sparse/test_sparse_recording.cpp

`AProductIsOrderedBehindTheTaskThatWroteItsValues` asserts on the recorded graph, for a sparse task
whose operand is three tensors and not one, that a product depends on the task that wrote the
matrix's `values` and on no task it reads nothing from. Its two rows are a writer of `A.values` and
a writer of an unrelated tensor.

## test_sparse/test_sparse_sptrsv.cpp

`TheOrderOfTheTripletsDoesNotChangeTheBits` is `Small_`: it assembles the same triangle from the
triplets in two orders with the host assembly of `sparse_assembly` and solves with the reference
kernel on the host, so no queue runs. The `Medium_` tests of the file launch the operation in both
forced modes.
