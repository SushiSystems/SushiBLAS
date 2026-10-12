# Sparse headers

`modules/operations/sparse/include/SushiBLAS/sparse` declares the record-time checks and the operand describers that
`SparseOps` in `sparse.hpp` runs on. The containers `SparseMatrix` and `SparseVector` and
the `SparseBuilder` are in [the sparse_storage module](../../data/sparse_storage/README.md); `Core::IndexBase` is in
[the core module](../../foundation/core/README.md); the assembly the builder uses is in
[the support module](../../foundation/support/README.md).

| File | Declares |
| --- | --- |
| `index_check.hpp` | `require_index_base`, `require_compressed_array`, `require_index_array` |
| `sparse_vector_operand.hpp` | `SparseVectorOperand` and `describe_sparse_vector` |
| `sparse_matrix_operand.hpp` | `SparseMatrixOperand` and `describe_sparse_matrix` |
| `accumulator_size.hpp` | `accumulator_bytes`, the size of the type a kernel accumulates a dtype in |

## `sparse/sparse.hpp`

`SparseOps` is a view over a `TaskRecorder`, built from the recorder or from an `Engine&`, as every
other view is. It derives from nothing and no class derives from it. Each routine returns a
`TaskHandle` and records through `Internal::execute_typed_task` under the family name "Sparse
operation", so a sparse task refuses a device-only operand under `ForceReference` as a dense one
does. The three arrays of a matrix and the two of a vector are declared on the task's `OperandSet`
as the tensors they are.

| Routine | Computes |
| --- | --- |
| `axpyi` | `y[p] = alpha * x[p] + y[p]` over the stored positions of a sparse `x` |
| `doti`, `dotui`, `dotci` | The dot product of a sparse `x` and a dense `y`: real, unconjugated, conjugated |
| `gthr`, `gthrz` | A gather from a dense `y` into the values of `x`; `gthrz` also zeroes the gathered elements |
| `sctr` | A scatter of the values of `x` into a dense `y` |
| `roti` | A plane rotation over the stored positions of `x` |
| `spmv` | `y = alpha * op(A) * x + beta * y` for a CSR `A` |
| `spsymv`, `sphemv` | The same product for a symmetric or Hermitian `A` stored as one triangle |
| `spmm` | `C = alpha * op(A) * B + beta * C` for a CSR `A` and dense `B`, `C` |
| `sptrsv` | The solution of `op(A) * x = b` for a triangular CSR `A`, overwriting `b` |

Flags are `Core::Transpose`, `Core::Uplo` and `Core::Diag`, and a scalar that may be complex is a
`BLASScalar`, as in the dense BLAS; see [../blas/README.md](../blas/README.md). `doti` is defined
for real operands and refuses a complex one. `beta` exactly zero skips reading the output of a
product. The output of a product may not overlap an input or an array of `A`.

## `index_check.hpp`

The three checks every sparse describer is built from. Each throws `std::invalid_argument` through
`Detail::throw_operand_error`, naming the operation, the operand and the array.

- `require_index_base` refuses a `Core::IndexBase` whose value is not an enumerator, such as a
  value-initialised one.
- `require_compressed_array` requires one array of an operand to be a rank-1 tensor of the length
  the operand needs, with unit stride when it holds more than one element and with storage when it
  holds any.
- `require_index_array` adds that the array is INT32 or INT64 and that its dtype can hold the
  largest index the operand stores, index base included. An INT32 array is refused for a matrix
  whose entry count or column count does not fit in 31 bits.

## `sparse_vector_operand.hpp`, `sparse_matrix_operand.hpp`

`describe_sparse_vector` and `describe_sparse_matrix` check one operand and return its geometry:
the extents, the entry count and `first_index`, the stored value that names row 0, column 0 and
position 0. A matrix must have `rows + 1` row offsets, one index dtype for both index arrays, and
no entries when it has no rows or no columns. A vector stores at most `size` entries. Both call
`require_device_dtype` on the values, which throws `UnsupportedExecution`.

**What they do not check.** The describers read shapes and dtypes, never the stored indices. That
a column index lies inside the matrix, that the row offsets do not decrease, or that a vector
stores no position twice is not verified when a call is recorded: reading the arrays would need
the graph to have run. `SparseBuilder` builds operands that keep those rules. The reference
kernels test them with `Kernels::Reference::find_defect` inside an `SB_ASSERT`, which aborts in a
build without `NDEBUG` and is empty with it.

## `accumulator_size.hpp`

`Internal::accumulator_bytes` returns `sizeof(Kernels::Accumulator<T>)` for the element type of a
dtype, and 0 for the integer dtypes, which no sparse kernel accumulates. `spmv` and `spsymv` use
it to size the device scratch their SYCL kernels stage partial sums in, at record time, where the
element type is a runtime value.
