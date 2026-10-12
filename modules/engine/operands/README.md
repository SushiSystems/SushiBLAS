# operands

The operand checks and operand views that more than one operation family uses: the BLAS argument
checks, the vector and matrix operand views, the scalar narrowing, the task metadata, the scalar result
check and the scratch leases. `blas` and `sparse` both build on them.

It depends on `core`, `support`, `tensor` and `engine`. Its headers are
included as `<SushiBLAS/operands/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `argument_check.hpp` | The record-time checks on the flags, scalars and routine domain of a BLAS call |
| `matrix_operand.hpp` | MatrixOperand, the batch and matrix geometry of one dense matrix operand of a BLAS call |
| `scalar.hpp` | BLASScalar, the real or complex scalar argument of a BLAS call |
| `blas_metadata.hpp` | BLASMetadata, which fills a BLAS task's metadata slots in one fixed order |
| `scalar_result.hpp` | The dtype and the shape check of a scalar result |
| `scratch_lease.hpp` | The helpers that size and lease the device scratch of a task |
| `vector_operand.hpp` | VectorOperand, the length and increment of one vector operand of a BLAS call |
