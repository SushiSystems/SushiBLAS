# operands

The operand checks and operand views that more than one operation family uses: the BLAS argument
checks, the vector and matrix operand views, the scalar narrowing, the task metadata and the helpers
of `utils.hpp`. `blas` and `sparse` both build on them.

It depends on `core`, `support`, `tensor`, `blas_seam_kernels` and `engine`. Its headers are
included as `<SushiBLAS/operands/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `argument_check.hpp` | The record-time checks on the flags, scalars and routine domain of a BLAS call |
| `matrix_operand.hpp` | MatrixOperand, the batch and matrix geometry of one dense matrix operand of a BLAS call |
| `scalar.hpp` | BlasScalar, the real or complex scalar argument of a BLAS call |
| `task_metadata.hpp` | BlasMetadata, which fills a BLAS task's metadata slots in one fixed order |
| `utils.hpp` | The BLAS helpers that size a result, allocate scratch and record a typed task |
| `vector_operand.hpp` | VectorOperand, the length and increment of one vector operand of a BLAS call |
