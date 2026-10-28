# Tests of blas_seam_kernels

The tests of the `blas_seam_kernels` module, built into `sushiblas_blas_seam_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `level3/test_channel_bias_epilogue.cpp` | Tests that the channel bias epilogue adds the bias its stride pair names |
| `level3/test_gemm_epilogue.cpp` | Tests that the identity epilogue returns the product element unchanged |
| `level3/test_gemm_tile_catalog.cpp` | Tests the sizes and the stable names of the tiled GEMM's catalog shapes |
| `level3/test_gemm_tile_choice.cpp` | Tests the tile chooser of the tiled GEMM on fake device profiles |
| `level3/test_gemm_tile_shape.cpp` | Tests the sizes GemmTileShape derives from its four parameters |
| `level3/test_gemm_tile_table.cpp` | Tests that the measured GEMM tile table holds only entries a chooser can match |
| `level3/test_packed_product_plan.cpp` | Tests the register block, the default blocking and the plan of a packed product |
| `reference/level3/test_gemm.cpp` | Tests the reference gemm against a dense loop, with transposes, batches and an epilogue |
| `sycl/level3/test_gemm.cpp` | Tests the unpacked SYCL gemm against the reference gemm on the host device |
| `sycl/level3/test_gemm_accessed_packed.cpp` | Tests the accessed SYCL gemm, with and without scratch, against the reference gemm |
| `sycl/level3/test_gemm_joint_matrix.cpp` | Tests the joint_matrix HALF gemm bit for bit against the local-tiled kernel, and its fit check |
| `sycl/level3/test_gemm_packed.cpp` | Tests the packed SYCL gemm under three blockings against the reference gemm |
| `sycl/level3/test_packed_micro_kernel.cpp` | Tests the micro-kernel of a packed product on one register block per accumulation type |
| `sycl/level3/test_packed_panel.cpp` | Tests pack_panel, which copies a block of op(X) into unit-stride micro-panels |
| `sycl/level3/test_packed_product.cpp` | Tests that the packed product hands every sum of C, and only those, to its store |
| `test_band_access.cpp` | Tests the offsets, the band test and the column ranges of BandAccess |
| `test_conjugation.cpp` | Tests the two conjugation policies and the types they choose for a kernel |
| `test_givens.cpp` | Tests the Givens rotation brick on real and complex pairs |
| `test_modified_givens.cpp` | Tests the modified Givens transformation brick and the store of its parameters |
| `test_operand_access.cpp` | Tests OperandAccess: the strides of each layout and transpose, and the read it gives |
| `test_packed_access.cpp` | Tests the offsets, the triangle test and the column ranges of PackedAccess |
| `test_row_cursor.cpp` | Tests that a RowCursor steps by a growing stride and retreats along the same path |
| `test_symmetric_access.cpp` | Tests SymmetricAccess reads of both triangles, both layouts and both conjugation policies |
| `test_symmetric_row.cpp` | Tests that SymmetricRow visits a row of a band or packed symmetric matrix in order |
