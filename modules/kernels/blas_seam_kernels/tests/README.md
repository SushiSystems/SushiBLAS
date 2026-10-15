# Tests of blas_seam_kernels

The unit tests of the `blas_seam_kernels` module, built into `sushiblas_blas_seam_kernels_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_band_access.cpp` | Tests the offsets, the band test and the column ranges of BandAccess |
| `test_givens.cpp` | Tests the Givens rotation and the modified Givens transformation bricks |
| `test_operand_access.cpp` | Tests OperandAccess: the strides of each layout and transpose, and the read it gives |
| `test_packed_access.cpp` | Tests the offsets, the triangle test and the column ranges of PackedAccess |
| `test_packed_micro_kernel.cpp` | Tests the micro-kernel of a packed product on one register block per accumulation type |
| `test_packed_panel.cpp` | Tests pack_panel, which copies a block of op(X) into unit-stride micro-panels |
| `test_packed_product_plan.cpp` | Tests the register block, the default blocking and the plan of a packed product |
| `test_symmetric_access.cpp` | Tests SymmetricAccess reads of both triangles, both layouts and both conjugation policies |
| `test_symmetric_row.cpp` | Tests that SymmetricRow visits a row of a band or packed symmetric matrix in order |
