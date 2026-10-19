# Kernel headers

`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels` holds the kernel code a public header instantiates in a consumer's
translation unit: the GEMM kernels and epilogues behind `Dispatch::execute_gemm`, the packed
product they block for CPU caches, and the access and arithmetic bricks the BLAS kernels of both
back ends share. The private kernels are in the `modules/kernels/` tree. The accumulator
trait they share is in [the arithmetic module](../../foundation/arithmetic/README.md).

| File | Declares |
| --- | --- |
| `operand_access.hpp` | `OperandAccess`, how a kernel reads element (row, col) of op(X) |
| `symmetric_access.hpp` | `SymmetricAccess`, which reads a symmetric or Hermitian matrix from one triangle |
| `band_access.hpp` | `BandAccess`, how a kernel reads element (row, col) of a band-stored op(X) |
| `packed_access.hpp` | `PackedAccess`, how a kernel reads element (row, col) of a packed op(X) |
| `row_cursor.hpp` | `RowCursor`, the offset of an element that moves along one row of a band-stored or packed op(X) |
| `symmetric_row.hpp` | `SymmetricRow`, which walks one row of a band-stored or packed symmetric matrix |
| `conjugation.hpp` | `Unconjugated` and `Conjugated`, the policies that turn a symmetric kernel into its Hermitian sibling |
| `givens.hpp` | `GivensRotation` and `givens_rotation`, the arithmetic of xROTG |
| `modified_givens.hpp` | `ModifiedGivens` and `modified_givens`, the arithmetic of xROTMG |
| `level3/gemm_epilogue.hpp` | `GemmIdentityEpilogue`, the default GEMM epilogue |
| `level3/channel_bias_epilogue.hpp` | `ChannelBiasEpilogue`, the per-channel bias add |
| `level3/packed_product_plan.hpp` | `PackedRegisterBlock`, `PackedProductPlan` and the functions that plan a packed product |
| `reference/level3/gemm.hpp` | The reference kernel of GEMM and its form over access bricks |
| `sycl/level3/gemm.hpp` | The SYCL GEMM kernels and the tile geometry they launch with |
| `sycl/level3/gemm_joint_matrix.hpp` | The HALF GEMM kernel that multiplies on the matrix units through `joint_matrix` |
| `sycl/level3/gemm_packed.hpp` | The form of the SYCL GEMM that packs its operands where the profile selects it |
| `sycl/level3/gemm_accessed_packed.hpp` | The same form of `gemm_accessed`, over access bricks |
| `sycl/level3/packed_product.hpp` | `PackedProductKernel` and `packed_product`, the blocked product |
| `sycl/level3/packed_panel.hpp` | `pack_panel`, which copies a block of op(X) into unit-stride micro-panels |
| `sycl/level3/packed_micro_kernel.hpp` | The micro-kernel, which accumulates one register block |

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `operand_access.hpp`

### `OperandAccess`

A row step, a column step and a conjugation flag, resolved once on the host by `operand_access`
from a layout, a `Core::Transpose` and a leading dimension. `at(p, r, c)` returns element (r, c)
of op(X), conjugated when the operand was given as `ConjTrans`. A kernel written against it
carries no layout branch and no transpose branch in its inner loop, and `transposed()` gives the
access of the transpose without going back to the flags. The four (layout, transposed) cases are
two stride pairs; the table is under `operand_access` below.

### `operand_access`

The full table, checked element for element against `Kernels::Reference::gemm`. "Transposed" is
`Core::is_transposed(trans)`, which holds for `Trans` and `ConjTrans`:

| Layout | Transposed | Row step | Column step |
| --- | --- | --- | --- |
| `RowMajor` | no | `ld` | 1 |
| `RowMajor` | yes | 1 | `ld` |
| `ColumnMajor` | no | 1 | `ld` |
| `ColumnMajor` | yes | `ld` | 1 |

Transposing an operand and flipping its layout are the same operation, so the table is the XOR
of the two properties: `RowMajor` transposed and `ColumnMajor` untransposed address the same way,
and so do the other two.

## `symmetric_access.hpp`

### `SymmetricAccess`

Reads any element of a symmetric or Hermitian matrix with one read of the stored triangle: an
element on the other side is read at its mirror position, conjugated under the `Conjugated`
policy, and a diagonal element goes through the policy's `diagonal`, which drops the imaginary
part of a Hermitian diagonal. `symm` and `hemm` hand it to `gemm_accessed` as the access of
their symmetric factor, so a symmetric product runs on the same kernel as a general one.

## `band_access.hpp`, `packed_access.hpp`

### `BandAccess`, `PackedAccess`

"Packed" here is BLAS packed storage: one triangle of a matrix kept as a vector, the storage of
`spmv`, `spr`, `tpmv` and `tpsv`. It is not the packed panel of the CPU GEMM.

The same contract as `OperandAccess` for a band-stored and a packed operand: `at(p, r, c)`
returns element (r, c) of op(X). Both add `holds(r, c)`, which reports whether the element is
stored at all, and `first_column(r)` and `end_column(r, columns)`, which bound the stored part of
a row, so a kernel walks only what the storage holds. `band_access` takes the two band widths of
X, not of op(X); `triangle_band_access` and `packed_access` take the `Uplo` of X, not of op(X).
The Level 2 band and packed kernels of both back ends are written against these two bricks.

## `row_cursor.hpp`

### `RowCursor`

`row_cursor(r, c)` of a `BandAccess` or a `PackedAccess` returns the offset of element (r, c)
with the distance to the next column of the row and how that distance changes; `advance()` and
`retreat()` move one column, and `at(p, cursor)` reads the element. In a triangle packed by
columns the offset of (r, c) holds `c * (c + 1) / 2`, so a loop over c that calls `at(p, r, c)`
multiplies 64-bit integers once per element; the distance from column c to c + 1 is `c + 1`, or
`order - c - 1` when the lines shrink, so a cursor reaches the same offsets with two additions.
In band storage and in a triangle packed by rows the distance is constant and the slope is
zero. The `triangular_product` and `triangular_solve` of both back ends walk a row with a
cursor, and so do the reference `spr` and `spr2`.

## `symmetric_row.hpp`

### `SymmetricRow`

`symmetric_row<Conj>(access, n)` takes the `BandAccess` or `PackedAccess` of a stored triangle.
`for_each(p, i, visit)` calls `visit(value, column)` for every element of row i of the full
matrix in ascending column order: the columns left of the diagonal, the diagonal, then the
columns right of it. One side is read from the stored triangle and the other from its transpose
through `Conj`, each with a `RowCursor`. `first_column(i)` and `end_column(i)` bound the row.
The reference `symmetric_product` and the per-row path of the SYCL one sum a row through it; the
SYCL path that reduces a row across a work-group reads single elements and does not.

## `conjugation.hpp`

### `Unconjugated`, `Conjugated`

A symmetric routine and its Hermitian sibling differ in two places: one factor of each product is
conjugated, and the diagonal is real. A kernel takes the policy as a template parameter and calls
`Conj::apply` on that factor and `Conj::diagonal` on a diagonal element, so `syr` and `her`,
`syrk` and `herk`, `symv` and `hemv` are one kernel each. `Unconjugated` returns both unchanged.
The header also names the scalar type of a rank-1 or rank-k update under a policy: the real
component type under `Conjugated`, `T` otherwise.

## `givens.hpp`, `modified_givens.hpp`

### `givens_rotation`

Builds the plane rotation that maps (a, b) to (r, 0) by the algorithm of LAPACK 3.10 xROTG,
which scales its operands into a safe interval before it squares them. The square root is a
template parameter, `Root::apply`, so the reference kernel passes a host root and the SYCL kernel
a device one, and the arithmetic around it is written once. The complex form leaves `b` as it is
and returns a zero `z`.

### `modified_givens`

Builds the modified Givens transformation as the reference xROTMG does, with its rescaling factor
of `2^12` and the flag values -2, -1, 0 and 1. `K_MODIFIED_GIVENS_PARAMETERS` is the length, 5,
of the parameter array `rotmg` writes and `rotm` reads.

## `level3/packed_product_plan.hpp`

"Packed" here is the packed panel of the CPU GEMM: a block of an operand copied into contiguous,
aligned scratch before the micro-kernel reads it. It is not BLAS packed storage.

### `PackedRegisterBlock`

The rows and columns of C one micro-kernel call accumulates: 6 rows for a real accumulation type
and 4 for a complex one, and as many columns as fill `K_PACKED_ALIGNMENT`, 64 bytes, of that
type. A complex element is stored split, real parts first.

### `PackedProductPlan` and `plan_packed_product`

The block sizes, the work split and the scratch layout of one `m x n x k` product, computed on
the host from a `Device::Profile`. `selects_packed_product` chooses the packed kernel only on a
CPU device, and only when C spans at least `blocking.min_register_blocks` register blocks each
way and the two operands together hold at least `blocking.min_operand_bytes`. The plan is
computed twice from the same inputs: at record time, to lease `plan.scratch_bytes` of device
scratch (`Internal::allocate_packed_product_scratch`), and at launch. `packed_product_is_ready`
is the launch test: the profile selected the plan and the scratch holds its bytes. When it fails,
the GEMM forms run the kernel they ran before the packed path existed.

## `level3/channel_bias_epilogue.hpp`

The stride pair exists because "which axis is the channel" is a property of the *tensor layout*, not
of the bias operation. In NHWC the convolution GEMM is `[N*H*W, C_in*R*S] x [C_in*R*S, C_out]`, so
the channel index is the **column** of C, which means the row-vector broadcast of SushiAI's
fused GEMM with bias applies to convolution verbatim, with no new arithmetic. In
NCHW the same convolution transposes and the channel becomes the **row**. Encoding that as
`(row_stride, col_stride)` makes the second layout cost two integers on the host instead of a second
epilogue type, a second fused lowering recorder and a second set of tests.

It lives under `include/` rather than the private `modules/kernels/` tree for the reason
`modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/blas.hpp` states: `Dispatch::execute_gemm` (`<SushiBLAS/blas/dispatch.hpp>`)
instantiates the epilogue in a *consumer's* translation unit, and a template a public seam
instantiates cannot stay private.

### `ChannelBiasEpilogue`

`value + bias[row * row_stride + col * col_stride]`, with the stride pair set on the host. `(0, 1)`
makes the **column** the channel, which is what an NHWC convolution GEMM needs and what
the row-vector broadcast of SushiAI's fused GEMM with bias does; `(1, 0)` makes the **row** the
channel, which is what an NCHW one would need. Neither is baked in, so a second layout costs two
numbers rather than a second epilogue and a second fused lowering recorder.

Warning: The pointer is captured into a SYCL kernel. It must be device-accessible, must outlive the
submission, and the access set that reads it has to be declared on the task:
`Dispatch::gemm_epilogue(f).reads(bias_tensor)` is how, and `execute_gemm` cannot discover it (see
its documentation).

See Kernels::GemmIdentityEpilogue for the no-op this degenerates to at a zero bias, and
SushiBLAS::Dispatch::execute_gemm for the fusion contract and the host-readability constraint
`ForceReference` adds.

## `level3/gemm_epilogue.hpp`

It lives in its own header, under neither `reference/` nor `sycl/`, because it belongs to neither:
`Kernels::Reference::gemm` and `Kernels::SYCL::gemm` both take it as a template parameter and both
default to `GemmIdentityEpilogue`, and the public seam that exposes it, `Dispatch::execute_gemm`
(`<SushiBLAS/blas/dispatch.hpp>`), names it in its own signature. One definition, three
users, no direction of dependency between them.

### `GemmIdentityEpilogue`

Exists so that the fusion hook costs nothing when unused: the call inlines to the value itself and
the generated kernel is byte for byte the unfused one.

See Kernels::SYCL::gemm and SushiBLAS::Dispatch::execute_gemm for the fusion contract, and for the
training-correctness trap that comes with any *activation* epilogue.

## `reference/level3/gemm.hpp`

### `gemm`

Addressing an operand is two strides, not four cases: transposing it and flipping its layout are the
same operation, so `RowMajor`+transposed and `ColumnMajor`+untransposed index identically (as do the
other pair). That identity is the sanity check worth keeping in mind here: an earlier version of
this file had the `ColumnMajor` arm of A and B inverted, which made `layout` a no-op for both
operands and, with the column-major leading dimension the driver passes, read past the end of A
whenever m > k.

- `Epilogue`: Functor applied per element immediately before the store, exactly as in
  `SushiBLAS::Kernels::SYCL::gemm` and with the same `(value, row, col)` contract. It is here so
  that a *fused* GEMM has a ground truth at all: `ForceReference` mode is how this
  project judges a SYCL kernel, and a reference that dropped the epilogue would silently declare
  every fused result wrong. Defaults to `SushiBLAS::Kernels::GemmIdentityEpilogue`.

The store in the reference `gemm`: `beta == 0` must not read the prior C value, which may be
uninitialised. This matches standard BLAS semantics.

### Batch `gemm`

- `epilogue`: The per-element epilogue functor. Its `row`/`col` arguments are the coordinates within
  one matrix of the batch, matching `SushiBLAS::Kernels::SYCL::gemm`: a batch-aware epilogue would
  need a batch index the SYCL side does not supply either.

## `sycl/level3/gemm.hpp`

Tiling/blocking algorithm ported from portBLAS's gemm_local.hpp and gemm_no_local.hpp (see
NOTICE.md) into SushiBLAS's flat template-function convention: upstream's three-level-hierarchy
Tile<> and expression-tree executor collapse here into two fixed-shape kernels selected at launch
from a Device::Profile. See NOTICE.md's "Deviations" note for what this port still drops from
upstream and why none of it is a correctness concern.

Why two kernels and not one. On a device with real scratchpad memory, staging an A/B tile there and
reading it back K_ITEM_ROWS + K_ITEM_COLS times per K-step is the whole reason a tiled GEMM is fast. On
a CPU backend the "local memory" is the same cache the global pointer already reads through, so the
staging copies the data twice and every group_barrier is a real rendezvous between OS threads buying
nothing. portBLAS ships exactly this split; so does this file. The fact table both variants dispatch
on lives in `<SushiBLAS/launch/device_policy.hpp>` and knows nothing about GEMM: tile sizes are
chosen here, because only this file knows what its accumulator costs in registers.

### `Detail::load_operand`

The zero is what lets both kernels accumulate over a full padded tile instead of clamping the loop
to a ragged remainder: a zero contributes nothing to the running sum, so the arithmetic stays exact.

### `Detail::load_interior`

`gemm_local_tiled` stages a tile that lies wholly inside its operand through `load_interior`, with
no bounds check, and keeps `load_operand` for the edge tiles. Whether a tile is whole is the same
for every work-item of a group, so the branch never splits a sub-group. For `OperandAccess` the
64-bit origin of the tile does not change inside the staging loop, which leaves the compiler free
to compute it once; the offset inside the tile is 32-bit when `block_fits_int32` finds the
farthest one at or below `INT32_MAX`, which the host decides once per operand. Whether the compiler
does hoist the origin has not been read from the generated code. Any other brick reads through
`at()`.

### `Detail::K_LOCAL_ACC_BYTES`

256 B is what portBLAS's default `Tile<8,8,8,8>` spends on a float accumulator, and it is the number
the register block is derived from rather than a per-type list of specialisations. Deriving it from
`sizeof(T)` is the point: the previous fixed 4x4 block penalised `float`, the training path, to
accommodate `complex<double>`, a type almost nobody calls.

### `Detail::K_NO_LOCAL_ACC_BYTES`

A quarter of the local-tiled budget, because the constraint on a CPU backend is not the same one.
The OpenCL CPU runtime vectorises *across* work-items, packing neighbouring items into SIMD lanes,
so every scalar in a work-item's accumulator array is really a whole vector register. A 4x4 float
block is already 16 of them, the entire AVX2 register file, before the operand registers are
counted. Wider spills to stack, and a spilled accumulator is slower than a smaller one that stays
resident.

### `Detail::GemmLocalTile`

Everything here is `constexpr` because the accumulator is a fixed-size private array and the local
tiles are fixed-size local allocations; that is exactly why the two variants are two instantiations
selected by a plain runtime `if`, rather than one function threaded with `if constexpr`.

The register block comes out as 8x8 for `half` and `float` and 4x4 for `double`, `complex<float>`
and `complex<double>`: see `K_LOCAL_ACC_BYTES`.

### `Detail::GemmLocalTile::local_mem_bytes`

A `local_accessor` larger than the device's `local_mem_size` is a runtime launch failure, not a
compile error, so `gemm` checks this against the profile and falls back rather than crashing.

### `Detail::GemmNoLocalTile`

Only the register block is fixed at compile time. The work-group shape is not: with no local memory
and no barrier there is nothing whose size depends on it, so it is taken from
`Device::work_group_size` at launch and stays a runtime value.

Comes out as 4x4 for `half` and `float` and 2x2 for the 8- and 16-byte types: see
`K_NO_LOCAL_ACC_BYTES` for why smaller is right here.

### `Detail::group_shape`

Square-ish on purpose: a work-group covering a square-ish block of C reads the fewest operand
elements per element of C it produces, which is the whole arithmetic-intensity argument for register
blocking applied one level up.

`std::size_t rows = side;` in `group_shape`: a budget that is not a perfect square leaves room for a
second row of sub-groups, and taking it beats leaving half the group idle.

### `Detail::group_count`

The clamp matters for the legal degenerate cases: BLAS permits m or n of zero, and a zero-sized
`nd_range` is a launch failure. One group that stores nothing is the cheap, correct answer.

### `Detail::gemm_local_tiled`

Callers normally reach this through `gemm`, which picks between this and `gemm_no_local` from the
device profile and guarantees the local allocation fits. Called directly, those are the caller's
obligations.

At `const int64_t batch_id = static_cast<int64_t>(item.get_group(0));`: 64-bit only where a value
can exceed 2^31: the global coordinates, the strides, and the batch offsets. Everything scoped to a
tile is 32-bit, because 64-bit integer arithmetic is markedly slower on a GPU and a tile index never
comes close to needing it.

At `T* __restrict__ C = c + batch_id * str_c;`: only the output is __restrict__-qualified, and that
is deliberate. BLAS forbids C from overlapping A or B, so promising the compiler the store target is
unaliased is sound. `a` and `b` are deliberately NOT qualified: a caller computing C = A * A passes
one pointer for both, which is legal, and __restrict__ on both would make it undefined behaviour. Do
not "fix" this by adding it.

At `for (int64_t k0 = 0; k0 < k; k0 += Configuration::K_TILE_K)`: two buffers of each tile,
`K_BUFFERS`, and one barrier per K-step. The first tile is staged into buffer 0 before the loop,
behind a barrier. Step i stages the next tile into the buffer it does not read, multiplies out the
buffer it does, and ends on one barrier. The buffer step i writes was last read in step i - 1,
whose closing barrier every work-item has passed; the buffer it reads was written in step i - 1,
before that barrier. The local tile is twice the size of one buffer pair, and
`GemmLocalTile::local_mem_bytes()` counts both, so every check that a device hosts the tile
counts both too. Its effect on the RTX 3080 Ti is unmeasured; RW-92 measures it.

Bank-conflict arithmetic for the A read, worked out for float (32 banks of 4 bytes) at this file's
widest configuration: K_ITEM_ROWS = K_ITEM_COLS = 8, K_WG_ROWS = K_WG_COLS = 8, K_TILE_K = 16, K_TILE_A_LD = 17.

A work-group's linear local id is local_row * K_WG_COLS + local_col, so a 32-lane sub-group spans 32 /
K_WG_COLS = 4 distinct local_row values. The 8 lanes sharing a local_row read the identical address
and are served by a broadcast, so only those 4 matter.

K_TILE_A_LD = 16 (unpadded): stride 16 floats, 16 mod 32 == 16, banks {0, 16, 0, 16}: 2-way.

Contiguous ownership (the shape this kernel started from): stride K_ITEM_ROWS * K_TILE_K == 8 * 16 ==
128, and 128 mod 32 == 0, so all 4 land on one bank: 4-way.

K_TILE_A_LD = 17 (this code): stride 17 floats and gcd(17, 32) == 1, so the 4 lanes take banks {0, 17,
2, 19}: all distinct, conflict-free. It holds on AMD's 64-lane sub-groups too, where the 8 distinct
local_row give {0,17,2,19,4,21,6,23}.

The B read is conflict-free at any leading dimension: the strided ownership puts neighbouring
local_col on neighbouring floats of one tile row, so a sub-group's 8 distinct local_col cover 8
consecutive banks. Contiguous ownership would have spaced them K_ITEM_COLS == 8 apart: banks {0, 8,
16, 24} twice, a 2-way conflict. `K_TILE_B_LD` pads the B tile for the store instead (below).

Register-block ownership is therefore a strided lattice, not a contiguous block: work-item
(local_row, local_col) owns tile rows local_row + ii*K_WG_ROWS and tile columns local_col +
jj*K_WG_COLS. Neighbouring work-items touch neighbouring elements, which is what makes these reads
conflict-free and the C store coalesced.

The staging loops in `gemm_local_tiled` are cooperative: the whole group fills `K_BLOCK_M x K_TILE_K` of
A and `K_TILE_K x K_BLOCK_N` of B, each work-item taking a grid-strided share. Both go through
`Detail::stage_block`, which walks its block along the axis whose elements are adjacent in memory,
so neighbouring work-items read neighbouring addresses: rows first when a column of op(X) is
contiguous, as for a ColumnMajor NoTrans operand, and columns first otherwise. For A the rows are
rows of C and the columns are the depth; for B the rows are the depth. An operand read through
another brick, such as `SymmetricAccess`, walks columns first. The host fills one
`Detail::StagePlan` per operand: the walking order, whether offsets inside a block fit 32 bits,
and whether whole blocks take 128-bit loads.

A whole block of an `OperandAccess` operand in FLOAT32, FLOAT64 or HALF is read by `load_lanes`,
one `sycl::vec` of 16 bytes per load, when `stages_by_vector` finds the base address 16-byte
aligned and the leading dimension and the batch stride a multiple of the lanes. Block origins are
multiples of 16 rows or columns, so every load then starts aligned. The lanes run along the
contiguous axis and land in the tile one element each. A complex operand, a conjugated one and an
edge block are read by element. For B the depth-first lanes keep the bank spread of
`K_TILE_B_LD`; for A the rows-first lanes write four rows 17 floats apart per load, a 4-way bank
conflict on the store that this layout has not been measured against.

A depth-first B store writes 16 neighbouring work-items down one tile column. With an unpadded
leading dimension of `K_BLOCK_N` == 64 floats, all 16 land on one bank. `K_TILE_B_LD` is
`K_BLOCK_N + 2` == 66, so in float the 32 lanes of a sub-group take banks `(2 kk + cc) mod 32`:
the first 16 lanes the even banks from `cc`, the next 16 the odd ones from `cc + 1`, with no
conflict. A column-first store writes neighbouring floats of one row, which no padding changes.

The `kk` loop in `gemm_local_tiled` sums the full `K_TILE_K` width rather than clamping to the true
remainder on the last tile. The out-of-range elements the staging loops wrote are zero, so the sum
is still exact.

The store in `gemm_local_tiled`: `beta == 0` must not read the prior C value, which may be
uninitialised. This matches standard BLAS semantics.

### `Detail::gemm_no_local`

This is the variant a CPU backend wants. Its "local memory" is the same cache the global pointer
already reads through, so the tiling the other variant does would copy every element twice and pay a
real thread rendezvous per K-step for the privilege. Reuse still happens, an K_ITEM_ROWS x K_ITEM_COLS
register block reads (K_ITEM_ROWS + K_ITEM_COLS) operand elements per (K_ITEM_ROWS * K_ITEM_COLS) FMAs, it
just happens out of cache instead of out of scratchpad.

At `const int64_t row0 =`: same strided-lattice ownership as the local-tiled variant: this
work-item's rows are row0 + ii*wg_rows and its columns col0 + jj*wg_cols. The reason is different
here, there is no local memory to conflict on, but the payoff is the same and matters more. A CPU
backend vectorises across neighbouring work-items, so neighbouring items reading and writing
neighbouring addresses is what turns the B load and the C store into contiguous vector accesses
instead of a gather and a scatter.

In `gemm_no_local` only the output pointer `C` is `__restrict__`-qualified. `a` and `b` must not be,
for the reason given under `Detail::gemm_local_tiled`.

The `ii` loop in `gemm_no_local`: the zero fill in `load_operand` doubles as the tail guard. An
out-of-range row or column contributes nothing, so no ragged-remainder loop is needed.

The store in `gemm_no_local`: `beta == 0` must not read the prior C value, which may be
uninitialised. This matches standard BLAS semantics.

### `Detail::GemmDepthSplitTile`

The register block is the one `GemmNoLocalTile` takes, 4x4 for `half` and `float` and 2x2 for the
8- and 16-byte types, because this variant reads its operands from global memory too.
`K_MIN_ITEM_DEPTH` is 16: below 16 K-steps per work-item the fold of the partial sums, eight
barriers at a width of 256, costs more than the sums. `K_MAX_COLUMN_LANES` is 16, so the
work-items of one K-step read at most 64 neighbouring `float` values of op(B).

### `Detail::DepthSplitPlan` and `Detail::plan_depth_split`

Both tiled variants divide a product over the elements of C and give each work-item the whole K
extent. A product with a small C and a long K then launches a handful of work-groups, each
walking K alone, and most of the device idles. The weight gradient of a convolution is this
shape: C is output channels by kernel elements, and K is batch times output height times width.

The plan has a non-zero width, and so selects DEPTH_SPLIT, when all of these hold:

- the device has dedicated local memory, so a CPU backend never takes the variant;
- the device hosts `GemmDepthSplitTile::local_mem_bytes` of local memory at
  `Reduce::group_width`;
- LOCAL_TILED would launch fewer work-groups than the device has compute units;
- K is at least the width times `K_MIN_ITEM_DEPTH`.

A work-group has `width` work-items, `column_lanes` of them side by side along a row of op(B)
and `depth_lanes()` of them along K. `column_lanes` is 1 unless neighbouring columns of op(B)
are neighbouring elements, which `Detail::has_contiguous_rows` reads from an `OperandAccess`
and denies for every other access brick. When they are, `column_lanes` doubles up to
`K_MAX_COLUMN_LANES` for as long as the launch keeps one work-group per compute unit.

### `Detail::gemm_depth_split`

One work-group produces a block of C that is one register block tall and `column_lanes` register
blocks wide. Work-item `lane` has column lane `lane % column_lanes` and depth lane
`lane / column_lanes`. It owns one register block and takes the K-steps `depth_lane`,
`depth_lane + depth_lanes`, and so on.

The two lane kinds exist for the two ways an operand lies in memory. With one column lane,
neighbouring work-items take neighbouring K-steps, which are neighbouring addresses of an
operand that is contiguous along K. With several, the work-items of one K-step read neighbouring
columns of one row of op(B), which is one contiguous run when K is the leading axis in memory.

Each work-item keeps its block's partial sums in registers and writes them to local memory. The
group folds the depth lanes of each column lane in a halving tree of fixed order, so one device
repeats a result to the bit. The work-items of depth lane 0 store their blocks.

The sum of one element is formed in a different order than LOCAL_TILED forms it, so the two
variants differ in the last bits of a `float` result.

Measured on an RTX 3080 Ti in FLOAT32 by `benchmark_gemm`, median of 10, against LOCAL_TILED on
the same product:

| C | K | K in memory | LOCAL_TILED | DEPTH_SPLIT |
| --- | --- | --- | --- | --- |
| 32 x 27 | 739328 | contiguous | 383 ms | 1.4 ms |
| 64 x 288 | 184832 | contiguous | 122 ms | 3.8 ms |
| 128 x 576 | 46208 | contiguous | 30 ms | 4.0 ms |
| 32 x 27 | 86528 | leading axis | 57 ms | 0.37 ms |
| 64 x 576 | 21632 | leading axis | 16 ms | 0.58 ms |
| 128 x 1152 | 5408 | leading axis | 4.1 ms | 0.46 ms |

### `gemm`

Resolves the four (layout, transpose) combinations into plain row/column strides on the host, through
`operand_access`, so the innermost loop carries no layout branch and no extra template
instantiation, then picks one of three kernel shapes from the device profile:

- **DEPTH_SPLIT** when `Detail::plan_depth_split` returns a width: a GPU, a small C and a
long K.
- **LOCAL_TILED** when the device has genuine scratchpad memory and the
tile fits it. Cooperative A/B staging, bank-padded, barrier-synchronised.
- **NO_LOCAL** otherwise, and always on a CPU backend. No `local_accessor`,
no barrier, a smaller register block and the work-group size the profile
asks for.

The choice is logged at `SB_LOG_DEBUG` so it is observable rather than inferred, and a device that
declares scratchpad memory but cannot fit the tile or host the work-group falls back with a warning
instead of failing the launch.

- `Epilogue`: A functor applied to each element of C immediately before it is stored: `T
  operator()(T value, int64_t row, int64_t col) const`. `value` is the fully formed `alpha * op(A) *
  op(B) + beta * C` element and `row`/`col` are its coordinates in C. The coordinates are the point:
  without them an epilogue can only do elementwise work, and with them a bias broadcast (`value +
  bias[col]`) becomes expressible. This is the library's highest-value fusion hook: it lets the
  layer above fold a bias-add and an activation into GEMM instead of paying two more kernel
  launches. Defaults to `SushiBLAS::Kernels::GemmIdentityEpilogue`, which costs nothing.

Reachable from outside this library through `SushiBLAS::Dispatch::execute_gemm`, which is the
supported seam; this entry point is the kernel underneath it.

Warning: An **activation** epilogue used in a *forward training* pass is silently wrong unless it
also writes its pre-activation input to an auxiliary buffer, or is invertible. The backward pass has
to differentiate the activation at its input; if the only thing that reached memory is the
activation's output, that input does not exist any more and the gradient is computed at the wrong point.
Nothing throws, nothing produces NaN: the model simply trains towards the wrong place. This is
exactly why cuBLASLt exposes `GELU_AUX_BIAS`, which spills the pre-activation, and not merely
`GELU_BIAS`. Inference-only fusion carries no such constraint, and neither does a bias-only
epilogue, which is invertible.

- `beta`: Scale applied to the incoming C; exactly zero skips reading C.
- `c`: Base pointer of C. Must not overlap A or B.
- `deps`: Events this submission must depend on.
- `epilogue`: The per-element epilogue functor.

Returns: The submitted work's event.

At `if (profile.has_dedicated_local_memory && tile_hosted)`: all three variants are named
from this one function, so each is instantiated and each reaches the device image whichever branch a
given machine takes at runtime: there is no dead-stripping hazard to guard against, and no
configuration in which one of them stops compiling unnoticed.

## `sycl/level3/gemm_joint_matrix.hpp`

`gemm_joint_matrix` computes a HALF GEMM on the tensor cores of a CUDA device through
the oneAPI matrix extension, accumulating in FLOAT32 and rounding to HALF once, at the store. The
local-tiled kernel stays the fallback on every device and is the truth its test compares against
bit for bit.

### `Detail::GemmJointMatrixTile`

One matrix-unit product is 16 x 16 x 16. A sub-group of 32 work-items, one CUDA warp, owns a
32 x 32 block of C as 2 x 2 fragments, so each A fragment and each B fragment it loads feeds two
products. A work-group holds 2 x 2 sub-groups and owns a 64 x 64 block.

### `Detail::joint_matrix_fits`

A fragment load reads a 16 x 16 block of op(X) from global memory with one stride, so the kernel
asks every extent to be a multiple of 16, and each operand to have a contiguous axis, a leading
dimension that is a multiple of 8 HALF elements, a base address aligned to 32 bytes and a batch
stride that is a multiple of 16 elements. With those, every fragment origin is 32-byte aligned.
A product that fails any of them runs on the local-tiled kernel.

### `Detail::gemm_joint_matrix_as`

The layouts of A and B are template arguments, because a `joint_matrix` names its layout in its
type; `gemm_joint_matrix` picks one of the four instantiations from the access strides. Each
sub-group loads its two A and two B fragments per K-step straight from global memory and
accumulates four FLOAT32 fragments. At the end it stores each fragment, one at a time, into a
32-byte aligned `JointMatrixStoreTile` taken with `group_local_memory_for_overwrite`, whose
alignment follows its type, and its 32 work-items scale, round and write the 256 elements through
`c_access`, so C may have either layout. `alpha`, `beta` and the epilogue apply there, and
`beta == 0` never reads C.

The body is compiled for NVPTX alone, under `#if !defined(__SYCL_DEVICE_ONLY__) ||
defined(__NVPTX__)`. The SPIR-V image of the same translation unit carries an empty body, so an
OpenCL or Level Zero runtime never meets a cooperative matrix instruction when it builds the
module, and `Profile::supports_half_matrix` is false on every backend but CUDA, so the empty body
is never launched. Its speed on the RTX 3080 Ti is unmeasured until RW-92 runs; the plan reverts
the routing if it does not beat the local-tiled kernel.

## `sycl/level3/gemm_packed.hpp`, `gemm_accessed_packed.hpp`

Each declares one more overload of an existing entry point, `gemm` and `gemm_accessed`, that
takes a scratch pointer and its byte count. It plans the product with `plan_packed_product` and,
when `packed_product_is_ready` holds, launches `packed_gemm`; otherwise it calls the overload
without scratch. `Dispatch::execute_gemm` calls the scratch form. `GemmStore` is the store the
packed kernel hands each finished sum to: it writes `epilogue(alpha * sum + beta * C)` and does
not read C when `beta` is exactly zero.

## `sycl/level3/packed_product.hpp`

### `PackedProductKernel`

One work-item per worker of the plan. Each owns one scratch slot, which holds a packed panel of
A, a packed panel of B and an accumulation tile, and computes its macro-blocks of C in ascending
order. The operands are read through access bricks, so the same kernel serves a general, a
symmetric and a Hermitian first factor. A finished sum goes to a `Store`: `wants(row0, rows,
col0, cols)` says whether a macro-block is written at all, which is how `syrk` skips the
triangle it does not store, and `operator()` writes one element. `c` may overlap neither operand
nor the scratch.

## `sycl/level3/packed_panel.hpp`

### `pack_panel`

Copies `count` lines of op(X) over `depth` terms into micro-panels of `Width` lines at unit
stride, so the micro-kernel reads both factors contiguously whatever the operand's layout and
transpose. The last micro-panel may be short, and its missing lines are written as zero.
`has_adjacent_lines` only picks the order of the copy, to walk the axis the source stores
contiguously; it does not change what is written.

## `sycl/level3/packed_micro_kernel.hpp`

Adds `depth` outer products of one slice of the A panel and one slice of the B panel to a
`K_ROWS x K_COLS` micro-tile, in ascending term order. The real form works on `sycl::vec` rows
loaded from memory aligned to `K_PACKED_ALIGNMENT`. The complex form keeps real and imaginary
parts in separate rows and adds each term by the four-product formula of
`arithmetic/complex_arithmetic.hpp`.
