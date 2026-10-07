# Kernel headers

`include/SushiBLAS/kernels` holds the GEMM kernels and epilogues that a public seam instantiates in
a consumer's translation unit: the reference kernel, the SYCL kernel and the two epilogue functors.
The accumulator trait they share is in [../support/README.md](../support/README.md).

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `blas/level3/channel_bias_epilogue.hpp`

The stride pair exists because "which axis is the channel" is a property of the *tensor layout*, not
of the bias operation. In NHWC the convolution GEMM is `[N*H*W, C_in*R*S] x [C_in*R*S, C_out]`, so
the channel index is the **column** of C, which means the row-vector broadcast the existing
`Ops::GEMM_BIAS` family already performs applies to convolution verbatim, with no new arithmetic. In
NCHW the same convolution transposes and the channel becomes the **row**. Encoding that as
`(row_stride, col_stride)` makes the second layout cost two integers on the host instead of a second
epilogue type, a second fused lowering recorder and a second set of tests.

It lives under `include/` rather than the private repo-root `kernels/` tree for the reason
`kernels/blas.hpp` states: `Dispatch::execute_gemm` (`<SushiBLAS/engine/blas/dispatch.hpp>`)
instantiates the epilogue in a *consumer's* translation unit, and a template a public seam
instantiates cannot stay private.

### `ChannelBiasEpilogue`

`value + bias[row * row_stride + col * col_stride]`, with the stride pair set on the host. `(0, 1)`
makes the **column** the channel, which is what an NHWC convolution GEMM needs and what
`Ops::GEMM_BIAS`'s existing row-vector broadcast already does; `(1, 0)` makes the **row** the
channel, which is what an NCHW one would need. Neither is baked in, so a second layout costs two
numbers rather than a second epilogue and a second fused lowering recorder.

Warning: The pointer is captured into a SYCL kernel. It must be device-accessible, must outlive the
submission, and the access set that reads it has to be declared on the task:
`Dispatch::gemm_epilogue(f).reads(bias_tensor)` is how, and `execute_gemm` cannot discover it (see
its documentation).

See Kernels::GemmIdentityEpilogue for the no-op this degenerates to at a zero bias, and
SushiBLAS::Dispatch::execute_gemm for the fusion contract and the host-readability constraint
FORCE_REFERENCE adds.

## `blas/level3/gemm_epilogue.hpp`

It lives in its own header, under neither `reference/` nor `sycl/`, because it belongs to neither:
`Kernels::Reference::gemm` and `Kernels::SYCL::gemm` both take it as a template parameter and both
default to `GemmIdentityEpilogue`, and the public seam that exposes it, `Dispatch::execute_gemm`
(`<SushiBLAS/engine/blas/dispatch.hpp>`), names it in its own signature. One definition, three
users, no direction of dependency between them.

### `GemmIdentityEpilogue`

Exists so that the fusion hook costs nothing when unused: the call inlines to the value itself and
the generated kernel is byte for byte the unfused one.

See Kernels::SYCL::gemm and SushiBLAS::Dispatch::execute_gemm for the fusion contract, and for the
training-correctness trap that comes with any *activation* epilogue.

## `reference/blas/level3/gemm.hpp`

### `gemm`

Addressing an operand is two strides, not four cases: transposing it and flipping its layout are the
same operation, so ROW_MAJOR+transposed and COLUMN_MAJOR+untransposed index identically (as do the
other pair). That identity is the sanity check worth keeping in mind here: an earlier version of
this file had the COLUMN_MAJOR arm of A and B inverted, which made `layout` a no-op for both
operands and, with the column-major leading dimension the driver passes, read past the end of A
whenever m > k.

- `Epilogue`: Functor applied per element immediately before the store, exactly as in
  `SushiBLAS::Kernels::SYCL::gemm` and with the same `(value, row, col)` contract. It is here so
  that a *fused* GEMM has a ground truth at all: `FORCE_REFERENCE` mode is how this
  project judges a SYCL kernel, and a reference that dropped the epilogue would silently declare
  every fused result wrong. Defaults to `SushiBLAS::Kernels::GemmIdentityEpilogue`.

The store in the reference `gemm`: `beta == 0` must not read the prior C value, which may be
uninitialised. This matches standard BLAS semantics.

### Batch `gemm`

- `epilogue`: The per-element epilogue functor. Its `row`/`col` arguments are the coordinates within
  one matrix of the batch, matching `SushiBLAS::Kernels::SYCL::gemm`: a batch-aware epilogue would
  need a batch index the SYCL side does not supply either.

## `sycl/blas/level3/gemm.hpp`

Tiling/blocking algorithm ported from portBLAS's gemm_local.hpp and gemm_no_local.hpp (see
NOTICE.md) into SushiBLAS's flat template-function convention: upstream's three-level-hierarchy
Tile<> and expression-tree executor collapse here into two fixed-shape kernels selected at launch
from a Device::Profile. See NOTICE.md's "Deviations" note for what this port still drops from
upstream and why none of it is a correctness concern.

Why two kernels and not one. On a device with real scratchpad memory, staging an A/B tile there and
reading it back item_rows + item_cols times per K-step is the whole reason a tiled GEMM is fast. On
a CPU backend the "local memory" is the same cache the global pointer already reads through, so the
staging copies the data twice and every group_barrier is a real rendezvous between OS threads buying
nothing. portBLAS ships exactly this split; so does this file. The fact table both variants dispatch
on lives in `<SushiBLAS/support/device_policy.hpp>` and knows nothing about GEMM: tile sizes are
chosen here, because only this file knows what its accumulator costs in registers.

### `Detail::OperandStrides`

The point of carrying these instead of a (layout, transposed) pair is that the four combinations of
the two are only ever two strides, and resolving them once on the host removes a branch from the
innermost loop of every kernel in the file without adding a single template instantiation.

### `Detail::strides_for`

The full table, verified element for element against `Kernels::Reference::gemm` (which is this
operation's specification):

| layout       | transposed | row stride | col stride |
|--------------|------------|------------|------------|
| ROW_MAJOR    | false      | `ld`       | 1          |
| ROW_MAJOR    | true       | 1          | `ld`       |
| COLUMN_MAJOR | false      | 1          | `ld`       |
| COLUMN_MAJOR | true       | `ld`       | 1          |

Transposing an operand and flipping its layout are the same operation, so the table is just the XOR
of the two flags, which is the sanity check worth remembering: ROW_MAJOR+transposed and
COLUMN_MAJOR+untransposed are literally the same addressing, and so are ROW_MAJOR+untransposed and
COLUMN_MAJOR+transposed.

### `Detail::load_operand`

The zero is what lets both kernels accumulate over a full padded tile instead of clamping the loop
to a ragged remainder: a zero contributes nothing to the running sum, so the arithmetic stays exact.

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

At `for (int64_t k0 = 0; k0 < k; k0 += Cfg::tile_k)`: no double buffering, deliberately. Alternating
between two local buffers would elide one of the two barriers in the K-step loop, but the win over a
correctly bank-padded single-buffered kernel is modest, and this project has no GPU to measure it
on: the only reachable device is an OpenCL CPU, which takes the no-local variant and has no barriers
to elide. Restructuring a barrier sequence that cannot be verified is the wrong risk to take; a
stated omission beats an unmeasurable optimisation.

Bank-conflict arithmetic for the A read, worked out for float (32 banks of 4 bytes) at this file's
widest configuration: item_rows = item_cols = 8, wg_rows = wg_cols = 8, tile_k = 16, tile_a_ld = 17.

A work-group's linear local id is local_row * wg_cols + local_col, so a 32-lane sub-group spans 32 /
wg_cols = 4 distinct local_row values. The 8 lanes sharing a local_row read the identical address
and are served by a broadcast, so only those 4 matter.

tile_a_ld = 16 (unpadded): stride 16 floats, 16 mod 32 == 16, banks {0, 16, 0, 16}: 2-way.

Contiguous ownership (the shape this kernel started from): stride item_rows * tile_k == 8 * 16 ==
128, and 128 mod 32 == 0, so all 4 land on one bank: 4-way.

tile_a_ld = 17 (this code): stride 17 floats and gcd(17, 32) == 1, so the 4 lanes take banks {0, 17,
2, 19}: all distinct, conflict-free. It holds on AMD's 64-lane sub-groups too, where the 8 distinct
local_row give {0,17,2,19,4,21,6,23}.

The B read needs no padding: the strided ownership puts neighbouring local_col on neighbouring
floats, so a sub-group's 8 distinct local_col cover 8 consecutive banks. Contiguous ownership would
have spaced them item_cols == 8 apart: banks {0, 8, 16, 24} twice, a 2-way conflict.

Register-block ownership is therefore a strided lattice, not a contiguous block: work-item
(local_row, local_col) owns tile rows local_row + ii*wg_rows and tile columns local_col +
jj*wg_cols. Neighbouring work-items touch neighbouring elements, which is what makes these reads
conflict-free and the C store coalesced.

The staging loops in `gemm_local_tiled` are cooperative: the whole group fills `block_m x tile_k` of
A and `tile_k x block_n` of B, each work-item taking a grid-strided share.

The `kk` loop in `gemm_local_tiled` sums the full `tile_k` width rather than clamping to the true
remainder on the last tile. The out-of-range elements the staging loops wrote are zero, so the sum
is still exact.

The store in `gemm_local_tiled`: `beta == 0` must not read the prior C value, which may be
uninitialised. This matches standard BLAS semantics.

### `Detail::gemm_no_local`

This is the variant a CPU backend wants. Its "local memory" is the same cache the global pointer
already reads through, so the tiling the other variant does would copy every element twice and pay a
real thread rendezvous per K-step for the privilege. Reuse still happens, an item_rows x item_cols
register block reads (item_rows + item_cols) operand elements per (item_rows * item_cols) FMAs, it
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

### `gemm`

Resolves the four (layout, transpose) combinations into plain row/column strides on the host, see
`Detail::strides_for`, so the innermost loop carries no layout branch and no extra template
instantiation, then picks one of two kernel shapes from the device profile:

- **LOCAL_TILED** when the device has genuine scratchpad memory and the
tile fits it. Cooperative A/B staging, bank-padded, barrier-synchronised.
- **NO_LOCAL** otherwise, and always on a CPU backend. No `local_accessor`,
no barrier, a smaller register block and the work-group size the profile
asks for.

The choice is logged at `SB_LOG_INFO` so it is observable rather than inferred, and a device that
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
activation's output, that input no longer exists and the gradient is computed at the wrong point.
Nothing throws, nothing produces NaN: the model simply trains towards the wrong place. This is
exactly why cuBLASLt exposes `GELU_AUX_BIAS`, which spills the pre-activation, and not merely
`GELU_BIAS`. Inference-only fusion carries no such constraint, and neither does a bias-only
epilogue, which is invertible.

- `beta`: Scale applied to the incoming C; exactly zero skips reading C.
- `c`: Base pointer of C. Must not overlap A or B.
- `deps`: Events this submission must depend on.
- `epilogue`: The per-element epilogue functor.

Returns: The submitted work's event.

At `if (profile.has_dedicated_local_memory && tile_fits && group_fits)`: both variants are named
from this one function, so both are instantiated and both reach the device image whichever branch a
given machine takes at runtime: there is no dead-stripping hazard to guard against, and no
configuration in which one of them stops compiling unnoticed.
