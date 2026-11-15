# Tests of broadcast

The tests of the `broadcast` module, built into `sushiblas_broadcast_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `compile_broadcast_checks.cpp` | Checks that `broadcast_checks.hpp` compiles on its own; compiled, never run |
| `compile_broadcast_operand.cpp` | Checks that `broadcast_operand.hpp` compiles on its own; compiled, never run |
| `benchmark_broadcast.cpp` | Times BroadcastOps::mul beside channel_affine and beside a per-index launch that divides |
| `test_broadcast.cpp` | Tests BroadcastOps::add, sub, mul and div along each axis against a host loop, and C as A |
| `test_dispatch.cpp` | Tests Dispatch::execute_broadcast with several vectors and a bound scalar, its refusals, and that the host path equals the device bit for bit |

## test_broadcast.cpp

The expectation is a host loop in double over the same flat index, rounded once into the element
type. The matrix holds multiples of 0.25 between -0.5 and 2.0, and so does the vector for `add`,
`sub` and `mul`: every sum and product is a multiple of 1/16 below 8, which HALF, FLOAT32 and
FLOAT64 all hold exactly, so the comparison is exact and no tolerance applies. The divisors of
`div` are powers of two, which makes every quotient exact too. A case runs under `ForceReference`
and under `ForceSYCL` against the same expectation, so the two paths are each checked against the
host loop and not against each other.

The shape rows give ranks 2, 3 and 4. Each row names the vector of each axis by its own extents:
`[2, 3]` against `[2, 3, 4]` is a leading vector of rank 2 and `[3, 4]` a trailing table of rank 2,
the `[B, T, D] + [T, D]` use. HALF cases are skipped where the device reports no fp16 aspect, and
FLOAT64 cases where it reports no fp64.

The two aliasing cases run `C` as the very `Tensor` object `A` and as a copy of it, which is
another view of the same bytes.

## test_dispatch.cpp

The mixed call, one leading and two trailing vectors with a bound scalar, is the layer
normalisation shape. Its values are multiples of 0.25 so the host loop and the call agree even if
the device contracts `(x - l) * t + u * s` into a fused multiply-add. The first trailing vector is a
`[4, 5]` table and the second a flat `[20]`, which states that only the element count of a vector
matters.

`TheReferenceEqualsTheDeviceBitForBit` uses values that are not exact in binary and a functor,
`x / l + t`, that no device fuses, so equality of the bytes proves the two paths evaluate one
expression in one order. A vector that is a slice of an arena is read at its offset.

Each refusal has its own case and asserts on the operand it names: a count that is no prefix
product (four rows: a count that divides but is no prefix, one that does not divide, a trailing
count whose outer extent is no prefix, and two counts whose product is not the element count), two
leading vectors of different counts, a `ColumnMajor` `A`, vector and `C`, a partial overlap of `C`
with `A`, an overlap of `C` with a vector, a vector and an output of another dtype, and a null
vector. The `ColumnMajor` cases copy a row-major tensor and set its layout, so the check that
refuses is the layout check and not the stride check behind it.

## benchmark_broadcast.cpp

Prints, for a FLOAT32 matrix of `[8192, 384]` and one of `[384, 65536]`, the rate of `mul` with a
leading and with a trailing vector, of `channel_affine` with a scale of ones and a shift of zeros,
and of the same two products as one index per element through `IndexOps::for_each`, whose functor
divides or takes a remainder where the seam's `range<2>` launch does not. The rate counts one read
and one write of the matrix. `channel_affine` has no leading form, so its row stands beside the
trailing rows only.
