# Tests of blend

The tests of the `blend` module, built into `sushiblas_blend_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_blend/test_blend_transforms.cpp` | Tests BlendOps::blend_transforms in both forced modes against hand values, an independent formula and a rounding bound, and its refusals |

## test_blend/test_blend_transforms.cpp

Every typed case runs the five shapes of `table_cases()`: 3×4 and 4×4 transforms, positions and
homogeneous points, `J` = 1 and `J` = 4.

Three kinds of weights are drawn. Integer weights from -2 to 2 do not sum to one. Binary fractions
(`0.5, 0.25, 0.125, 0.125`, rotated per point, or 1 when `J` = 1) sum to one. Fractional weights do
not sum to one and no type holds them. With integer data, and with the binary weights, every
product and sum is a multiple of one eighth below 2048 / 8, which HALF holds, so the library must
equal a formula loop written apart from the kernels exactly, in both modes. The identity case
builds identity transforms with the binary weights and expects each point back.

Fractional data is compared with a bound, because the two modes sum differently. The oracle is
`TruthSum`, an exact sum of the products `(weight * entry) * value`; the factor in brackets is
formed in double, which is exact for HALF and FLOAT32 and rounds once for FLOAT64. One output value
is a sum over `j` of a weight times a four-term row product. The device forms the row product in
`Accumulator<T>`: four products and three additions, which `4` roundings of that accumulator's unit
roundoff cover. The weight multiplies it, one more rounding, and the `J - 1` additions that join the
terms add up to `J - 1` more. That is `J + 4` roundings, and the bound uses `J + 5` for the oracle's
own rounding of FLOAT64. The reference runs the same operations in double, so it uses the same
count with the unit roundoff of double. Each mode then rounds once to the element type, which the
bound adds as `unit_roundoff<T>() * |value|`. The bound is on `Σ |weight * entry * value|`, the
magnitude `TruthSum` returns, so a cancelling sum gets the slack it needs and a large one does not
hide an error. The reference of the operation is not the oracle.

`AnIndexOutsideTheTransformsAddsNoTerm` puts indices below zero and at or above `T` in both
index types, including the limits of INT32 and values beyond INT32 for INT64, and expects the
formula that skips them; a point whose indices are all invalid must come out as zeros and not as
the value the output held before. Whether a device read through such an index would fault is not
tested: the test shows the term is skipped, and the kernel code shows no transform is addressed
through it. `TheDeviceRepeatsToTheBit` holds the fixed `j` order: two device runs on one device
write the same bits.

The refusal test records one well-formed call as the control, then one call per malformed operand
and checks that the message names the operation and the operand. Every malformed operand is a
dtype, a rank, an extent, a zero count, a transposed view or an overlapping output. A value in the
indices cannot be refused at record time, which the module README states.
