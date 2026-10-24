# Tests of signal

The tests of the `signal` module, built into `sushiblas_signal_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_signal/test_signal_convolve_1d.cpp` | Tests SignalOps::convolve_1d in both forced modes against hand values, a formula and a rounding bound, and its refusals |

## test_signal/test_signal_convolve_1d.cpp

Integer data is compared exactly: the largest sum is below 2048, so HALF holds every value, and
the two modes agree with a formula loop written apart from the kernels. Fractional data is
compared with a bound, because the two modes sum differently. The device adds `K` products in
`Accumulator<T>` and rounds each FLOAT32 and FLOAT64 product once more, which is `2K` roundings of
that accumulator's unit roundoff; a HALF product is exact in FLOAT32. The reference sums in
`ReferenceSum`, in double, so its bound is `K + 2` roundings of double for HALF and FLOAT32 and
five for FLOAT64, whose sum is compensated. Each mode then rounds once to the element type, which
the bound adds as `unit_roundoff<T>() * |sum|`. The reference of the operation is not the oracle:
the oracle is `TruthSum`, an exact sum of the products. `TheDeviceRepeatsToTheBit` holds the
fixed `k` order: two device runs on one device write the same bits.
`TheNodeCarriesTheNameAndTheDeterminismClass` reads the name and the class of a call with options
and of one without from the runtime plan, and `AnEmptyNameOrAnUnknownClassIsRefused` covers the
refusals of the options.
