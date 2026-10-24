# Tests of elementwise

The tests of the `elementwise` module, built into `sushiblas_elementwise_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_dispatch.cpp` | Tests that the functor seams give the same result on the host path as on the SYCL path |
| `test_elementwise/test_abs.cpp` | Tests ElementwiseOps::abs against fixed expected values |
| `test_elementwise/test_acos.cpp` | Tests ElementwiseOps::acos against fixed expected values |
| `test_elementwise/test_acosh.cpp` | Tests ElementwiseOps::acosh against fixed expected values |
| `test_elementwise/test_add.cpp` | Tests ElementwiseOps::add against fixed expected values |
| `test_elementwise/test_asin.cpp` | Tests ElementwiseOps::asin against fixed expected values |
| `test_elementwise/test_asinh.cpp` | Tests ElementwiseOps::asinh against fixed expected values |
| `test_elementwise/test_atan.cpp` | Tests ElementwiseOps::atan against fixed expected values |
| `test_elementwise/test_atanh.cpp` | Tests ElementwiseOps::atanh against fixed expected values |
| `test_elementwise/test_ceil.cpp` | Tests ElementwiseOps::ceil against fixed expected values |
| `test_elementwise/test_clamp.cpp` | Tests ElementwiseOps::clamp against fixed expected values |
| `test_elementwise/test_convert.cpp` | Tests ElementwiseOps::convert between the real dtypes, its refusals and its ordering |
| `test_elementwise/test_cos.cpp` | Tests ElementwiseOps::cos against fixed expected values |
| `test_elementwise/test_cosh.cpp` | Tests ElementwiseOps::cosh against fixed expected values |
| `test_elementwise/test_div.cpp` | Tests ElementwiseOps::div against fixed expected values |
| `test_elementwise/test_exp.cpp` | Tests ElementwiseOps::exp against fixed expected values |
| `test_elementwise/test_floor.cpp` | Tests ElementwiseOps::floor against fixed expected values |
| `test_elementwise/test_fmod.cpp` | Tests ElementwiseOps::fmod against fixed expected values |
| `test_elementwise/test_integer_arithmetic.cpp` | Tests integer elementwise arithmetic, wrap-around included, and double scalars |
| `test_elementwise/test_log.cpp` | Tests ElementwiseOps::log against fixed expected values |
| `test_elementwise/test_max.cpp` | Tests ElementwiseOps::max against fixed expected values |
| `test_elementwise/test_min.cpp` | Tests ElementwiseOps::min against fixed expected values |
| `test_elementwise/test_mul.cpp` | Tests ElementwiseOps::mul against fixed expected values |
| `test_elementwise/test_neg.cpp` | Tests ElementwiseOps::neg against fixed expected values |
| `test_elementwise/test_ordered_sum.cpp` | Tests ElementwiseOps::ordered_sum against a serial sum of the same order and type, bit for bit, and its refusals |
| `test_elementwise/test_pow.cpp` | Tests ElementwiseOps::pow against fixed expected values |
| `test_elementwise/test_reciprocal.cpp` | Tests ElementwiseOps::reciprocal against fixed expected values |
| `test_elementwise/test_remainder.cpp` | Tests ElementwiseOps::remainder against fixed expected values |
| `test_elementwise/test_round.cpp` | Tests ElementwiseOps::round against fixed expected values |
| `test_elementwise/test_sin.cpp` | Tests ElementwiseOps::sin against fixed expected values |
| `test_elementwise/test_sinh.cpp` | Tests ElementwiseOps::sinh against fixed expected values |
| `test_elementwise/test_sqrt.cpp` | Tests ElementwiseOps::sqrt against fixed expected values |
| `test_elementwise/test_square.cpp` | Tests ElementwiseOps::square against fixed expected values |
| `test_elementwise/test_sub.cpp` | Tests ElementwiseOps::sub against fixed expected values |
| `test_elementwise/test_tan.cpp` | Tests ElementwiseOps::tan against fixed expected values |
| `test_wrapping_math.cpp` | Tests that integer arithmetic wraps modulo its width and floating arithmetic is plain |

## test_elementwise/test_convert.cpp

The values are chosen so the assertions cannot pass without the conversion happening. A round
trip of small integers through fp16 is a true statement about almost any implementation,
including a `memcpy` that got the widths right by luck, so it appears only as a labelled sanity
case. Everything else asserts on a value that changes: an fp32 number fp16 cannot represent
must come back as the correctly rounded fp16 neighbour (not merely "close"), one past fp16's
range must come back as an infinity, and one below it must come back as zero.

The expectations are exact, and they are worked out by hand from IEEE-754 binary16. A
differential oracle against `static_cast` on the host would agree with a device kernel that
made the same mistake. The ordering cases assert on `dependency_count`; see "Ordering is asserted on
in-degree".

The file is laid out in five groups: the cases the conversion has to change a value to pass,
the sanity case, what the operation refuses and whether the refusal says enough to act on, what
it declares to the dependency tracker, and the offset case.

`Fp32ToFp16RoundsToTheNearestRepresentableValue`: the assertion is on the exact correctly
rounded neighbour, and a second assertion says the result is not the input. An implementation
that widened, reinterpreted or otherwise carried the fp32 value through unchanged fails here.

`Fp32ToFp16BreaksTiesToEven`: the rule is round-to-nearest-even, not round-away and not
truncate. `1 + 2^-11` sits exactly halfway between the binary16 neighbours 1.0 (mantissa even)
and `1 + 2^-10` (mantissa odd), so only ties-to-even gives 1.0. The second value is just past
halfway and must go the other way, which stops "always round down" from passing the first.

`Fp32PastTheHalfRangeBecomesInfinity` is the behaviour a loss-scaling backoff is built on: a
gradient scaled past fp16's range does not wrap, saturate or clamp. It becomes an infinity,
which `count_nonfinite` can then find.

`Fp32BelowTheHalfRangeUnderflowsToZeroButSubnormalsSurvive`: 1e-8 is below half of the
smallest subnormal and must flush to zero. 1e-7 is not, and must land on a subnormal and not be
flushed with it.

`NonFiniteValuesSurviveTheConversion`: this pairing makes `ReductionOps::count_nonfinite`
meaningful on a converted tensor. If a cast quietly turned a NaN into a number, screening the
fp16 copy would say the fp32 original was clean.

`ARoundTripThroughHalfKeepsOnlyWhatHalfCouldHold` is the complement of the sanity case: it says
what the round trip costs when the value is one fp16 cannot hold.

`EveryRealDTypePairConvertsExactlyRepresentableValues` is the sanity case. Exactly representable
values survive every conversion, which is necessary and nowhere near sufficient: it would pass
for an implementation that had the element widths right and the rounding wrong. Its whole job
is to cover all nine arms of the dtype cross product in one place.

`ConvertingADTypeToItselfIsAPlainCopy`: this is the decision documented on
`ElementwiseOps::convert`. A mixed-precision policy holds its compute dtype as runtime data,
and when it equals the master dtype the operation has to degenerate into a copy and not throw.
The test writes 2^40 and not a decimal like 1e30, because the expectation is an exact equality
and a decimal that fp32 cannot hold would be rounded by `write_exact` before `convert` ever saw
it, which would make this a test of the test.

`ANonContiguousOperandIsRefused`: a slice along an inner axis keeps its parent's strides, so
its elements are not where a flat walk would look for them.

`OverlappingOperandsAreRefused`: overlapping operands are a race between work-items, not an
in-place update, because each item reads one element and writes another of a different width.

`AConvertIsOrderedBehindTheOpThatWroteItsSource`: the source is a view at a non-zero offset,
for the reason given under "Views at a non-zero offset". Verified by mutation: rewriting
`src.region()` as the bare pointer makes the assertion report 0.

`ConvertsOverDisjointViewsOfOneArenaDoNotSerialise` is the control: two converts over disjoint
views of one arena must not serialise. The two ordering tests catch an under-declared operand
(no edge where one is needed); this one catches an over-declared operand (an edge where none
is). Verified by mutation as well: declaring `src.storage->allocation_base()` in place of
`src.region()` makes the assertion report 1, while both ordering tests stay green.
