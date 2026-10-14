# Tests of geometry

The unit tests of the `geometry` module, one file per header. The source comments state what a
test checks; a section below holds why a test is written the way it is. `tests/unit/README.md`
holds the reasoning the tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_vector3.cpp` | Tests Vector3T, its operators, dot, cross, length and normalize against hand values |
| `test_quaternion.cpp` | Tests QuaternionT, its product, conjugate, norm and vector rotation against hand values |
| `test_matrix4.cpp` | Tests Matrix4T's default, its column-major layout and its product against hand values |
| `test_interpolation.cpp` | Tests lerp, nlerp and slerp against hand values, with the short-arc and parallel cases |
| `test_transform.cpp` | Tests the functions that build a 4x4 transform or take one apart against hand values |
| `test_spatial_geometry.cpp` | Tests output_extent, the window span and the default ConvGeometry without a device |

Every suite is typed over `float` and `double` and runs on the host; no test submits to a device.

## Expected values

An expected value is a number worked out by hand and written in the test, never the output of the
function under test or of its inverse. The rotations are chosen so that the numbers are short: a
quarter turn about an axis, whose quaternion holds `sqrt(1/2)`; a third of a turn about
`(1, 1, 1)`, whose quaternion is `(0.5, 0.5, 0.5, 0.5)` and which cycles the axes; half turns
about `(0.8, 0, 0.6)` and its permutations, whose matrices hold 0.28 and 0.96.

A comparison allows 16 epsilons of the element type as an absolute distance, the constant
`K_<FILE>_TOLERANCE_EPSILONS` at the top of each file. The values compared against that bound
are of the order of 1; the larger ones, such as a cross product of 27, are sums of small integers
and exact in both types. The checks that use `EXPECT_EQ` are the ones where the function promises
an exact value: a default-constructed object, a degenerate `normalize`, the identity that
`affine_inverse` returns for a singular matrix.

## test_vector3.cpp and test_quaternion.cpp

`NormalizeReturnsAZeroLengthVectorUnchanged` and `NormalizeReturnsTheIdentityForAZeroLength` each
run two inputs: all zeros, and the smallest normal value of `T` in one component. The square of
the second underflows to zero, so its length is zero although the input is not. The two functions
answer the same case differently, the vector unchanged and the quaternion as the identity, and
each test holds its function to its own answer.

## test_interpolation.cpp

`NlerpFlipsTheSecondRotationToTheShortArc` and `SlerpFlipsTheSecondRotationToTheShortArc` pass the
negation of a quarter turn about z. The expected value is the one the un-negated rotation gives;
without the flip the result is a different rotation, which the comment in each test names.

`SlerpOfEqualRotationsIsFiniteAndReturnsThem` is the case the small-angle branch exists for: the
angle is exactly zero and the sine quotient of the main branch would be `0 / 0`.
`SlerpOfNearlyParallelRotationsIsTheHalfwayRotation` takes the same branch with a rotation of
about 0.0012 radians, where `dot(a, b)` is 0.99999982, above the threshold of `1 - 1e-6`.

## test_transform.cpp

`quaternion_from_matrix` has four branches and four tests. The trace branch runs a quarter turn
about each axis, which reads each of the three off-diagonal differences. The x, y and z branches
run half turns, whose trace is -1; each reads the two off-diagonal sums of its formula. The third
of a turn about `(1, 1, 1)` has trace 0 and three equal diagonal elements, so both comparisons
fail and it falls to the z branch.

`AffineInverseInvertsAShearedMatrix` uses a 3x3 part with nine different cofactors and
determinant 25, so every adjugate term of the function is compared with a hand value. The two
products with the original matrix are compared with the identity as well.

`DecomposeTransformRecoversWhatComposeTransformTook` compares the three outputs with the three
inputs written in the test. The reflection and the zero-scale cases build their matrix by hand,
because `compose_transform` followed by `decompose_transform` does not return a negative y or z
scale and the tests state what comes back and not what went in.

## test_spatial_geometry.cpp

The file is device-free by construction: `spatial_geometry.hpp` includes `<cstdint>` and nothing else,
so this file links no SYCL and needs no Engine. That is the property the header exists to have.
SushiAI's host-lane shape inference shares this arithmetic and must not inherit the device
stack to get it.

`AsymmetricPaddingIsWhatHoldsYoloTinysLastPool`: YOLOv3-tiny's 2x2 stride-1 max-pool is padded
on the bottom and right only, so 13x13 stays 13x13. This is the case CV-5 exists to make
ordinary, and it is the reason `pad_bottom` is a host-side number and not a ninth attribute
slot. The second assertion pads the other side, because symmetric and asymmetric padding of the
same total are not interchangeable.

`AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne`: the formula's numerator
goes negative here. Integer division of a negative numerator is implementation-flavoured and
would produce a garbage extent that later multiplies into an allocation size, so the case is
answered explicitly.

`TheDefaultGeometryIsTheIdentityWindow`: a default-constructed geometry is 1x1, stride 1, no
dilation and no padding, so it maps every extent to itself. Anything that reads a half-built
geometry therefore gets a shape it can recognise as wrong, not a plausible one.
