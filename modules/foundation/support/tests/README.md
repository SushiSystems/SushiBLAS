# Tests of support

The unit tests of the `support` module, built into `sushiblas_support_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_complex_arithmetic.cpp` | Tests the conjugation, product and quotient bricks on real and complex values |
| `test_launch_shapes.cpp` | Tests the shape-taking launch primitives and the local tile fit check |
| `test_reduce_plans.cpp` | Tests the plan-taking forms of Reduce::reduce and Reduce::reduce_indexed |
| `test_reduce_segments.cpp` | Tests Reduce::reduce_segments in both launch shapes and the axis fold that runs on it |
| `test_reference_accumulator.cpp` | Tests the type and the running sum a reference kernel accumulates in |
| `test_running_solution.cpp` | Tests RunningSolution: where it keeps a solved element and what it rounds |
| `test_serial_loops.cpp` | Tests the serial host loops and folds against the launch shapes and accumulators they mirror |
| `test_sparse_assembly.cpp` | Tests assemble_csr and require_distinct_positions on host triplets and positions |
| `test_spatial_geometry.cpp` | Tests output_extent, the window span and the default ConvGeometry without a device |

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
