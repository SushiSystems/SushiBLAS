# Tests of spatial

The unit tests of the `spatial` module, built into `sushiblas_spatial_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_spatial_broadcast.cpp` | Tests SpatialOps::channel_affine, broadcast_spatial and layout_convert |
| `test_spatial_channels.cpp` | Tests SpatialOps::concat_channels and slice_channels |
| `test_spatial_col2im.cpp` | Tests SpatialOps::col2im against overlap counts and the reference kernel |
| `test_spatial_half.cpp` | Tests that the spatial kernels sum HALF terms in FLOAT32 and agree across both modes |
| `test_spatial_im2col.cpp` | Tests SpatialOps::im2col's gather order, padding, dilation and refusals |
| `test_spatial_pool_avg.cpp` | Tests SpatialOps::pool_avg and pool_avg_backward |
| `test_spatial_pool_max.cpp` | Tests SpatialOps::pool_max and pool_max_backward |
| `test_spatial_upsample.cpp` | Tests SpatialOps::upsample_nearest and upsample_nearest_backward |
| `test_spatial_window_dot.cpp` | Tests the per-channel weighted window sum and its two gradients in both forced modes; a `HALF` result below 2^-13 may sit one unit in the last place under the device's, because the host conversion truncates there |

## test_spatial_broadcast.cpp

`nchw_view` builds its view through `Tensor`'s stride-explicit constructor, which takes the
whole fixed-width stride array, so nothing in the helper reaches past the library's own seams.
`create_tensor` only ever lays a shape out contiguously, so the allocation comes from it and
the layout comes from the helper. The logical shape stays `[N, H, W, C]` because that is what
`layout_convert` compares: a conversion permutes strides, not shapes.

`BroadcastSpatialAtOneOverHwIsTheMeanBackward` checks the property `SPATIAL_MEAN_BACKWARD`
needs: each (n, c) plane sums back to the 1.0 that global average pooling's forward divided
out. The sum is not exact and is not expected to be. 1/20 has no finite binary expansion, so
twenty copies of `float(1/20)` add up to 1.0 only to within the rounding each addition carries.
A few ulps of 1.0 is roughly 1e-7; the tolerance of 1e-6 leaves room without hiding a real
error, which would be off by a factor of H, of W or of N.

## test_spatial_channels.cpp

`ConcatPlacesEachSourceAtItsOffset` uses two pixels, so a source that ignored its column stride
would still pass the first pixel and fail the second.

`TheyAreEachOthersBackward` is the property the autograd rule stands on: the gradient of a
concat is a slice at the same offset and of the same width. It is written as one round trip
through the offset the second source occupies, so an off-by-one in either direction shows up
and does not cancel.

`AStridedSourceIsAddressedThroughItsStrides` builds its source through `Tensor`'s
stride-explicit constructor, the same public seam `test_tensor_view.cpp` exercises: a
`[1,2,2,2]` view over a `[1,2,2,4]` parent whose channel stride is 2, so it sees the parent's
even channels only. A kernel that assumed a unit channel stride would copy channels 0 and 1
instead of 0 and 2.

## test_spatial_col2im.cpp

`EachPixelReceivesOncePerWindowThatReadIt`: col2im of an all-ones column matrix is the overlap
count of each pixel, the number of windows that read it. The test computes that answer
independently, and it is the strongest cheap check that the gather visits exactly the right
set. A rule that drops contributions or double-counts fails it, where a round trip against
im2col would not necessarily. With a 3x3 stride-1 window padded by 1, a pixel is read by every
window whose origin is within one step of it and inside the image: 3x3 in the interior, fewer
at the edges. The count is made by brute force and not by a closed form, so the expectation
cannot share a mistake with the kernel.

`StrideAndDilationSelectTheRightContributors`: a stride-2 window means every pixel is read by
exactly one window. Getting the divisibility test wrong, by dropping the `% stride != 0` guard,
turns this into a count of 4 and is otherwise invisible.

`PixelsNoWindowReadsAreWrittenZeroRatherThanLeftAlone`: a 2x2 stride-2 window over a 5x5 image
covers rows 0..3 and columns 0..3, and the last row and column are read by nothing. The gather
form writes every pixel exactly once, so those must come back zero, which is what lets a caller
pass an uninitialised destination. The destination is seeded with a sentinel so "left alone" is
visible.

`TheSYCLKernelAgreesWithTheReferenceKernel` uses a ramp for the column entries and not all
ones, so a pixel's total depends on which entries it gathered and not only on how many. The
all-ones case cannot tell a wrong set of the right size from the right one.

## test_spatial_im2col.cpp

`AUnitWindowIsTheIdentityGather`: with a 1x1 window, stride 1 and no padding, the column matrix
is the input reshaped, element for element and in the same order. If this fails, the row and
element index arithmetic disagree with each other and nothing further is worth reading.

`EachRowIsOneWindowWithChannelInnermost`: a 2x2 stride-1 window over a 1x3x3x2 input gives 4
windows of 8 entries each. The rows are written out by hand and not derived, so the test is a
specification and not a second implementation.

`OutOfBoundsTapsReadZeroRatherThanNeighbouringMemory`: padding is a parameter, not a separate
zero-padded copy, which saves a full-tensor copy forward and a crop backward. This test is the
property that makes that safe.

`AsymmetricPaddingNeedsTheExplicitExtentOverload`: a 2x2 stride-1 window over a 3x3 image
padded on the bottom and right only gives 3x3 windows. The two-operand overload cannot express
this, because it has no `pad_right` to read, and the test checks that it refuses and does not
silently produce a differently shaped answer.

`TheSYCLKernelAgreesWithTheReferenceKernel` compares bit-exact, not near. im2col only ever
copies an element or writes a zero, so the two kernels must produce the same bits or one of
them is addressing memory differently from the other. See "Comparing two kernels under the two
forced modes" for why the comparison is made in the test.

## test_spatial_pool_avg.cpp

`TheAverageIsOverTheWholeWindowIncludingPaddedTaps` is the pin for decision D-A4. Without a
padded case the convention is invisible: an unpadded pool divides by the area and by the
in-bounds count at once, so every other forward test in the file passes under either rule. The
top-left window's only in-bounds tap is the element 4. Divided by the window area that is 1;
divided by the in-bounds count it would be 4. That difference is the whole of `count_include_pad`.

`OverlappingWindowsSumTheirShares`: with a 2x2 stride-1 window over a 3x3 image, the centre
pixel is covered by all four windows and the corners by one each. A gather that stopped at its
first covering window would give the centre 0.25.

`TheBackwardOfAConstantForwardIsUniform`: a stride-1 pool padded so that every pixel is covered
by exactly R*S windows. Each contributes 1/(R*S), so every pixel must come back exactly 1.0,
the closed-form check that the divisor and the window set agree with each other. A divisor
computed per window from the in-bounds count would make the border pixels larger.

## test_spatial_pool_max.cpp

`four_by_four` returns the image every hand-checked case pools. It is chosen so the 2x2
stride-2 windows' maxima do not all sit in the same corner of the window. A kernel that always
picked, say, the last tap in scan order would pass a window whose maximum happened to be there.

```
1  3 | 2  0
4  2 | 1  5
     +
0  7 | 6  2
8  1 | 3  9
```

`PaddingNeverWinsAWindow` was written and run before any kernel existed. A zero-initialised
running maximum is the single likeliest defect in this operation, and it is invisible in every
test whose data is non-negative. Every element is negative, and the window at (0,0) sees
exactly one of them; its other three taps fall outside the image. Those taps are the absence of
an element, not zeros. A max initialised to `T(0)` would report 0, a plausible number that no
shape or dtype check can catch. The elements are ordered so each window's maximum is a
different element. The largest sits at the bottom right, so a window that saw only the top-left
corner cannot accidentally agree with one that saw everything.

`TheIndexNamesTheElementTheMaximumCameFrom`: the index is the logical flat position
`((n*H + h)*W + w)*C + c`, not an offset into this particular view. A strided offset would stop
being comparable the moment the gradient arrived through an alias.

`ATieGoesToTheFirstElementInScanOrder`: deterministic tie-breaking is what makes the recorded
index single-valued, and therefore what stops the backward's equality test from matching two
taps of one window and double-counting the gradient.

`TheBackwardRoutesEachGradientToItsArgmaxAndNowhereElse`: every non-argmax pixel must be
exactly zero, not merely small. The gather writes each pixel once, so an unrouted pixel is a
written zero and never a leftover.

`OverlappingWindowsAccumulateAtASharedArgmax` is YOLOv3-tiny's 2x2 stride-1 pool. One dominant
pixel is selected by all four windows, so its gradient is the sum of all four. A scatter would
have needed an atomic add for this case, and a gather that stopped at its first match would get
it wrong.

`TheSYCLKernelAgreesWithTheReferenceKernel` compares the indices as well as the values. Two
kernels can agree on every maximum and still disagree on where it came from, which is what a
tie broken the other way is, and the gradient follows the index, not the value.

## test_spatial_upsample.cpp

`TheBackwardSumsTheBlockThatReadEachInput` sums the expectation by walking the block index by
index, with no closed form the kernel could be wrong in the same way.
