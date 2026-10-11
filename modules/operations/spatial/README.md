# Spatial headers

`window.hpp` declares the span builders
that `SpatialOps` in `spatial.hpp` runs on: `require_valid_geometry`,
`require_live_windows`, the two `window_span_for` forms, `pooling_span_for`, `column_span_for` and
`tap_span_for`. Every check in it throws `std::invalid_argument`. The window geometry they use is in
[the support module](../../foundation/support/README.md). The notes of `spatial.hpp` itself are
the last section of this page.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `window.hpp`

The place a layout is interpreted for the windowed operations. Everything downstream of
`window_span_for` sees element counts, so adding NCHW later is a second stride assignment here and
in the spans `modules/operations/spatial/source/broadcast.cpp`, `channels.cpp` and `upsample_nearest.cpp` build
by hand.

### `window_span_for` with stated extents

The form `im2col` and `col2im` take: their other operand is a rank-2 column matrix, so there is no
output image to read `H_o` and `W_o` off, and the caller states them.

Reads the strides off `in` rather than deriving them from its shape, so a view with a non-unit
stride, a channel slice, a `storage_offset` into a shared allocation, is addressed correctly without
any caller knowing it was one.

**What it can and cannot check.** The extents are taken, not recomputed, because `output_extent`
needs `pad_bottom` and `pad_right` and `ConvGeometry` deliberately does not carry them (see its
documentation). What it can prove is that every window's *origin* lands inside the padded input: an
output row whose window starts at or past `in_h` reads nothing but padding, which is always a
caller's arithmetic error and never a legitimate request.

At the `require_operand` call on `(out_h - 1) * g.stride_h - g.pad_top >= span.in_h`: A window whose
origin is at or past the last input row covers only padding. Legal arithmetic, never a legal
request, and the shape it produces is plausible enough to survive every other check here.

### `window_span_for` from two operands

The form the window product and its two gradients take, and pooling through `pooling_span_for`.
`H_o` and `W_o` come from `out`, which the caller sized with
`output_extent` while the trailing padding it needs was still in scope.

### `require_valid_geometry`

`window_span_for` calls it first, so every spatial operation that takes a `ConvGeometry` checks it
at record time. Kernel sizes, strides and dilations must be at least 1, and both paddings at least
0. A violation throws `std::invalid_argument` that names the field. The rule that both output
extents are at least 1 belongs to `window_span_for` with stated extents, which checks it after this
call.

### `tap_span_for`

Describes the per-channel tap tensor of `window_dot` and its two gradients. The tensor is rank 3,
`[kernel_h, kernel_w, C]`, and the builder checks each extent against the geometry and the span's
channel count, then returns the three strides as a `Spatial::TapSpan`. It reads the strides off
the tensor, as `window_span_for` does, so a view is addressed correctly. `operand` is the
tensor's name in the caller's signature, `weights` or `dweights`, for the message.

### `pooling_span_for`

The pooling operations (`pool_max`, `pool_avg` and their backwards) call this instead of the
two-operand `window_span_for`. On top of its checks it refuses a geometry that leaves any window
entirely in the padding, through `require_live_windows` on each axis. Only windows whose origin is
negative can miss the input, so the check walks those alone. A tap is live when it lands in
`[0, in)`. The padding must therefore be smaller than the covered extent `dilation * (kernel - 1) + 1`,
and with dilation a tap must also fall on a real row rather than step over the whole input.
`im2col` and `col2im` keep accepting such windows: their patches are zeros, which is a defined
result.

## `spatial.hpp`

The word "convolution" does not appear in this domain and should not be added to it. `im2col` is a
gather, `pool_max` is a windowed reduction, `channel_affine` is a broadcast elementwise. Parameter
ownership, training mode and differentiation belong to the layer above; what belongs here is moving
numbers around a grid.

Every operation in this domain has both a pure-C++ reference kernel and a SYCL one, and the
reference is the specification. The two are held to each other by this domain's unit tests, which
run `ExecutionMode::ForceSYCL` and `ExecutionMode::ForceReference` into two tensors and compare
them. `ExecutionMode` has no mode that runs both.

Every operation returns a `TaskHandle` and checks its operands when the call is recorded. A wrong
rank, dtype or extent throws `std::invalid_argument`. A dtype the operand's device cannot compute
in, or a device-only operand under `ForceReference`, throws `UnsupportedExecution`.

**Every backward here is a gather, never an atomic scatter.** SushiBLAS links
`SushiRuntime::SushiRuntime` PUBLIC and receives `sushiruntime_deterministic_fp` through it, and a
floating-point atomic add has a combination order that
varies between runs, so `col2im` written as "each column entry atomically adds into its pixel" would
silently break the library's bit-reproducibility guarantee. Each output element gathers its own
contributions in a fixed order instead. This is the same argument
`docs/architecture/KERNELS.md` gives for rejecting `sycl::reduction` in the folding reductions,
and `tests/regression/test_spatial_determinism.cpp` is what keeps it true.

### `SpatialOps::im2col` with stated extents

`[N, H, W, C]` to `[N * H_o * W_o, R * S * C]`, with the patch index running `(r, s, c)`: channel
innermost. Out-of-bounds taps are zero; padding is a parameter and never a materialised copy.

**This is the overload to call when the padding is asymmetric.** `Spatial::ConvGeometry` carries
`pad_top` and `pad_left` only (see `Spatial::output_extent` for why), so the trailing padding that
decides `H_o` and `W_o` is not recoverable from the geometry alone. A caller that has both extents,
which SushiAI's shape inference always does, states them here rather than letting the two-operand
form assume symmetry.

### `SpatialOps::im2col` with derived extents

Convenience over the explicit-extent overload, for the symmetric padding every convolution in the
ResNet-18 and YOLOv3-tiny bodies uses: `W_o` follows from `Spatial::output_extent` with `pad_left`
on both sides, and `H_o` from dividing the column matrix's rows per image by it. Asymmetric padding
must use the explicit overload, which is why that one is documented first.

### `SpatialOps::col2im`

`im2col`'s transpose, and its gradient: entry `(row, patch)` of `cols` is added into the pixel that
`im2col` read it from, and entries whose tap fell outside the image are dropped, which is the same
crop a materialised zero-padding would have needed.

Implemented as a gather over output pixels rather than a scatter over column entries, so overlapping
windows need no floating-point atomic add. The opening of the `spatial.hpp` section says why that is
a hard constraint rather than a preference.

Every pixel of `image` is written, including pixels no window read; those receive an explicit zero,
so the destination needs no prior clear.

### `SpatialOps::pool_max`

The running maximum starts at the lowest finite value of the element type, never at zero: padding is
the absence of an element, not a zero-valued candidate, and a zero-initialised maximum silently
turns every all-negative border window into 0. Ties keep the first element in scan order, which is
PyTorch's convention and what makes the recorded index single-valued.

`indices` must be INT64 and shaped like `out`. It holds the **logical** flat position `((n*H + h)*W
+ w)*C + c` of each window's maximum, which stays comparable across differently-strided views of the
same image; `pool_max_backward` needs it to route the gradient without a scatter.

### `SpatialOps::pool_max_backward`

A gather over the windows that could have selected each input pixel, testing the recorded index
rather than scattering through it, so overlapping windows need no floating-point atomic add. A pixel
selected by several windows accumulates all of their gradients in a fixed order.

`g` must be the geometry the forward used; the index buffer alone does not determine the window set.

### `SpatialOps::pool_avg`

The divisor is `kernel_h * kernel_w` and counts padded taps that contributed nothing: PyTorch's
`count_include_pad=True`, which is its default and therefore what a network ported from it expects.
The alternative, dividing by the in-bounds count, gives different numbers at every border pixel of
every padded pool.

No index buffer: unlike `pool_max`, this operation's backward needs nothing the geometry does not
already say.

### `SpatialOps::pool_avg_backward`

Each pixel gathers `dy / area` from every window that covered it. Kept separate from
`pool_max_backward` rather than folded behind a mode flag: the two share a loop shape and nothing
else, since one selects through a recorded index and this one broadcasts a constant.

### `SpatialOps::window_dot`

`out[n, h_o, w_o, c]` is the sum over the window's taps `(r, s)` of `in[n, h_in, w_in, c] *
weights[r, s, c]`: each channel is weighted by its own `[kernel_h, kernel_w]` taps and no channel
reads another. A tap that falls outside the input adds nothing. `weights` is rank 3, `[kernel_h,
kernel_w, C]`, and `out` states `H_o` and `W_o` through its own extents. The domain is real, and
the sum is carried in `Accumulator<T>`. `out` may share no byte with `in` or `weights`, because
the kernel reads them while it writes.

### `SpatialOps::window_dot_backward`

The gradient with respect to the input. Each input element gathers `dy * weights` from every
window that read it, in a fixed order, so overlapping windows need no atomic add. Every element
of `dx` is written. `g` and `weights` must be the ones the forward used.

### `SpatialOps::window_dot_weights_backward`

The gradient with respect to the taps. `dweights[r, s, c]` is the sum, over every output position,
of `dy` times the input element that tap read there. Each tap gathers its own sum, so this is not
a scatter either. `dweights` may share no byte with `dy` or `in`.

### `SpatialOps::upsample_nearest`

`out[n, h_o, w_o, c] = in[n, h_o / scale_h, w_o / scale_w, c]`.

Carries no `Spatial::ConvGeometry:` this operation has two numbers, not eight, and handing it a
geometry it does not read would be six fields a reader has to rule out before trusting the call. It
builds a `Spatial::WindowSpan` by hand, for the extents and the strides.

### `SpatialOps::upsample_nearest_backward`

Each input pixel gathers the fixed `scale_h * scale_w` output elements that read it, in ascending
order, so this needs no atomic add and no index buffer.

### `SpatialOps::concat_channels`

The write half of a channel concatenation: call it once per source, each with the offset its
channels begin at. `src` and `dst` must agree on `N`, `H` and `W`.

**Why not a zero-copy view.** A slice's gradient has to be a materialised operand for the next node,
and a graph whose values are sometimes views and sometimes buffers is a distinction the memory
planner would have to carry everywhere. The copy is paid once; the complexity would be paid forever.

**A known conservatism.** In NHWC a channel range is strided rather than a contiguous byte interval,
so two concatenations into one destination produce overlapping `Tensor::region()`s and
SushiRuntime's dependency tracker serialises them. That is correct and possibly slower than
necessary. Measure before addressing it.

`dst`'s channel count.

### `SpatialOps::slice_channels`

`concat_channels`'s inverse, and its backward. One kernel serves both; two names exist because
`slice_channels(dy, offset, dx)` says what it means and a shared `copy_channels(dy, offset, dx, 0,
count)` does not.

`src`'s channel count.

### `SpatialOps::channel_affine`

`out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. BatchNorm's apply half, and the whole of
BatchNorm in evaluation mode.

### `SpatialOps::broadcast_spatial`

`out[n,h,w,c] = in[n,c] * scale`. Global average pooling's backward, with `scale = 1/(H*W)`.

A **per-`(n, c)`** broadcast, which is neither of the two that already exist: SushiAI's
`Ops::GEMM_BIAS` broadcasts a row vector down every row, and `channel_affine` broadcasts one
value per channel across the whole batch.

### `SpatialOps::layout_convert`

The boundary transpose between NCHW and NHWC. **Its direction is not a flag**: it reads both
tensors' strides and copies by logical `(n, c, h, w)` position, so NCHW-to-NHWC and NHWC-to-NCHW are
the same call with the operands' strides swapped.

This buys interoperability, NCHW data and NCHW checkpoints in, and is not an NCHW implementation.
Two O(N*C*H*W) passes per step are immaterial beside the convolutions they bracket, which is exactly
the trade PyTorch's `channels_last` makes.
