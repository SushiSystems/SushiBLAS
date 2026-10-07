# Spatial headers

`include/SushiBLAS/engine/spatial` declares the span builders (`window.hpp`) that `SpatialOps` in
`engine/spatial.hpp` runs on. The window geometry they use is in
[../../support/README.md](../../support/README.md). The reasoning for
`engine/spatial.hpp` itself is in [../README.md](../README.md).

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `window.hpp`

The single place a layout is interpreted. Everything downstream of `window_span_for` sees element
counts, so adding NCHW later is a second stride assignment here and nothing else anywhere.

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

At `SB_THROW_IF(out_h > 0 && (out_h - 1) * g.stride_h - g.pad_top >= span.in_h,`: A window whose
origin is at or past the last input row covers only padding. Legal arithmetic, never a legal
request, and the shape it produces is plausible enough to survive every other check here.

### `window_span_for` from two operands

The form every operation with an image output takes: pooling, upsampling, the channel copies, the
affine. `H_o` and `W_o` come from `out`, which the caller sized with `output_extent` while the
trailing padding it needs was still in scope.

### `require_valid_geometry`

`window_span_for` calls it first, so every spatial operation that takes a `ConvGeometry` checks it
at record time. Kernel sizes, strides and dilations must be at least 1, and both paddings at least
0. A violation throws `std::invalid_argument` that names the field. The rule that both output
extents are at least 1 belongs to `window_span_for` with stated extents, which checks it after this
call.

### `pooling_span_for`

The pooling operations (`pool_max`, `pool_avg` and their backwards) call this instead of the
two-operand `window_span_for`. On top of its checks it refuses a geometry that leaves any window
entirely in the padding, through `require_live_windows` on each axis. Only windows whose origin is
negative can miss the input, so the check walks those alone. A tap is live when it lands in
`[0, in)`. The padding must therefore be smaller than the covered extent `dilation * (kernel - 1) + 1`,
and with dilation a tap must also fall on a real row rather than step over the whole input.
`im2col` and `col2im` keep accepting such windows: their patches are zeros, which is a defined
result.
