# Spatial headers

`include/SushiBLAS/engine/spatial` declares the window geometry (`geometry.hpp`) and the span
builders (`window.hpp`) that `SpatialOps` in `engine/spatial.hpp` runs on. The reasoning for
`engine/spatial.hpp` itself is in [../README.md](../README.md).

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `geometry.hpp`

Separate from `window.hpp` and free of `Tensor` on purpose. The output-extent formula
(`output_extent`) is the *only* statement of how an input extent, a padding pair, a dilation, a
kernel and a stride combine, and it has two consumers on either side of a repository boundary: this
library's kernels, and SushiAI's host-lane shape inference. That inference is device-free by design,
it is exercised in a unit tier that links no SYCL, so the arithmetic has to be reachable without
dragging the device stack behind it. Two copies of one formula that must agree is the alternative,
and they would agree until the day they did not.

Includes `<cstdint>` and nothing else. Keep it that way.

### `ConvGeometry`

Exactly eight numbers, and the count is load-bearing rather than tidy. SushiAI carries this into
`Graph::Attributes`, whose capacity is bounded by `SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS`:
eight.

**`pad_bottom` and `pad_right` are deliberately absent.** A kernel addresses its input as `h_in =
h_o * stride_h - pad_top + r * dilation_h` and bounds-checks against `[0, H)`. The trailing pads
never enter that expression; they affect only *how many output rows exist*, which is
`output_extent`'s answer and which the host has already written into the output tensor's shape by
the time any of this reaches a device. So asymmetric padding costs zero slots over symmetric
padding, and every geometry is expressible: non-square kernels, anisotropic stride and dilation, and
independent padding on all four sides.

### `output_extent`

`floor((in + pad_before + pad_after - dilation*(kernel - 1) - 1) / stride) + 1`, which is the same
expression PyTorch, Darknet and cuDNN all use.

A window wider than the padded input answers **0** rather than letting the numerator go negative.
That is not defensiveness: the result is multiplied into an allocation size two layers up, and a
negative numerator under integer division would produce a plausible-looking positive extent for some
operand combinations.

### `WindowSpan`

Modelled on `SushiBLAS::Reduce::AxisSpan`, and for the same reason: it carries counts and strides
only, so the layout, the axis order and any non-unit stride a view inherited are resolved once on
the host and the device sees none of them. One kernel per operation, no layout branch in the inner
loop, no second instantiation.

The struct names no layout. Every caller in the convolutional stack builds it from an NHWC tensor
(see the spec's CV-3), but adding NCHW later is filling in a different stride assignment in
`Spatial::window_span_for`, not re-cutting this abstraction.

All strides are element counts and all are non-negative, which `Tensor`'s stride-explicit
constructor already guarantees.

### `ColumnSpan`

Kept out of `WindowSpan` because the column matrix is rank-2 and everything else in this family is
rank-4. Folding two unrelated addressing schemes into one struct would leave four fields meaningless
for every operation but two.

Entry `(row, element)` lives at `data[row * row_stride + element * element_stride]`, where `row`
indexes a window in `(n, h_o, w_o)` order and `element` indexes a patch position in `(r, s, c)`
order: channel innermost, so an NHWC gather copies contiguous runs of `channels` elements.

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
