# spatial_kernels

The reference and SYCL kernels of the window, pooling, channel and broadcast operations. The module
is internal: its headers are not installed. The index and tail math of each kernel sits in the
`*_index.hpp` headers, which include no SYCL header and run on the host and in a kernel; the reference
and the SYCL kernel call the same copy. `modules/kernels/README.md` holds the notes of each
kernel.

It depends on `core` and `support`. Its headers are included as
`<SushiBLAS/spatial_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `bilinear_taps.hpp` | The taps, the blend and the axis weights that both bilinear kernels share |
| `im2col_index.hpp` | The index math of one im2col matrix element, shared by the host and the device |
| `pixel_index.hpp` | The flat-index and stride-offset math of an NHWC tensor, shared by the host and the device |
| `pool_max_index.hpp` | The step that decides whether a tap replaces the running max-pool maximum |
| `reference/broadcast.hpp` | The reference kernels of the three broadcast copies |
| `reference/channels.hpp` | The reference kernel that copies a run of channels |
| `reference/col2im.hpp` | The reference kernel of col2im |
| `reference/im2col.hpp` | The reference kernel of im2col |
| `reference/pool_avg.hpp` | The reference kernels of average pooling and its gradient |
| `reference/pool_max.hpp` | The reference kernels of max pooling and its gradient |
| `reference/upsample.hpp` | The reference kernels of nearest-neighbour upsampling and its gradient |
| `reference/upsample_bilinear.hpp` | The reference kernels of bilinear upsampling and its gradient |
| `reference/window_dot.hpp` | The reference kernels of the per-channel weighted window sum and its input gradient |
| `reference/window_dot_weights.hpp` | The reference kernel of the weight gradient of the per-channel weighted window sum |
| `spatial.hpp` | Includes every spatial kernel, reference and SYCL |
| `sycl/broadcast.hpp` | The SYCL kernels of the three broadcast copies |
| `sycl/channels.hpp` | The SYCL kernel that copies a run of channels |
| `sycl/col2im.hpp` | The SYCL kernel of col2im |
| `sycl/im2col.hpp` | The SYCL kernel of im2col |
| `sycl/pool_avg.hpp` | The SYCL kernels of average pooling and its gradient |
| `sycl/pool_max.hpp` | The SYCL kernels of max pooling and its gradient |
| `sycl/upsample.hpp` | The SYCL kernels of nearest-neighbour upsampling and its gradient |
| `sycl/upsample_bilinear.hpp` | The SYCL kernels of bilinear upsampling and its gradient |
| `sycl/window_dot.hpp` | The SYCL kernels of the per-channel weighted window sum and its input gradient |
| `sycl/window_dot_weights.hpp` | The SYCL kernel of the weight gradient of the per-channel weighted window sum |
| `transpose_index.hpp` | The tile geometry and index math of the transposing layout copy |
| `upsample_index.hpp` | The index math of nearest-neighbour upsampling |
| `window_index.hpp` | The sliding-window index math: gather and scatter coordinates, window and patch coordinates, column and tap offsets |

## Bilinear upsampling

`upsample_bilinear` interpolates each output pixel from the four input pixels around its centre. The
centre of output index `o` maps to `s = (o + 0.5) / scale - 0.5`, which `bilinear_taps.hpp` holds as
the integer fraction `(2 * o + 1 - scale) / (2 * scale)`: the lower source index is its floor, the
weight is the remainder over the denominator, and both source indices clamp to the extent. The value
is `(a * (1 - t_w) + b * t_w) * (1 - t_h) + (c * (1 - t_w) + d * t_w) * t_h`, evaluated in that order
in `Accumulator<T>` and rounded once on the store. A blend with a zero weight returns its first
operand without reading the second, so a scale of 1 on both axes copies the input's bit patterns,
an infinity or a negative zero among them. The reference and the SYCL kernel call the same taps and
the same blend.

`upsample_bilinear_backward` makes one pass per input pixel. It visits the output rows from
`(h - 1) * scale_h` to `(h + 2) * scale_h - 1`, clamped to the output, and the columns likewise, in
ascending `(h_o, w_o)` order, and adds each gradient times the product of the two axis weights the
forward gave this pixel. An axis weight is the sum of both taps when clamping makes them the same
pixel, and an output that gives the pixel weight zero on either axis is skipped, so a NaN gradient
elsewhere does not reach it. The reference sums in `ReferenceSum` with the weights in double; the
SYCL kernel sums in `Accumulator<T>`. The SYCL kernels run one work-item per output element and per
input element, each gathering its own window in ascending order; nothing scatters, no atomic is
used and `sycl/upsample_bilinear.hpp` contains no `sycl::atomic_ref`.
