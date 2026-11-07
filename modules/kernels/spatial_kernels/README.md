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
| `sycl/window_dot.hpp` | The SYCL kernels of the per-channel weighted window sum and its input gradient |
| `sycl/window_dot_weights.hpp` | The SYCL kernel of the weight gradient of the per-channel weighted window sum |
| `transpose_index.hpp` | The tile geometry and index math of the transposing layout copy |
| `upsample_index.hpp` | The index math of nearest-neighbour upsampling |
| `window_index.hpp` | The sliding-window index math: gather and scatter coordinates, window and patch coordinates, column and tap offsets |
