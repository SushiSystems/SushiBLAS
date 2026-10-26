# signal_kernels

The reference and SYCL kernels of the operations on rows of samples. The module is internal: its
headers are not installed.

It depends on `arithmetic`, `launch` and `reference`. Its headers are included as
`<SushiBLAS/signal_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `fft_plan.hpp` | `FftPlan`, the tables and scratch layout one batched FFT shape cooks once, and the host line transform |
| `reference/convolve_1d.hpp` | The reference kernel of the valid-mode one-dimensional convolution |
| `reference/fft.hpp` | The reference batched forward and inverse FFT |
| `signal.hpp` | Includes every signal kernel, reference and SYCL |
| `sycl/convolve_1d.hpp` | The SYCL kernel of the valid-mode one-dimensional convolution |
| `sycl/fft.hpp` | The SYCL batched forward and inverse FFT, radix-2 with Bluestein's transform for other lengths |

The FFT kernels follow the bitwise agreement rules in
[the linalg_kernels README](../linalg_kernels/README.md#bitwise-agreement).
