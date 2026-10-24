# signal_kernels

The reference and SYCL kernels of the operations on rows of samples. The module is internal: its
headers are not installed.

It depends on `arithmetic`, `launch` and `reference`. Its headers are included as
`<SushiBLAS/signal_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `reference/convolve_1d.hpp` | The reference kernel of the valid-mode one-dimensional convolution |
| `signal.hpp` | Includes every signal kernel, reference and SYCL |
| `sycl/convolve_1d.hpp` | The SYCL kernel of the valid-mode one-dimensional convolution |
