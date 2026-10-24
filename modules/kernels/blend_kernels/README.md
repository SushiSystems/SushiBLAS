# blend_kernels

The reference and SYCL kernels of the operations that blend transforms and apply them to points.
The module is internal: its headers are not installed.

It depends on `arithmetic`, `launch` and `reference`. Its headers are included as
`<SushiBLAS/blend_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `blend.hpp` | Includes every blend kernel, reference and SYCL |
| `reference/blend_transforms.hpp` | The reference kernel of the weighted blend of transforms applied to points |
| `sycl/blend_transforms.hpp` | The SYCL kernel of the weighted blend of transforms applied to points |
