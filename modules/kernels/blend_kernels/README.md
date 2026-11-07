# blend_kernels

The reference and SYCL kernels of the operations that blend transforms and apply them to points.
The module is internal: its headers are not installed. The index math of the blend sits in
`blend_index.hpp`, which includes no SYCL header; the reference and the SYCL kernel call the same copy.

It depends on `arithmetic`, `launch` and `reference`. Its headers are included as
`<SushiBLAS/blend_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `blend_index.hpp` | The index math and the homogeneous-point step of the blend, shared by the host and the device |
| `blend.hpp` | Includes every blend kernel, reference and SYCL |
| `reference/blend_transforms.hpp` | The reference kernel of the weighted blend of transforms applied to points |
| `sycl/blend_transforms.hpp` | The SYCL kernel of the weighted blend of transforms applied to points |
