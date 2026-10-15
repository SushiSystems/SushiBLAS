# host_device

The SYCL queue on the host CPU device that a kernel test runs its SYCL kernel on, the fixture that
skips a test when the machine has no such device, and the shared USM array both sides of a
comparison read. The module is internal and test-only: its headers are not installed, and it is
added only when `SB_BUILD_TESTS` is on.

It depends on SYCL and GoogleTest only. Its headers are included as
`<SushiBLAS/host_device/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_queue.hpp` | `host_queue()`, the in-order queue on the first CPU device, or none |
| `host_device_test.hpp` | `HostDeviceTest`, the fixture that skips without a CPU device |
| `shared_array.hpp` | `SharedArray`, one aligned shared USM array freed on destruction |

## Why the host device

A kernel test compares the reference kernel, which runs on the host, with the SYCL kernel. On the
CPU device both run on one processor with one floating-point unit, so a kernel that promises the
reference's bits can be held to exact equality. The tests carry the `Device_` suite prefix and
the `device` label: without a CPU device they are reported as skipped.
