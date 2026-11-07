# host_device

The fixture that hands a kernel test the host CPU queue its SYCL kernel runs on, and skips the test
when the machine has no such device, and the shared USM array both sides of a
comparison read. The module is internal and test-only: its headers are not installed, and it is
added only when `SB_BUILD_TESTS` is on.

It depends on `test_runner`, SYCL and GoogleTest. The edge to `test_runner` is test-only, as both
modules are: the test environment makes the host CPU queue once per program, before the first test,
and `HostDeviceTest` hands it out, so no test makes a queue. Its headers are included as
`<SushiBLAS/host_device/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_device_test.hpp` | `HostDeviceTest`, the fixture that hands out `TestEnvironment::cpu_queue()` and skips without a CPU device |
| `shared_array.hpp` | `SharedArray`, one aligned array freed on destruction; shared USM unless given an allocator |
| `shared_allocator.hpp` | `SharedAllocator`, the port through which `SharedArray` gets and frees its memory |
| `usm_shared_allocator.hpp` | `UsmSharedAllocator`, the default adapter: shared USM on one queue |
| `host_shared_allocator.hpp` | `HostSharedAllocator`, the in-memory adapter: aligned host memory, counted, poisoned, able to fail |

## Why the host device

A kernel test compares the reference kernel, which runs on the host, with the SYCL kernel. On the
CPU device both run on one processor with one floating-point unit, so a kernel that promises the
reference's bits can be held to exact equality. The tests carry the `Device_` suite prefix and
the `device` label: without a CPU device they are reported as skipped.

## Why an allocator port

A `SharedArray` test about values, size or failure needs memory, not a device. The array takes its memory
from a `SharedAllocator`: the queue constructors use `UsmSharedAllocator`, so a kernel test is unchanged,
and the allocator constructors take any adapter, so a `Small_` test uses `HostSharedAllocator` with no
runtime and no queue. The same adapters are held to one contract in `tests/support/shared_allocator_contract.hpp`.
