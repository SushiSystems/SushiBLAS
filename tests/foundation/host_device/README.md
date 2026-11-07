# Tests of host_device

The tests of the `host_device` module, built into `sushiblas_host_device_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `compile_host_device_test.cpp` | Compile check, never run: `host_device_test.hpp` stands alone |
| `test_host_queue.cpp` | Tests that host_queue returns one in-order queue on a CPU device, or none; an empty device list is `Small_` |
| `compile_shared_allocator.cpp` | Compile check, never run: `SharedAllocator` is an abstract port, not copied or moved |
| `test_host_shared_allocator.cpp` | Tests that HostSharedAllocator keeps the SharedAllocator contract and counts, poisons and refuses on request (`Small_`) |
| `test_usm_shared_allocator.cpp` | Tests that UsmSharedAllocator keeps the SharedAllocator contract on the queue of the shared test environment (`Medium_`) |
| `test_shared_array.cpp` | Tests that SharedArray fills, copies and aligns its elements through its allocator (`Small_`) and that a kernel reaches them (`Medium_`) |

## The allocator contract

`tests/support/shared_allocator_contract.hpp` holds the checks every `SharedAllocator` passes. Each adapter
runs them in one typed suite named `SharedAllocatorContract`: `Small_` for the host adapter, `Medium_` for the
USM adapter, whose queue comes from the test program's shared environment.
