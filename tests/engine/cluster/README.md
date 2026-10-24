# Tests of cluster

The tests of the `cluster` module, built into `sushiblas_cluster_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake` when `SB_ENABLE_DISTRIBUTED` is
on. The source comments state what a test checks; a section below holds why a test is written
the way it is. [`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_bound_pack.cpp` | Tests that bind_views() checks views against layouts and hands segments back typed |
| `test_cluster_binding.cpp` | Tests that ClusterBinding shares a pack with a cluster and reports what it supports |
| `test_pack_layout.cpp` | Tests that PackLayout groups tensors by type, aligns segments and fingerprints the layout |
| `test_remote_task.cpp` | Tests that a pack of every type round-trips through a worker and a stale worker refuses |
| `test_tensor_pack.cpp` | Tests that TensorPack gathers tensors into its buffer, scatters them back unchanged and hands out tensor views of its entries and segments |

## test_remote_task.cpp

Each test builds a master runtime of its own and a worker runtime joined over an
`InProcessFabric`, because `Cluster::attach()` changes the runtime it attaches to and the shared
test runtime must stay a single machine. The master packs one tensor of each of the eight types,
`record_remote()` sends the pack to the kernel registered under the layout's fingerprint, the
kernel checks the fingerprint the task names, binds both views, copies the input into the output
and transforms each segment, and the master scatters the output and compares it with the
transform computed on the host, byte for byte.

The transform doubles a floating or complex scalar, which is exact, and maps an integer `x` to
`3x + 1` in unsigned arithmetic, so it wraps without undefined behaviour.

The second test registers the worker's kernel for a layout without the last tensor. Its operation
name differs, the master never sees it advertised, and the task runs on the master's own kernel,
which gives the same bytes.
