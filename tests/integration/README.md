# Integration tests

These tests run whole graphs through the engine and the SushiRuntime scheduler. They check that
the graph executes, compare its results with values computed on the host and report how long it
took. The suites carry the `Integration_` prefix, which `tests/CMakeLists.txt`
turns into the `integration` CTest label.

| File | What it runs |
| --- | --- |
| `test_dependency_stress.cpp` | Graphs shaped to load the dependency tracker |
| `test_framework_sim.cpp` | Graphs shaped like a framework's forward pass and model serving |

## test_dependency_stress.cpp

### ManyReadersOneWriter

One tensor is written once and then read by 5000 independent tasks, so 5000 reads are appended
to the access history SushiRuntime's dependency tracker keeps for that one allocation.

The write is a GEMM of `dummy_weights` with itself into `writer_tensor`. `writer_tensor` is the
C operand there, so the tracker records a write. Each of the 5000 following GEMMs takes
`writer_tensor` as A, so the tracker records a read.

### ShardCollisionStress

SushiRuntime's dependency tracker files an allocation under one of 1024 shards, picked by a
hash of its base address. With 3000 independent GEMMs over 9000 distinct tensors, many tensors
share a shard, and no task may gain a dependency from that. Each A is filled with a constant that depends on its
index and each B with 0.5, so the test knows what every C must hold.

## test_framework_sim.cpp

### DeepForwardPass

The graph is the forward pass of a deep network, ten layers of 512 x 1024 activations:

```
H1  = X * W1
H2  = H1 * W2
...
H_n = H_{n-1} * W_n
```

A ReLU follows every GEMM except the last. The weights are 0.01 in even columns and -0.01 in
odd ones, so the activations do not grow out of range across ten layers.

### ParallelBatchedInference

The graph serves 24 independent models at once, each a GEMM followed by a ReLU. It tests whether
the scheduler spreads independent tasks across all workers. The test was written against 12
workers, and 24 models is more models than threads.
