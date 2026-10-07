# Package consumer

This folder is a separate CMake project. It finds an installed SushiBLAS through
`find_package`, builds `main.cpp` against it and runs the result. It has no access to the
sushiblas source tree. `sb test --suite package` installs the package and drives it; see
`cli/README.md`.

## main.cpp

### What is under test

The program is the smallest thing that proves the package works: a real op (AXPY, one of SB-3's
from-scratch kernels), recorded and run against an installed SushiBLAS.

It is small on purpose. Anything more would test SushiBLAS itself, which its own suite does.
What is under test is the package: that the headers are found, that the kernel template compiles
in this project's translation unit under the toolchain the consumer chose, and that the library
links and runs end to end through SushiBLAS and its SushiRuntime dependency.

The second thing under test came with the promotion of the five `*_internal.hpp` op-dispatch
helpers out of `src/` into the installed headers: that `SushiBLAS::Dispatch` is reachable from
here. `execute_binary` receives a lambda this file wrote, and it compiles and runs. That is the
difference between "the fusion seam was moved" and "the fusion seam is public API". Nothing in
sushiblas's own suite can show it, because in-tree every header is on the include path whether
it gets installed or not.

The third came with the promotion of the GEMM epilogue to a public seam: that
`SushiBLAS::Dispatch::execute_gemm` reaches a consumer with the consumer's own epilogue functor.
This one differs from the others. It instantiates a real tiling kernel
(`kernels/.../level3/gemm.hpp`) in this translation unit, so it also checks that the kernel
header is installed and not merely present in the source tree next door.

### ConsumerBiasRelu

The functor is a column-bias broadcast plus ReLU, the shape a linear layer wants. `col` is what
makes the broadcast expressible, and `row` selects the per-row term. The functor reads both, so
a seam that passed only one of them would fail this program.

An activation fused into a forward training pass is silently wrong unless the pre-activation is
also spilled; the warning on `Dispatch::execute_gemm` says why. This program is
inference-shaped, so it is not affected.

### The fused binary kernel

`execute_binary` runs multiply, add and divide as one device-side expression. No SYCL is written
in this file, and the graph holds one task where separate ops would make three.

### The variadic in-place seam

`execute_zip_inplace` advances three tensors together with one consumer-authored functor, in one
task. It also proves that `<SushiBLAS/support/launch_1d.hpp>`, the header that holds the seam's
launch geometry, reaches an installed consumer. Nothing in-tree can show that.

### The GEMM epilogue numbers

The program computes a 3x3 product with its own bias-plus-ReLU functor folded into the store.
Spelled as separate ops, the bias and the ReLU would cost two more kernel launches over C.

```
A (3x2)  = [[1,0],[0,1],[1,1]]
B (2x3)  = [[1,2,3],[4,5,6]]
A*B      = [[1,2,3],[4,5,6],[5,7,9]]
row_bias = [0,-6,5]
col_bias = [1,-8,3]
biased   = [[2,-6,6],[-1,-9,3],[11,4,17]]
```

The ReLU clamps the three negatives. C is square so that a seam which transposed row and col
would produce a different answer here and not an out-of-range read.
