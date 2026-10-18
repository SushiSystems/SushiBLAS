# A first program

A GEMM recorded on an engine and read back, then what the library does with it.

```cpp
#include <vector>
#include <cstdint>
#include <SushiBLAS/SushiBLAS.hpp>

namespace sb = SushiBLAS;
namespace sr = SushiRuntime;

int main()
{
    constexpr int64_t K_SIZE = 256;
    constexpr float K_A_VALUE = 1.0f;
    constexpr float K_B_VALUE = 2.0f;

    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime, sb::Core::Layout::ColumnMajor);

    auto A = engine.create_tensor({K_SIZE, K_SIZE});
    auto B = engine.create_tensor({K_SIZE, K_SIZE});
    auto C = engine.create_tensor({K_SIZE, K_SIZE});

    const size_t count = static_cast<size_t>(K_SIZE * K_SIZE);
    SushiBLAS::IO(engine).from_vector(A, std::vector<float>(count, K_A_VALUE));
    SushiBLAS::IO(engine).from_vector(B, std::vector<float>(count, K_B_VALUE));

    SushiBLAS::BLASOps(engine).gemm(A, B, C);

    const std::vector<float> result = SushiBLAS::IO(engine).to_vector<float>(C);

    // ...
}
```

Source: `applications/examples/quickstart_demo.cpp`

The omitted lines compare every element of `C` with the value the host computes, print one element and return 1 on a mismatch. `create_tensor` gives FLOAT32 tensors in shared USM by default. `gemm` records a task and runs nothing. `IO::to_vector` runs the recorded graph and then copies `C` to the host, so the example never calls `execute()` itself.

A program that reads no tensor through `IO` runs the graph with `Engine::execute()`:

```cpp
SushiBLAS::BLASOps(engine).axpy(2.0f, x, y);
engine.execute().value();
```

Source: `tests/package/main.cpp`

`execute()` returns a `SushiRuntime::Result<void>` and is `[[nodiscard]]`. `.value()` on a result that holds an error stops the process; a caller that can go on tests the result and reads `error()`.

`Engine` is built on a `SushiRuntime::Runtime`, which must outlive it. It owns the graph and has no accessor per operation family. A family is a view class built from the engine: `BLASOps`, `SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`, `SpatialOps` and `IO`. A view holds references and no other state. Each record call returns a `TaskHandle`.

The BLAS flags are the enumerations `Core::Transpose`, `Core::Uplo`, `Core::Diag` and `Core::Side`, and a scalar argument is a `BLASScalar`, which converts from a `double` or a `std::complex<double>`.

`Tensor` is a metadata view (shape, strides, rank, `Core::DataType`, layout) over a ref-counted `Storage`. `transpose()`, `reshape()` and `slice()` return new `Tensor` views over the same storage and move no data. A recorded task keeps the storage of its operands alive until the graph drops it. [The architecture](../architecture/README.md) describes each layer.

## What a program can count on

- Operations are recorded, then run. `Engine::create_tensor()` allocates a `Tensor`, which is shape, stride and dtype metadata over a ref-counted `Storage` backed by SushiRuntime USM. Calling an operation, such as `SushiBLAS::BLASOps(engine).axpy(...)`, records one task on the engine's graph and returns a `TaskHandle`. `engine.execute()` runs the graph, returns when every task has finished, and reports the run as a `SushiRuntime::Result<void>`.
- There are eight operation families: dense BLAS levels 1 to 3 with the band, packed, rotation, complex and Hermitian routines; sparse BLAS over CSR; elementwise arithmetic; non-linear activations; logic and comparison; reductions; random generation; and spatial (windowed) operations. [The architecture, section 7](../architecture/KERNELS.md) lists the routines.
- Every operation has a reference path and a SYCL kernel. The BLAS, sparse and spatial families have a plain C++ reference kernel in a `reference/` folder beside the SYCL kernel in a `sycl/` folder, under `modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/`. The other families run one functor two ways: inlined into a SYCL kernel, or in a serial host loop. The GEMM and SYRK tiling is ported from portBLAS; see [NOTICE.md](../../NOTICE.md).
- The execution mode is stamped on each task as it is recorded. `Engine::set_mode()` selects `Auto`, `ForceSYCL` or `ForceReference` for every later operation. See [the architecture, section 3](../architecture/EXECUTION_MODES.md).
- A wrong call throws when it is recorded. A wrong operand, flag or scalar throws `std::invalid_argument` and names the operation and the operand; a dtype with no meaning for an operation, such as `max` on a complex tensor, is one of those. An operation that cannot run under the mode or on the device throws `SushiBLAS::UnsupportedExecution`, which derives from it. A failed run throws nothing: `execute()` returns the error. See [the architecture, section 11](../architecture/EXCEPTIONS.md).
- Fusion is written by the caller. The `SushiBLAS::Dispatch` seams take a functor and run it as one kernel, and `Dispatch::execute_gemm` takes an epilogue applied to each element of the product. No pass fuses recorded tasks automatically.
