# 2. The layered architecture

```
   ┌────────────────────────────────────────────────────────────────┐
   │  Engine                 composition root: owns the scratch     │  engine.hpp
   │                         pool, the graph, the recorder, the     │
   │                         tensor factory and the random stream   │
   ├────────────────────────────────────────────────────────────────┤
   │  Operation views        BLASOps · SparseOps · ElementwiseOps   │  engine/*.hpp
   │                         NonLinearOps · LogicOps · ReductionOps │  built from an Engine
   │                         RandomOps · SpatialOps · IO            │  or a TaskRecorder
   ├────────────────────────────────────────────────────────────────┤
   │  Seams                  Dispatch::execute_* and                │  engine/**/dispatch.hpp
   │                         Internal::execute_typed_task: take a   │  engine/task.hpp
   │                         TaskRecorder and an OperandSet, return │
   │                         a TaskHandle                           │
   ├────────────────────────────────────────────────────────────────┤
   │  TaskRecorder           the one caller of TaskGraph::add_task; │  engine/task_recorder.hpp
   │                         stamps the execution mode on the task  │
   ├───────────────────────┬────────────────────────────────────────┤
   │  Reference kernels    │  SYCL kernels                          │  kernels/,
   │  plain C++ on the     │  take the queue and the dependency     │  include/SushiBLAS/kernels/
   │  host                 │  events, return a sycl::event          │
   ├───────────────────────┴────────────────────────────────────────┤
   │  Support                launch and fold shapes, serial loops,  │  include/SushiBLAS/support/
   │                         accumulators, the device policy        │
   ├────────────────────────────────────────────────────────────────┤
   │  Tensor / Storage       shape, stride and dtype over a         │  tensor.hpp, storage.hpp,
   │                         ref-counted USM allocation             │  tensor_factory.hpp
   ├────────────────────────────────────────────────────────────────┤
   │  SushiRuntime           TaskGraph · RuntimeContext ·           │  not SushiBLAS's code
   │                         Residency · execution engine           │
   └────────────────────────────────────────────────────────────────┘
```

A layer depends on the layers below it and on none above. Four rules state what each layer may
know, and `tools/layering/check_layering.py` checks them from the include lines and the call
sites:

- **`Engine` knows no operation.** `engine.hpp` includes the four bricks it composes
  (`engine/rng_stream.hpp`, `engine/scratch_pool.hpp`, `engine/task_recorder.hpp`,
  `engine/execution_mode.hpp`) and no view, seam or experimental header. It has no `blas()` or
  `elementwise()` accessor.
- **A view knows the engine only as a source of bricks.** A view's constructor takes an `Engine&`
  and calls `get_recorder()` (and `get_rng()` for `RandomOps`), or takes the `TaskRecorder&`
  directly. An operation source under `source/engine/` never includes `engine.hpp`.
- **Only `TaskRecorder` inserts into the graph.** No seam, view or operation source calls
  `TaskGraph::add_task`.
- **A kernel knows neither `Engine` nor `Tensor`.** A header under `source/kernels/` or
  `include/SushiBLAS/kernels/` includes no `engine/` header, no `tensor.hpp` and no sparse
  operand header. It receives pointers, extents and strides. A support header obeys the same rule
  and also includes no kernel.
