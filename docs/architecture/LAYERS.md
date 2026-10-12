# 2. The layered architecture

```
   ┌────────────────────────────────────────────────────────────────┐
   │  Engine                 composition root: owns the scratch     │  engine.hpp
   │                         pool, the graph, the recorder, the     │
   │                         tensor factory and the random stream   │
   ├────────────────────────────────────────────────────────────────┤
   │  Operation views        BLASOps · SparseOps · ElementwiseOps   │  <family>/<family>.hpp
   │                         NonLinearOps · LogicOps · ReductionOps │  built from an Engine
   │                         RandomOps · SpatialOps · IO            │  or a TaskRecorder
   ├────────────────────────────────────────────────────────────────┤
   │  Seams                  Dispatch::execute_* and                │  dispatch.hpp of a family
   │                         Internal::execute_typed_task: take a   │  engine/task.hpp
   │                         TaskRecorder and an OperandSet, return │
   │                         a TaskHandle                           │
   ├────────────────────────────────────────────────────────────────┤
   │  TaskRecorder           the one caller of Recorder::task;      │  engine/task_recorder.hpp
   │                         stamps the execution mode on the task  │
   ├───────────────────────┬────────────────────────────────────────┤
   │  Reference kernels    │  SYCL kernels                          │  modules/kernels/<family>_kernels/
   │  plain C++ on the     │  take the queue and the dependency     │  modules/kernels/blas_seam_kernels/
   │  host                 │  events, return a sycl::event          │
   ├───────────────────────┴────────────────────────────────────────┤
   │  Support                launch and fold shapes, serial loops,  │  modules/foundation/support/
   │                         accumulators, the device policy        │
   ├────────────────────────────────────────────────────────────────┤
   │  Tensor / Storage       shape, stride and dtype over a         │  tensor/tensor.hpp, tensor/storage.hpp,
   │                         ref-counted USM allocation             │  tensor/tensor_factory.hpp
   ├────────────────────────────────────────────────────────────────┤
   │  SushiRuntime           Runtime · Graph · Recorder · Buffer ·  │  not SushiBLAS's code
   │                         Residency · Result                     │
   └────────────────────────────────────────────────────────────────┘
```

A layer depends on the layers below it and on none above. Four rules state what each layer may
know, and `tools/layering/check_layering.py` checks them from the include lines and the call
sites:

- **`Engine` knows no operation.** `engine/engine.hpp` includes the five bricks it composes
  (`engine/rng_stream.hpp`, `engine/recorded_graph.hpp`, `engine/scratch_pool.hpp`,
  `engine/task_recorder.hpp`, `engine/execution_mode.hpp`) and no view, seam or experimental
  header. It has no accessor that returns a view.
- **A view knows the engine only as a source of bricks.** A view's constructor takes an `Engine&`
  and calls `recorder()` (and `rng()` for `RandomOps`), or takes the `TaskRecorder&`
  directly. An operation source under `modules/operations/` never includes `engine/engine.hpp`.
- **Only `TaskRecorder` records a tensor operation.** No seam, view or operation source calls a
  recording member of `SushiRuntime::Recorder` (`task`, `kernel`, `state_rule`, `host`,
  `untracked_kernel`, `dynamic_kernel`, `coupled_rule`). The graph primitives of
  `modules/foundation/graph_primitives/` are the other caller: they record on a `Recorder` the
  caller hands them and know no tensor.
- **A kernel knows neither `Engine` nor `Tensor`.** A header under `modules/kernels/` includes
  no `engine/` header, no `tensor/tensor.hpp` and no sparse operand header. It receives pointers,
  extents and strides. A support header obeys the same rule and also includes no kernel.

`cmake/SushiBLASLayers.cmake` holds the tier order of the modules, and each module's README says
what it owns: [modules/README.md](../../modules/README.md) links to them.
