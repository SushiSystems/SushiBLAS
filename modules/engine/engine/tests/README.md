# Tests of engine

The unit tests of the `engine` module, built into `sushiblas_engine_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_execution_mode.cpp` | Tests how the engine's execution mode selects and refuses kernels |
| `test_operand_check.cpp` | Tests the operand checks, the scalar binding and the Tensor constructor and transpose guards |
| `test_rng_stream.cpp` | Tests RNGStream's reseed, offset reservation, thread safety and seed determinism |
| `test_scratch_shelf.cpp` | Tests the shelf of idle device scratch blocks and its size classes |
| `test_task_handle.cpp` | Tests TaskHandle: validity, equality and the handle of a call that records nothing |
| `test_task_recorder.cpp` | Tests TaskRecorder: the graph and pool it is bound to, recording from two threads, and what each mode records |
| `test_unsupported_execution.cpp` | Tests which record-time refusals throw UnsupportedExecution and which do not |
