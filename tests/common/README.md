# Shared test support

Support code the functional tests include.

| File | What it holds |
| --- | --- |
| `test_main.cpp` | `main`: synchronous logging, the environment, then `RUN_ALL_TESTS` |
| `test_environment.hpp` | The GoogleTest environment that owns the one `SushiRuntime::Runtime` |
| `test_utils.hpp` | Tensor fill and comparison helpers, the `SushiBLASTest` fixture, the suite macros |
| `graph_probe.hpp` | `GraphProbe`, a graph a test owns so it can read what its record calls added |
| `reference_reductions.hpp` | Serial reductions over a host vector, in namespace `ReferenceKernels` |
| `blas_flags.hpp` | The mappings from a test's bool parameters to the BLAS flag enumerations |
| `blas_truth.hpp` | The double-precision truth a BLAS parity test measures both kernels against |
| `blas_structured.hpp` | The conversions from a dense test matrix to CBLAS band and packed storage |
| `packed_product_harness.hpp` | The problem, the data and the blockings the packed product tests share |
| `sparse_samples.hpp` | The host matrices the sparse tests build from and the host products they expect |

## test_environment.hpp

`TestEnvironment` creates the runtime in `SetUp()` with `SushiRuntime::Runtime::create()` and
hands it out through `runtime()`. `queue(device_index)` and `device_count()` forward to the
runtime's `advanced()` view.

`live_allocation_count()` returns how many USM allocations of the runtime are live; the lifetime
tests count with it. It forwards to `AdvancedControls::live_allocation_count()`.

## graph_probe.hpp

`Engine` has no accessor for its graph, so a test that asserts on what an operation declared
cannot read it from an engine. `GraphProbe` owns a `ScratchPool`, a `SushiRuntime::Graph` and a
`TaskRecorder` over them on the shared test runtime. A test builds a view on
`probe.recorder()` and records. `recorded_count()` returns how many tasks wait on the graph, and
`take_recorded_count()` returns that count and clears the graph. `execute()` runs what is
recorded and returns the result, as `Engine::execute` does. The constructor takes the execution
mode the recorder records under.

`recorded_tasks()` is the view a test reads a recorded task through. It calls `Graph::plan()`,
which compiles the graph when no plan is current and runs nothing, and copies each `PlanNode`
into a `RecordedTask`: the task's `name` and `operation`, its `dependency_count` (the node's
`predecessor_count`), and the regions it declared as `reads` and `writes`. It returns a
`SushiRuntime::Result`: the tasks in record order, an empty list when nothing is recorded, or the
error that kept the graph from compiling. `dependency_counts()` returns the counts alone. A test
reads the tasks before `execute()`, which leaves the graph empty. The plan holds no task body, so
a test cannot call one; it runs a task through `execute()`.

## blas_flags.hpp

The BLAS suites are parameterised over `bool` values (transposed or not, upper or lower), and the
public API takes `Core::Transpose`, `Core::Uplo`, `Core::Diag` and `Core::Side`. The four
functions here map one to the other: `true` is `Trans`, `Upper`, `Unit` and `Left`. A test names
the flag it means at the call site and keeps its parameter list.

## blas_truth.hpp

The reference kernel sums in `double` and the SYCL kernel in the element type, FLOAT32 for HALF, so the two do not
agree bit for bit on a rounded sum, and a comparison of one against the other cannot say which is
off. Both are measured against a third value instead.

`TruthSum` adds products of doubles and keeps the rounding error of each product and each
addition, so its sum is what twice the working precision would give. `value()` returns the sum
together with the magnitude behind it, the sum of the absolute values of the products.
`rounding_bound(roundings, unit, magnitude)` is the largest error that many rounded operations
can leave on a sum of that magnitude, and `unit_roundoff<Real>()` is the unit of the element
type. A test asserts that each kernel lies within the bound of the truth.

## blas_structured.hpp

Builds the operands of the band and packed tests from a dense matrix given row by row:
`in_layout` orders it for a layout, `to_band` and `to_triangle_band` lay it out as CBLAS band
storage, and `to_packed` as CBLAS packed storage of one triangle. The expected result of a band or
packed routine is then the result of its dense sibling on the same matrix.

## packed_product_harness.hpp

`GemmProblem<T>` owns the three operands of one batched GEMM in shared USM, with leading
dimensions and batch strides larger than the extents, and runs the reference kernel, the packed
kernel under a blocking a test chooses, and the kernel that takes no scratch on the same data.
Each run returns the whole C buffer, padding included, so a test can check that no kernel wrote
outside its matrices. After `poison_output()` C starts each run as NaN, so an element a kernel
failed to write shows.

The three blockings are the edges of the plan: every block one register block and one term deep,
every block larger than any matrix, and a depth of 7 with two register blocks of rows and three
of columns. Integer-valued data makes every sum exact, so the kernels can be compared bit for
bit whatever order they add in.

## sparse_samples.hpp

`sample_matrix()` is a 4 x 5 matrix of seven entries with an empty row and no two values equal,
listed out of row order so that assembly has to sort. `HostTriplets` holds a matrix as zero-based
triplets; the expected product of a test is computed from them on the host in double, by a loop
that shares no code with either kernel. `K_BOTH_BASES` and the list of execution modes are what
every sparse suite runs over, and the typed suites cover every value dtype with both index
widths.

## The `Unit_Level2SyclParityTest` fixture

`modules/operations/blas/tests/test_level2_sycl_parity.cpp` runs GEMV and SYMV in a chosen execution mode
and measures each against a `TruthSum`, as described under `blas_truth.hpp` above.

## test_utils.hpp

### The complex-type trait

The helpers use the library's own trait, `SushiBLAS::Core::is_complex_v` from
`<SushiBLAS/core/traits.hpp>`. This file once carried its own copy, the fourth in the project.
The project now has exactly one.

### fill_tensor on integer tensors

The `INT32` and `INT64` arms truncate, like every other arm: the caller states the values it
wants and the tensor's dtype decides how they are stored. Filling an integer tensor from
`{1.0f, 0.0f}` is the natural way to write a comparison operand or an expected truth vector.

### Dense tensors only

Every helper here reads or writes `num_elements` consecutive elements from `tensor.data()`. They
handle dense, contiguous tensors only. A strided view or a view at an offset needs its own
comparison, as `modules/data/tensor/tests/test_storage_offset.cpp` does.

### verify_tensor

`verify_tensor` takes a `std::vector<float>` or, through a template overload, a
`std::vector<double>`, `std::vector<std::complex<float>>` or `std::vector<std::complex<double>>`.
A braced list such as `{1.0f, 2.0f}` picks the `float` overload, because a braced list cannot
deduce the template's element type.

On a complex tensor both parts are compared. A real expectation asks for a zero imaginary part.

A NaN expectation passes only on a NaN element, and an infinite expectation only on the same
infinity. `EXPECT_NEAR` alone fails both, since the difference it computes is NaN.

Integer elements are compared with `EXPECT_EQ`. An integer element is the expected value or it
is not, and a tolerance would only hide an off-by-one.

### fill_random

`fill_random` draws from one `std::mt19937` seeded with `K_FILL_RANDOM_SEED`. `SushiBLASTest`
restarts it in `SetUp`, so a test sees the same inputs whether ctest runs it alone or the test
binary runs the whole suite. Two calls in one test draw different values.

A complex element takes two draws, the real part first and the imaginary part second.

On a floating dtype the values lie in `[min_val, max_val]`, not `[min_val, max_val)`: the draw
is a `float` below `max_val`, and rounding it to `HALF` (or to `float` from a nearby double)
can land on `max_val` itself.

On an integer dtype each draw is truncated toward zero. With the default range `(-1, 1)` every
element is 0. A test that needs varied integers passes a wider range or uses `fill_tensor`.

`read_tensor<T>` copies a tensor's elements into a host vector for checks that `verify_tensor`
cannot express, such as sample statistics. It throws `std::invalid_argument` when `T` does not
map to the tensor's dtype, before it copies anything.

### DefaultTypeNames and SB_TYPED_TEST_SUITE

`TYPED_TEST_SUITE(Suite, Types)` looks like a two-argument macro and is declared
`TYPED_TEST_SUITE(CaseName, Types, ...)`. The `...` is an optional name generator. Passing no
argument at all for a `...` is allowed from C++20 and is a diagnosed extension before it. On this
project's C++17 baseline the two-argument spelling is an error under `-pedantic -Werror`, raised
inside a third-party header the project does not own and cannot fix.

Naming a generator explicitly settles it. `DefaultTypeNames` restates GoogleTest's default
against its public API, without reaching into its `internal` namespace. The suffix is the type's
index, as before, so every generated test name is unchanged and the fix renames no test.
