# Contributing to SushiBLAS

Contributions from outside Sushi Systems are not accepted yet; pull requests
from outside are closed. Bug reports and design discussion are welcome as
issues. This document states how a change is built, tested, documented and
committed. SushiBLAS is source-available under the PolyForm Noncommercial
License 1.0.0 (see [`LICENSE`](../LICENSE)).

Read [the architecture](architecture/README.md) first. It explains the
`Tensor` and `Storage` model, `Engine`, the recorder and the operation views,
the three values of `ExecutionMode` and the reference path every operation has
beside its SYCL kernel.

The public interface is below 1.0.0 and may break in a MINOR release; the
[root README](../README.md#stability) states the rule a consumer follows. A
change that breaks a public name says so in its changelog entry.

---

## 1. Getting set up

The fastest path from a fresh machine to a built tree is SushiStack's one-shot
installer, which installs `hub` (SushiStack owns dependency provisioning for
the whole stack) and runs `hub install`:

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

That provisions every checkout in the workspace at once. To provision this
checkout alone, with a SushiRuntime checkout beside it, install the CLI and run
`sb setup`:

```bash
hub install-cli sushiblas  # puts `sb` / `sushiblas` on PATH
sb setup                  # toolchain and libraries for this checkout
sb build                  # Release (default)
sb test                   # functional suite
```

See [`cli/README.md`](../cli/README.md) for the full command reference and how
machine-specific toolchain paths are resolved (`cli/config.local.toml`).

---

## 2. Before you start coding

- From outside Sushi Systems, open an issue: a bug report or a design
  discussion.
- **Keep changes focused.** One logical change per commit. A refactor and a
  behavior change together are hard to review and hard to revert.
- **Match the layer boundaries.** `Engine` knows nothing about individual
  operations; the operation views (`BLASOps`, `ElementwiseOps`, ...) record
  through a `TaskRecorder` and never call a `SushiRuntime::Recorder`
  themselves; kernels know nothing about the recorder or `Engine` (see
  [the layers](architecture/LAYERS.md)). New cross-layer coupling needs a
  reason.

---

## 3. Building and testing your change

```bash
sb build                          # Release
sb build --type debug             # Debug
sb build --clean                  # wipe the build tree first

sb test --suite unit              # fast, run this often
sb test --suite integration
sb test --suite regression
sb test --suite all               # every CTest test
sb test --suite package           # the out-of-tree consumer
sb test --filter 'Gemm.*'         # ctest -R over 'Suite.Case' test names
```

A change is built and tested through `sb`. CMake can also be driven directly,
as the [root README](../README.md#building-with-cmake-directly) shows; an agent
working in this repository always uses `sb` (see `CLAUDE.md` at the root).

Expectations for a mergeable change:

1. **Every CTest test passes** (`sb test --suite all`). A change that adds a
   public header or touches the install rules in `modules/library/sushiblas/CMakeLists.txt` also
   passes `sb test --suite package`, which `all` does not include.
2. **A new operation ships with a typed test.** Every test suite runs across
   every dtype the op claims to support (`float`, `double`, `half`, and the
   complex types where applicable); see [section 5](#5-adding-a-new-operation).
   A `ForceSYCL` against `ForceReference` comparison is how a new SYCL kernel
   is checked against its reference implementation.
3. **A fixed bug gets a regression test** in
   `tests/regression/`, named for what it guards against (see
   `Regression_Sentinel.SyrkAcceptsHalfDtype` for the shape).

---

## 4. Coding style

The codebase has a deliberate, consistent style. Match the file you're editing.

### License header

Every source file opens with the license block of the `source-comments` skill:
the file name, the project line `SushiBLAS - https://github.com/SushiSystems/SushiBLAS`,
`Copyright (c) <year>-present Mustafa Garip & Sushi Systems` and the two
PolyForm Noncommercial lines.
`tools/licensing/write_license_block.py` in SushiSkills writes it; do not type
or edit it by hand. [`NOTICE.md`](../NOTICE.md) records the exact command,
which also keeps the upstream lines of the three files that carry portBLAS tiling.

### C++ standard and naming

- **C++17**, the standard SushiRuntime is built at; see
  [the architecture, section 1](architecture/OVERVIEW.md). Indentation is 4
  spaces.
- **Classes/structs/namespaces**: `PascalCase` (`Engine`, `Tensor`,
  `SushiBLAS`, `Core`).
- **Functions and variables**: `snake_case` (`create_tensor`,
  `num_elements`).
- **Private members**: trailing underscore (`engine_`, `recorder_`).
- **Accessors**: the bare name, with no `get_` prefix (`recorder()`,
  `mode()`, `layout()`).
- **Acronyms** stay in capitals inside a name (`BLASOps`, `IOStatus`,
  `RNGStream`, `ForceSYCL`).
- **Enumerators**: `PascalCase` (`DataType::Float32`, `Transpose::NoTrans`).
  An explicit value stays as written; the values of `DataType`, of the BLAS
  flags and of `IndexBase` are contracts.
- **Constants** at namespace or class scope: `K_UPPER_SNAKE`
  (`K_MAX_TENSOR_RANK`).

### Braces

Allman style (opening brace alone on its own line) for multi-statement
blocks. A single-statement `if` or loop omits braces. Put the statement on
the same line if it is short, or indented on the next line if it is long:

```cpp
// Correct: short single-line
if (condition) return;

// Correct: single statement on its own line (longer)
if (very_long_condition_that_requires_careful_checking)
    execute_specific_logic_for_this_unique_case();

// Correct: multi-statement block, Allman braces
if (condition)
{
    do_first();
    do_second();
}

// Wrong: braces on a single-statement block
if (condition)
{
    do_something();
}
```

### Include and declaration ordering

1. `#pragma once`, then exactly one empty line.
2. **Group 1: `<>` includes**, sorted shortest to longest; ties broken
   alphabetically. A header of SushiBLAS or SushiRuntime is included with
   angle brackets by its module path.
3. **Group 2: `""` includes**, for a header beside the file, as the tests
   include `tests/common/`; sorted longest to shortest, ties broken
   alphabetically.

```cpp
#pragma once

#include <cassert>
#include <cstddef>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/sushiruntime.hpp>
```

### Comments

Source comments follow the `source-comments` skill of SushiSkills: a `@file`
block after the license block, a `@brief` on every symbol, private members
included, and no `TODO`, history or design reasoning in source. Open work goes
to [`design/REMAINING_WORK.md`](design/REMAINING_WORK.md); the reason for a
design goes to a document under `design/`, cited by path.

### Error handling

An operand rule (dtype, shape, layout, flag) is checked at record time by
`engine/operand_check.hpp` and the family's describers, which throw
`std::invalid_argument` with the operation and the operand named. A refusal
by mode or device throws `UnsupportedExecution`, which derives from it.
`SB_THROW_IF(condition, message, ...)` throws `std::runtime_error` and is for
an error that is not an operand rule, as in
`modules/operations/io/source/io.cpp`, `tensor/tensor.hpp` and
`tensor/storage.hpp`. An unsupported case (dtype, layout, mode) always throws;
it never does nothing.

Nothing throws at run time. `Engine::execute()` and the graph primitives
return a `SushiRuntime::Result<void>`, and a file operation returns an
`IOStatus`. `SB_ASSERT` is for a defect inside the library, not for a
caller's input. [The architecture, section 11](architecture/EXCEPTIONS.md)
lists every channel.

### Build-system names

Every CMake target, function, macro and build-tree artifact this repository
defines starts with `sushiblas_`, as `sushiblas_warnings` and
`sushiblas_apply_build_policy` do. The bare `sushi_` prefix is not used.

Each program of the stack prefixes with its own full name (`sushiruntime_`,
`sushiengine_`, `sushiai_`). CMake targets and functions share one flat
namespace: SushiBLAS can consume SushiRuntime by `add_subdirectory`, and a
name defined twice in one configure is an error. A CMake file that defines
such names carries the program's name too, as `SushiBLASBuildPolicy.cmake`
does. `Vcpkg.cmake`, `Toolchain.cmake` and `SyclTarget.cmake` stay unprefixed
because their logic gives the same result whichever repository's copy loads
first; see [the architecture, section 10](architecture/PACKAGING.md).

The rule covers build-system identifiers and build-tree artifacts, not C++
symbols: the public C++ API is already scoped by the `SushiBLAS` namespace.

---

## 5. Adding a new operation

An operation is a view method, a seam call with an `OperandSet`, the code that
runs under each execution mode, and tests in both modes. Sections 5 and 6 of
[the architecture](architecture/ENGINE_AND_RECORDER.md)
describe the recorder and the seams this section names. What the operation
carries depends on its family.

### A kernel-pair operation (dense BLAS, sparse, spatial)

| Piece | Where |
| --- | --- |
| View method, returning `TaskHandle`, with its `@throws` lines | The family's view header: `blas/level<N>.hpp`, `sparse/sparse.hpp`, `spatial/spatial.hpp` |
| Operation source | `modules/operations/<family>/source/<op>.cpp`, listed in the module's `CMakeLists.txt` |
| Reference kernel | `modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/reference/<op>.hpp` |
| SYCL kernel | `modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/sycl/<op>.hpp` |
| Both kernels' include line | The family's kernel header: `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/blas.hpp`, `modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sparse.hpp`, `modules/kernels/spatial_kernels/include/SushiBLAS/spatial_kernels/spatial.hpp` |
| Test | `modules/operations/<family>/tests/test_<op>.cpp`, listed in the module's `CMakeLists.txt` |

The operation source has four steps. This is `modules/operations/blas/source/level1/axpy.cpp`:

```cpp
TaskHandle Level1::axpy(BLASScalar alpha, const Tensor& x, Tensor& y)
{
    const char* const op = "blas.lvl1.axpy";

    // 1. Record-time checks, each naming the operation and the operand.
    Dispatch::require_dtype(y, x.dtype, op, "y");
    Dispatch::require_real_scalar(alpha, x.dtype, op, "alpha");
    const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
    const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
    Dispatch::require_same_element_count(x, y, op, "x", "y");

    // 2. The operand set: what the task reads and writes, as tensors.
    OperandSet operands;
    operands.reads(x).updates(y);

    const Internal::BLASMetadata metadata =
        Internal::BLASMetadata(op, "blas.lvl1.axpy"_op).scalar(alpha);

    // 3. Everything the body needs, captured by value.
    const int64_t n = vx.length;
    const int64_t incx = vx.increment;
    const int64_t incy = vy.increment;
    const void* read_x = x.data_or_null();
    void* rw_y = y.data_or_null();

    // 4. The seam call: one task, one kernel pair.
    return Internal::execute_blas_task(recorder_, metadata.get(), operands, x.dtype,
        [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
            sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
        {
            using T = decltype(scalar_type);
            auto pX = static_cast<const T*>(read_x);
            auto pY = static_cast<T*>(rw_y);
            const T alpha_t = alpha.as<T>();

            return dispatcher.run(q, deps,
                [=]
                {
                    Kernels::Reference::axpy<T>(n, alpha_t, pX, incx, pY, incy);
                },
                [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return Kernels::SYCL::axpy<T>(q2, n, alpha_t, pX, incx, pY, incy, deps2);
                });
        });
}
```

The numbered comments are this page's and are not in the source file.

**1. Check at record time.** Every operand rule goes through
`engine/operand_check.hpp` or the family's describers (`operands/matrix_operand.hpp`,
`operands/vector_operand.hpp`, `operands/argument_check.hpp`, the `*_operand.hpp` headers
under `blas/` and `sparse/`, `spatial/window.hpp`),
which throw `std::invalid_argument` with the operation and the operand named. A
refusal by mode or device is `UnsupportedExecution`: `require_device_dtype`
throws it, the describers call it, and the recorder throws it for a device-only
operand under `ForceReference`. Do not write `SB_THROW_IF` for an operand
rule; it throws `std::runtime_error`. A new rule that two operations share is a
new function in `operand_check.hpp` or in the family's check header, not a copy.

**2. Declare every tensor in the `OperandSet`.** `reads` for an input, `writes`
for an output that is only written, `updates` for one changed in place, and
`scratch` for a leased `ScratchBuffer`. Pass tensors, never a region built from
a pointer. An output that is read only when a scalar is non-zero (`beta`) is
declared `writes` when the scalar is zero and `updates` otherwise. The set is
also what keeps the storage alive and what `ForceReference` is checked
against, so a tensor left out of it is unordered, unretained and unchecked.

**3. Capture by value.** The body runs later and may run more than once. It
holds extents, strides, flags and the pointers `data_or_null()` returned. It
holds no `Tensor`, no reference to a local and no reference to the view.

**4. Call the seam.** `Internal::execute_blas_task` for dense BLAS,
`Internal::execute_typed_task` with the family string for sparse and spatial.
Pass the `Dispatch::Domain` when the operation does not serve the default,
which is `RealAndComplex` for both: `Domain::All` for a copy, `Domain::Real`
for a routine with no complex meaning. Inside the body, `dispatcher.run` takes the
reference kernel and the SYCL kernel. An operation source never reads the
execution mode, never includes `engine/engine.hpp` and never calls
`SushiRuntime::Recorder::task`; `tools/layering/check_layering.py` reports the last two.

**The reference kernel** is plain C++ with no dependency beyond the support
headers. It handles both `Core::Layout` values, is templated on the element
type, and forms its sums in `Kernels::ReferenceSum<T>`
(`support/reference_accumulator.hpp`). It is written to be read: an
optimisation in it is a second implementation that can be wrong.

**The SYCL kernel** takes the `sycl::queue&` and a
`const std::vector<sycl::event>& deps` and returns the event of its last
submission. Every command group calls `h.depends_on(deps)`, and a second
launch depends on the event of the first. It launches through `Launch::` or
`Reduce::` where one of their shapes fits, forms its sums in
`Kernels::Accumulator<T>` (HALF in FLOAT32, `support/accumulator.hpp`), and
keeps the order of every floating-point sum a function of the device and the
extents: no atomic add, no `sycl::reduction`. Scratch it needs is leased by the
operation source at record time and passed in.

Neither kernel includes an `engine/` header or `tensor.hpp`. A kernel that a
public `Dispatch` seam instantiates in a consumer's translation unit goes under
`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/`; every other kernel stays in `modules/kernels/`.

### A functor operation (elementwise, non-linear, logic, reductions, random)

These families have no kernel files. The operation is a view method and a
source that hands one functor to the family's `Dispatch::execute_*` seam. The
seam runs the checks, builds the `OperandSet`, and runs the functor in a SYCL
kernel or, under `ForceReference`, in a serial host loop. This is
`modules/operations/elementwise/source/add.cpp`:

```cpp
TaskHandle ElementwiseOps::add(const Tensor& A, const Tensor& B, Tensor& C)
{
    return Dispatch::execute_binary<Dispatch::Domain::All>(
        recorder_, A, B, C, "math.ew.add", "math.ew.add"_op,
        [](auto x, auto y) { return Dispatch::wrapping_add(x, y); });
}
```

- State the `Domain` the functor is well-formed for. The default is
  `Domain::Real`. A functor instantiated for complex uses the `safe_*` helpers
  of `support/safe_math.hpp`, and one instantiated for the integers uses
  `elementwise/wrapping_math.hpp`.
- The functor reaches a device: it is trivially copyable, captures by value,
  holds no host pointer, allocates nothing and throws nothing.
- A scalar parameter is bound with `Dispatch::bind_scalars` as a
  `Dispatch::ElementScalar` and also listed in the seam's `params`, which
  writes it to the parameter slots of the task's `TaskDescription`. Do not
  capture a `double` in the functor.
- A tensor the functor reads beyond the seam's operands is declared with
  `Dispatch::bind_reads`. A pointer captured in the functor is not declared.
- A reduction passes a dispatcher with both members, the SYCL fold and
  `reference`, or calls `execute_folding_reduction` with an accumulator, a
  transform and a finalize functor.
- An operation that needs a check its seam does not make (a parameter range, a
  second dtype) makes it in the source, through `operand_check.hpp`, before the
  seam call.

### Tests

- **Typed tests** over every `Core::DataType` the operation serves.
- **A hand-computed expectation** for at least one case. Comparing the two
  modes is a differential check, and it passes when both paths are wrong the
  same way. For a functor operation the two modes share the formula, so only a
  hand-computed value tests it.
- **Both modes.** Record the operation under `ForceSYCL` and under
  `ForceReference` into two tensors and compare them. Use
  `SB_ASSERT_TENSOR_NEAR` or `EXPECT_TENSOR_NEAR` with a tolerance for the
  dtype, and compare bits only where the two paths are defined to agree.
- **Each record-time refusal**, asserting the type: `std::invalid_argument`
  for a wrong operand, `SushiBLAS::UnsupportedExecution` for a mode or device
  refusal.
- **Ordering on the dependency count.** A test that the task is ordered
  against another asserts on the task's `dependency_count`, read from
  `GraphProbe::recorded_tasks()` (`tests/common/graph_probe.hpp`), not on an
  output value.
- **More than one shape**: square, rectangular, a vector, an empty operand,
  and for BLAS both layouts and each flag value.

[`tests/unit/README.md`](../tests/unit/README.md) holds
the reasoning behind these rules.

---

## 6. Documentation is part of the change

Documentation lands in the same commit as the code it describes. Where a
document goes follows the `documentation` skill of SushiSkills; the tree is
indexed in [`README.md`](README.md).

- **Every change adds one changelog line** under `## Unreleased` in
  [`reference/CHANGELOG.md`](reference/CHANGELOG.md): the date, the commit
  scope, a past-tense verb, what changed and where, in at most 240 characters.
  The line never says why.
- **A new or changed operation updates
  [the architecture](architecture/README.md)** when it changes a class,
  dispatch helper or kernel design the document names. Every symbol it names
  must still exist.
- **A fact about one module goes in that module's `README.md`**, and the
  manual links to it.
- **A code block in the manual comes from a file the build compiles** and
  names that file; see the [style guide](DOCUMENTATION_STYLE_GUIDE.md).
- **A CLI change updates [`cli/README.md`](../cli/README.md)** and the build
  and test snippets here and in the root README.
- **A design decision with an open phase** is a document under `design/` with a
  status line, listed in [`design/README.md`](design/README.md). Open work is a
  line in [`design/REMAINING_WORK.md`](design/REMAINING_WORK.md), the only
  backlog.
- **A defect that stays open** is a row in
  [`reference/KNOWN_ISSUES.md`](reference/KNOWN_ISSUES.md).
- **Agent work** is written to its own folder,
  `docs/agent/<YYYY_MM_DD>_<WORK_NAME>/`, as `SPEC.md`, `PLAN.md` and
  `REPORT.md`.
- **A new dependency updates the dependency surface**; see
  [section 8](#8-adding-a-c-dependency).

Before a commit that touches `docs/`, run the two documentation checkers; see
[`tools/README.md`](../tools/README.md).

```bash
python tools/documentation/check_docs_layout.py .
python tools/documentation/check_changelog.py .
```

---

## 7. Commits

### Branch names

A type prefix and a short, lowercase, hyphenated description:

```
feature/trsm-right-side
fix/gemm-beta-zero-nan
chore/bump-googletest
refactor/operand-describers
```

### Commit headers

`type(scope): short summary`, in lower case, with no trailing period. The
scope is the module or subsystem folder, and a change that breaks a public
interface puts `!` before the colon:

```
feat(blas): port trsm off oneMKL onto a from-scratch tiled kernel
fix(gemm): skip reading C when beta is exactly zero
docs(architecture): describe the engine as the code has it
fix(api)!: close the findings of the review made before the first build
```

### Commit body

Bullet points, each starting with a past-tense verb:

```
- Added a from-scratch SYCL kernel for TRSM (level3) and TRSV (level2).
- Fixed a NaN-propagation bug in GEMM/SYRK when beta == 0.
- Removed the now-dead level1_internal.hpp after the level1 port.
```

Plain English, technical, objective: state the mechanism and the effect, with
no marketing language.

---

## 8. Adding a C++ dependency

Dependency provisioning is owned by [SushiStack](https://sushisystems.io)
(the `hub` CLI). SushiBLAS's own manifest fragment is
[`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml):

```toml
[module]
depends_on = ["sushiruntime"]
```

`depends_on` folds SushiRuntime's own toolchain requirements (the SYCL
compiler, hwloc, GoogleTest) into SushiBLAS's readiness picture, so nothing
SushiRuntime already provisions is declared again. If SushiBLAS
ever needs a library of its own, add a top-level TOML table the same way
SushiRuntime's own fragment does:

```toml
[mylib]
description   = "What it is and why it is needed"
required      = true
linux_apt     = ["libmylib-dev"]
windows_vcpkg = ["mylib"]
check_cmd     = ["pkg-config", "--exists", "mylib"]
```

Then also update `CMakeLists.txt` (a `find_package`/`pkg_check_modules` call
and the link line) if the C++ build needs to find it. Verify with
`sb doctor` (confirms the entry parses) and `sb setup`.

---

## 9. Reporting bugs and requesting features

Open a GitHub issue. A good bug report includes the platform and
toolchain, the build type, the exact `sb` command, and the smallest
reproducer you can manage. For anything that looks like a **security** issue,
do not open a public issue; follow [`SECURITY.md`](../.github/SECURITY.md).

---

## 10. Questions

Open a GitHub Discussion, join the community Discord (see
[`CODE_OF_CONDUCT.md`](../.github/CODE_OF_CONDUCT.md) for the invite), or email
**mustafagarip@sushisystems.io**.

Conduct in project spaces is covered by
[`CODE_OF_CONDUCT.md`](../.github/CODE_OF_CONDUCT.md).
