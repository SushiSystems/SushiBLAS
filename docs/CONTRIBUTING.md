# Contributing to SushiBLAS

Contributions from outside Sushi Systems are not accepted yet; pull requests
from outside are closed. Bug reports and design discussion are welcome as
issues. This document states how a change is built, tested, documented and
committed. SushiBLAS is source-available under the PolyForm Noncommercial
License 1.0.0 (see [`LICENSE`](../LICENSE)).

Read [the architecture](architecture/ARCHITECTURE.md) first. It explains the
`Tensor` and `Storage` model, the `Engine` facade, the `ExecutionMode` triad
and the reference and SYCL kernel pair every operation follows.

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
  operations; op classes (`BLASOps`, `ElementwiseOps`, ...) know nothing about
  the task graph beyond `engine.get_graph()`; kernels know nothing about
  `Engine` at all (see ARCHITECTURE §2). New cross-layer coupling needs a
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
   public header or touches the install rules in `src/CMakeLists.txt` also
   passes `sb test --suite package`, which `all` does not include.
2. **A new operation ships with a typed test.** Every test suite runs across
   every dtype the op claims to support (`float`, `double`, `half`, and the
   complex types where applicable) — see [§5](#5-adding-a-new-operation).
   A `FORCE_SYCL` against `FORCE_REFERENCE` comparison is how a new SYCL kernel
   is checked against its reference implementation.
3. **A fixed bug gets a regression test** in
   `tests/functional/regression/`, named for what it guards against (see
   `Regression_Sentinel.SyrkAcceptsHalfDtype` for the shape).

---

## 4. Coding style

The codebase has a deliberate, consistent style. Match the file you're editing.

### License header

Every source file opens with the license block of the `source-comments` skill:
the file name, the project line `SushiBLAS - https://github.com/SushiSystems/SushiBLAS`,
`Copyright (c) <year> Sushi Systems` and the two PolyForm Noncommercial lines.
`tools/licensing/write_license_block.py` in SushiSkills writes it; do not type
or edit it by hand. [`NOTICE.md`](../NOTICE.md) records the exact command,
which also keeps the upstream lines of the two files ported from portBLAS.

### C++ standard and naming

- **C++17**, matching SushiRuntime — see
  [the architecture, section 1](architecture/ARCHITECTURE.md#1-what-sushiblas-is) for why the whole
  stack has to agree rather than each repo picking its own. Indentation is 4
  spaces.
- **Classes/structs/namespaces**: `PascalCase` (`Engine`, `Tensor`,
  `SushiBLAS`, `Core`).
- **Functions and variables**: `snake_case` (`create_tensor`,
  `num_elements`).
- **Private members**: trailing underscore (`engine_`, `data_ptr_`).

### Braces

Allman style (opening brace alone on its own line) for multi-statement
blocks. A single-statement `if`/loop **omits** braces — put the statement on
the same line if it's short, or indented on the next line if it's long:

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
2. **Group 1: `<>` includes** (system headers) — sorted **shortest to
   longest**; ties broken alphabetically.
3. **Group 2: `""` includes** (local headers) — sorted **longest to
   shortest**; ties broken alphabetically.

```cpp
#pragma once

#include <cassert>
#include <cstddef>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
```

### Comments

Source comments follow the `source-comments` skill of SushiSkills: a `@file`
block after the license block, a `@brief` on every symbol, private members
included, and no `TODO`, history or design reasoning in source. Open work goes
to [`design/REMAINING_WORK.md`](design/REMAINING_WORK.md); the reason for a
design goes to a document under `design/`, cited by path.

The existing source does not meet this yet.
[`reference/KNOWN_ISSUES.md`](reference/KNOWN_ISSUES.md) records the gap.

### Error handling

`SB_THROW_IF(condition, message, ...)` for errors a caller can act on
(unsupported dtype, shape mismatch, invalid storage). Assertions are for
internal bugs, not user-facing input validation. An unsupported case
(dtype, layout, mode) always throws loudly — it never silently no-ops.

### Build-system names

Every CMake target, function, macro, and internal artifact this repo defines
starts with `sushiblas_` — `sushiblas_warnings`, `sushiblas_apply_build_policy`,
`.sushi_env.json`. Never the bare `sushi_` prefix.

This is a **SushiStack-wide convention**, not a SushiBLAS one: each program
prefixes with its own full name (`sushiruntime_`, `sushiengine_`, `sushiai_`).
The reason is that these names share a single flat namespace the moment two
of them meet — SushiBLAS consumes SushiRuntime by `add_subdirectory`, and a
CMake target or function name defined twice in one configure is a hard error
with no scoping to fall back on. `sushi_warnings` defined by both repos
collides; `sushiblas_warnings` and `sushiruntime_warnings` cannot. Files whose
logic is genuinely idempotent regardless of which sibling's copy loads first
(`Vcpkg.cmake`, `Toolchain.cmake`, `SyclTarget.cmake`) are the deliberate
exception and stay unprefixed; a file that defines a *differently-named*
function per repo (`SushiBLASBuildPolicy.cmake` vs. a hypothetical
`BuildPolicy.cmake` shared with sushiruntime) must be prefixed — this is
exactly the collision SB-6 found and fixed (see
[the architecture, section 10](architecture/ARCHITECTURE.md#10-packaging)).

The rule covers build-system identifiers and build-tree artifacts, not C++
symbols: the public C++ API is already scoped by the `SushiBLAS` namespace.

---

## 5. Adding a new operation

Every operation (BLAS, elementwise, non-linear, logic, reduction) follows the
same shape, regardless of category:

### Directory structure

- **Reference kernel**: `kernels/reference/<category>/<sub-category>/<op>.hpp`
- **SYCL kernel**: `kernels/sycl/<category>/<sub-category>/<op>.hpp`
- **Engine driver**: `src/engine/<category>/<sub-category>/<op>.cpp`
- **Test**: `tests/functional/unit/test_<op>.cpp`

(Non-BLAS math/logic domains keep their kernels inline in a public
`SushiBLAS::Dispatch` helper under `include/SushiBLAS/engine/**/dispatch.hpp`
rather than under `kernels/` — see
[the architecture, section 7](architecture/ARCHITECTURE.md#7-kernels-reference-and-sycl) — but the
driver/test split is identical.)

### Reference kernel rules

- Pure C++, **zero external library dependencies** (no oneMKL, no vendor
  math library, ever).
- The mathematical ground truth: correctness is judged against this, not the
  other way around.
- Templated on the scalar type `T` to support every `Core::DataType` the op
  claims to handle.
- Must handle both `Core::Layout` values (row-major and column-major).

### SYCL kernel rules

- Takes the `sycl::queue&` it must submit to and a
  `const std::vector<sycl::event>& deps` it must `depends_on(...)`; returns
  the submitted work's `sycl::event`.
- If a dtype genuinely cannot be supported on a code path (a hardware or
  algorithmic limitation, not an oversight), throw a descriptive
  `SB_THROW_IF` rather than a silent fallback.

### Engine driver rules (`src/engine/...`)

The driver is the orchestrator between SushiRuntime's `TaskGraph` and the two
kernels. It should:

- Validate inputs with `SB_THROW_IF` — rank/shape compatibility, dtype
  support, non-null storage — before queuing anything.
- Capture raw pointers from `Tensor` objects safely for the asynchronous
  closure, and compute any auxiliary parameters (strides, offsets, batch
  sizes) up front.
- Use `Internal::execute_blas_task` (for BLAS-shaped ops) or the equivalent
  domain dispatch helper to register the task and switch on the runtime
  dtype — do not hand-write a new `add_task`/`switch(dtype)` pair when the
  existing helper already fits.
- Dispatch through `Internal::dispatch_with_mode` (`engine/task.hpp`) for the
  `FORCE_REFERENCE`/normal/`VERIFY` triad — never hand-write the triad
  per op.

### Testing standards

- **Typed tests** across every supported `Core::DataType` the op claims.
- Type-appropriate tolerances (`SB_ASSERT_TENSOR_NEAR` / `EXPECT_TENSOR_NEAR`)
  — do not compare floating-point results for bitwise equality.
- Run the operation under `FORCE_SYCL` and under `FORCE_REFERENCE` into two
  tensors and compare them, so the SYCL kernel is checked against its
  reference. `VERIFY` does not compare the two kernels.
- Cover more than one shape: square, rectangular, and vector, and (for BLAS
  ops) both memory layouts.

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
  [the architecture](architecture/ARCHITECTURE.md)** when it changes a class,
  dispatch helper or kernel design the document names. Every symbol it names
  must still exist.
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
  [§8](#8-adding-a-c-dependency).

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
docs/architecture-dispatch-section
```

### Commit headers

`type: short summary` — lowercase, imperative mood, no trailing period:

```
feat(blas): port trsm off oneMKL onto a from-scratch tiled kernel
fix(gemm): skip reading C when beta is exactly zero
docs: document the ExecutionMode dispatch triad
```

### Commit body

Bullet points, each starting with a past-tense verb:

```
- Added a from-scratch SYCL kernel for TRSM (level3) and TRSV (level2).
- Fixed a NaN-propagation bug in GEMM/SYRK when beta == 0.
- Removed the now-dead level1_internal.hpp after the level1 port.
```

Plain English, technical, objective — state the mechanism and the effect, no
marketing language.

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
compiler, hwloc, GoogleTest) into SushiBLAS's readiness picture — you do not
need to re-declare anything SushiRuntime already provisions. If SushiBLAS
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
do **not** open a public issue — follow [`SECURITY.md`](reference/SECURITY.md).

---

## 10. Questions

Open a GitHub Discussion, join the community Discord (see
[`CODE_OF_CONDUCT.md`](reference/CODE_OF_CONDUCT.md) for the invite), or email
**mustafagarip@sushisystems.io**.

Please be respectful in all project spaces — see
[`CODE_OF_CONDUCT.md`](reference/CODE_OF_CONDUCT.md).
