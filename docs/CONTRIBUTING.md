# Contributing to SushiBLAS

SushiBLAS is currently a personal project developed by Mustafa Garip. To
maintain a unified vision while the architecture is still settling, **external
pull requests are not being accepted at this stage** — but bug reports and
feature/architecture discussion are very welcome, and this document is the
reference for how the project is actually built and how a change (mine or,
eventually, yours) should look.

By contributing you agree that your contributions are licensed under the
project's **Apache License, Version 2.0** (see [`LICENSE`](../LICENSE)).

> New to the codebase? Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first — it
> explains the `Tensor`/`Storage` model, the `Engine` facade, the
> `ExecutionMode` dispatch triad, and the reference/SYCL kernel split every
> operation follows.

---

## 1. Getting set up

The fastest path from a fresh machine to a built tree is SushiStack's one-shot
installer, which installs `ss` (SushiStack owns dependency provisioning for
the whole stack) and runs `ss install`:

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

If you already have a SushiRuntime sibling checkout and its bundled SYCL
toolchain, install the CLI and build directly:

```bash
ss install-cli sushiblas  # puts `sb` / `sushiblas` on PATH
sb build                  # Release (default)
sb test                   # functional suite
```

See [`CLI_GUIDE.md`](CLI_GUIDE.md) for the full command reference and how
machine-specific toolchain paths are resolved (`cli/config.local.toml`).

---

## 2. Before you start coding

- This is a personal project; if you're not the maintainer, the productive
  path is an issue — a bug report or a design discussion — not a PR.
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
sb test --suite all
sb test --filter 'Gemm.*'         # ctest -R over 'Suite.Case' test names
```

Expectations for a mergeable change:

1. **The full test suite passes** (`sb test --suite all`).
2. **A new operation ships with a typed test.** Every test suite runs across
   every dtype the op claims to support (`float`, `double`, `half`, and the
   complex types where applicable) — see [§5](#5-adding-a-new-operation).
   `VERIFY` mode (or a manual `FORCE_SYCL` vs. `FORCE_REFERENCE` comparison) is
   how a new SYCL kernel earns confidence against its reference implementation.
3. **A fixed bug gets a regression test** in
   `tests/functional/regression/`, named for what it guards against (see
   `Regression_Sentinel.SyrkAcceptsHalfDtype` for the shape).

---

## 4. Coding style

The codebase has a deliberate, consistent style. Match the file you're editing.

### License header

Every file starts with this signature (replace `[filename]`):

```cpp
/**************************************************************************/
/* [filename]                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/
```

### C++ standard and naming

- **C++20.** Indentation is 4 spaces.
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

### Documentation

Doxygen-style comments on every public API, in plain English: `@brief` for
the one-line summary, `@param` per parameter, `@return` for the result.

### Comments and TODOs

`// TODO:` on its own line, directly **above** the code it refers to — never
on the same line as code.

```cpp
// TODO: Implement support for half-precision floats
void process_data() { ... }
```

### Error handling

`SB_THROW_IF(condition, message, ...)` for errors a caller can act on
(unsupported dtype, shape mismatch, invalid storage). Assertions are for
internal bugs, not user-facing input validation. An unsupported case
(dtype, layout, mode) always throws loudly — it never silently no-ops.

### Build-system names are prefixed with the full program name

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
[ARCHITECTURE.md §10](ARCHITECTURE.md#10-packaging)).

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
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-kernels-reference-and-sycl) — but the
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
- Dispatch through `Internal::dispatch_with_mode` for the
  `FORCE_REFERENCE`/normal/`VERIFY` triad — never hand-write the triad
  per op.

### Testing standards

- **Typed tests** across every supported `Core::DataType` the op claims.
- Type-appropriate tolerances (`SB_ASSERT_TENSOR_NEAR` / `EXPECT_TENSOR_NEAR`)
  — do not compare floating-point results for bitwise equality.
- Exercise `VERIFY` mode, or a manual `FORCE_SYCL` vs. `FORCE_REFERENCE`
  comparison, so the SYCL kernel is checked against its own reference.
- Cover more than one shape: square, rectangular, and vector, and (for BLAS
  ops) both memory layouts.

---

## 6. Documentation is part of the change

Documentation ships in the same commit as the code it describes.

- **Changelog entries start at v1.0.0.** The project has not yet been
  assigned a semantic version — it is in prototype stage. Do not add entries
  to [`CHANGELOG.md`](CHANGELOG.md) until the project cuts its v1.0.0
  release; the `[Unreleased]` section documents prototype-era milestones
  informally until then.
- **A new or changed operation updates [`ARCHITECTURE.md`](ARCHITECTURE.md)**
  when it changes a class, dispatch helper, or kernel design the document
  names — every symbol it names must still exist.
- **Any CLI change updates [`CLI_GUIDE.md`](CLI_GUIDE.md)** — the full `sb`
  command reference — and the build/test snippets here and in the README
  that invoke it.
- **A new dependency updates the dependency surface** — see
  [§8](#8-adding-a-c-dependency).

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
(the `ss` CLI). SushiBLAS's own manifest fragment is
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
`ss doctor` (confirms the entry parses) and `ss install`.

---

## 9. Reporting bugs and requesting features

Use the GitHub issue templates. A good bug report includes the platform and
toolchain, the build type, the exact `sb` command, and the smallest
reproducer you can manage. For anything that looks like a **security** issue,
do **not** open a public issue — follow [`SECURITY.md`](SECURITY.md).

---

## 10. Questions

Open a GitHub Discussion, join the community Discord (see
[`SECURITY.md`](SECURITY.md) for the current invite), or email
**mustafagarip@sushisystems.io**.

Please be respectful in all project spaces — see
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
