# Known issues

Defects that are recorded and not yet fixed. The estate audit of 2026-10-05 found them; its full
text, with evidence, is `docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`. Work that is planned and
is not a defect is in [the backlog](../design/REMAINING_WORK.md).

## Behaviour

| Issue | Where |
| --- | --- |
| `Storage` and `Engine` name `SushiRuntime::Memory::USMAllocator` and `IAllocatorStrategy`, which SushiRuntime removed in its commit `6e4b87b` of 2026-08-18; a `clang -fsyntax-only` run against the sibling checkout fails in both headers | `include/SushiBLAS/storage.hpp`, `include/SushiBLAS/engine.hpp` |
| `Engine::set_mode` is honoured by the BLAS and spatial families only; the elementwise, nonlinear, logic, reduction and random families never read the mode | `src/engine/math/`, `src/engine/logic/` |
| `IO::load`, `load_bin` and `load_npy` write into a tensor without first running the recorded graph, so a pending write to that tensor overwrites the loaded data | `src/io.cpp` |
| `IO::load_npy` finds the shape by substring search, skips the check when the header has no shape, and never checks dtype or `fortran_order` | `src/io.cpp` |
| Every IO path writes `num_elements` bytes from `data()`, which is wrong for a strided view | `src/io.cpp` |
| Nine random distributions reject a complex dtype inside the task, at execution, with a message that names no dtype | `src/engine/math/random/bernoulli.cpp` and its eight siblings |
| `RandomOps::shuffle` rejects INT32 and INT64 inside the task, at execution, where the architecture page says rejection happens at record time | `src/engine/math/random/shuffle.cpp` |
| `gelu.cpp` includes `<numbers>`, a C++20 header, in a library built at C++17 | `src/engine/math/nonlinear/gelu.cpp` |
| `sb test --suite package` ends in a traceback when a DLL beside the consumer is locked | `cli/sushiblas/services/project.py` |
| `sb run --sort` keeps a flag name the sibling CLIs renamed to `--pick` | `cli/sushiblas/cli.py` |

## Comments and documents

| Issue | Where |
| --- | --- |
| The enumerator comments of `ExecutionMode` say `VERIFY` runs both kernels and compares them, and name MKL; neither is true | `include/SushiBLAS/engine.hpp` |
| The source comment checker reports 1 051 findings: 312 missing file headers, 207 blocks over the ceiling, 457 comment runs, 40 separators and 35 history words, 32 of them `TODO` lines | `python tools/documentation/check_source_comments.py . --report` |
| The CLI manual says `sushicore` is published to no index, while the changelog records it taken from PyPI on 2026-09-22; which versions PyPI holds was not checked | `cli/README.md` |
| The architecture page says the non-BLAS families follow the execution-mode pattern inline; they do not read the mode | `docs/architecture/ARCHITECTURE.md`, section 3 |
| The architecture, integration and GPU validation pages use an em dash about once every 60 to 75 words and have not been through the humanizer pass | `docs/architecture/ARCHITECTURE.md`, `docs/guides/INTEGRATION.md`, `docs/design/GPU_VALIDATION.md` |
| The archived wave 1 report says it created `docs/reference/CHANGELOG.md` and that no commit was made; the entry went to the old `docs/CHANGELOG.md` and commit `e7783de` is on main | `docs/archive/agent/reports/2026-09-22-wave-1-task-7d.md` |
| The archived help screen report says its branch is unmerged and that `sb` has seven commands; the work is on main and `sb` has eleven | `docs/archive/agent/reports/2026-09-22-help-screen-task-28.md` |
| `NOTICE.md` cites `docs/CODE_OF_CONDUCT.md`, which is now `docs/reference/CODE_OF_CONDUCT.md` | `NOTICE.md`, "Adapted text" |
| Comments cite documents by their old paths: `docs/CLAUDE.md` and `CLI_GUIDE.md`, now `CLAUDE.md` and `cli/README.md`, and `docs/INTEGRATION.md`, now `docs/guides/INTEGRATION.md` | `.github/workflows/ci.yml`, `Doxyfile`, `tests/package/CMakeLists.txt` |

## Layout

| Issue | Where |
| --- | --- |
| `docs/api/` and `docs/api-site/` are not part of the documentation tree, and `mainpage.md` is not named in upper snake case; `check_docs_layout.py` reports all three | `Doxyfile`, `docs/api/mainpage.md` |
| `Doxyfile` sits in the root, where the layout forbids it, and states the version as "v1 (unreleased)" against 0.1.0 in the build files | `Doxyfile` |
| The source folder is `src/` where the layout names `source/`, and `kernels/` and `examples/` are top-level folders outside the layout | repository root |
| `tests/` has a `functional/` level over unit, integration and regression, and a `package/` folder the layout does not name | `tests/` |
| The root lacks `.clang-format`, `.editorconfig` and `.gitattributes` | repository root |
| `.gitignore` has no rule for `.pytest_cache/`, `venv/`, `.venv/`, `build-*/` or `dist/` | `.gitignore` |
| `sushi-module.toml` spells the remote in lower case, the remote itself is `SushiSystems/SushiBLAS` | `sushi-module.toml` |
| The security policy and the code of conduct are under `docs/reference/`, where GitHub does not look for them | `docs/reference/SECURITY.md`, `docs/reference/CODE_OF_CONDUCT.md` |

## Continuous integration

| Issue | Where |
| --- | --- |
| The `cli` job installs `sushicore` from a SushiStack path that no longer exists | `.github/workflows/ci.yml` |
| The pyflakes step ends in `\|\| true` and cannot fail | `.github/workflows/ci.yml` |
| The workflow says the repository has no `Doxyfile` and names the hub command `ss install` | `.github/workflows/ci.yml` |
| One workflow, Linux only, no checker job, no release trigger, and it calls cmake and ctest directly | `.github/workflows/ci.yml` |
| The CLI test fixture probes `sushistack/sushicore`, a path that no longer exists | `cli/tests/conftest.py` |

## Release state

| Issue | Where |
| --- | --- |
| No release was ever tagged, and the Unreleased section holds 88 entries | `docs/reference/CHANGELOG.md` |
| `cli/pyproject.toml` requires `sushicore>=0.7.0`; the sushicore repository is tagged up to v0.4.0 | `cli/pyproject.toml` |
| The CMake version and the CLI package version are two literals that nothing checks against each other | `CMakeLists.txt`, `cli/pyproject.toml` |
