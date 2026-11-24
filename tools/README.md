# tools

This repository's side of the shared checkers, which run from the sushicore package with
`python -m sushicore.tools check .`:

| Path | Holds |
| --- | --- |
| `settings.py` | The settings the checkers read for this repository |
| `known_findings.toml` | The findings the checkers accept for now, each with its backlog ID |
| `hooks/pre-commit`, `hooks/commit-msg` | The git hooks; `python -m sushicore.tools install-hooks .` points git at them |

The repository's own tools:

| Path | Does |
| --- | --- |
| `layering/check_tier_contents.py` | SushiBLAS's own: headers of the foundation modules other than `core` and `graph_primitives` including Engine, `engine/`, `modules/kernels/` or the tensor header; kernels including Engine, `engine/` or the tensor header; `engine.hpp` including an operation family; operation sources calling an Engine member; a graph insertion (a recording verb of `SushiRuntime::Recorder`) outside `TaskRecorder`. Entries in `K_KNOWN_VIOLATIONS` print as notes, a stale entry fails | Repository root |
| `layering/check_engine_seams.py` | SushiBLAS's own: a dispatch seam including `engine.hpp`; a view header calling an Engine member other than `recorder` and `rng`; a task body that does not order its device work behind the events it is given. The test-only and benchmark-only modules, INTERNAL and added only under `SB_BUILD_TESTS` or `SB_BUILD_BENCHMARKS`, are exempt | Repository root |
| `styling/check_raw_colours.py` | SushiBLAS's own: a string literal of `cli/sushiblas`, f-string parts included, that opens or closes a raw colour tag such as `[green]` instead of a theme style | Repository root |
| `versioning/check_version.py` | `version.hpp`, `.config/doxygen/Doxyfile`, `cli/pyproject.toml`, `sushi-module.toml`, `README.md` and `docs/reference/CHANGELOG.md` against `project()` in the root `CMakeLists.txt` | Repository root |
| `naming/sushiblas_0_1_0.tsv` | Nothing; lists the 94 renames of the v0.1.0 naming sweep (kind, old name, new name) for a consumer's rename tool | None |
