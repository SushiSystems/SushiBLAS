# tools

Reference implementations of the checkers every Sushi Systems repository carries, and the
version checker of this one. A repository copies this folder whole; a rule is fixed here first
and copied out.

| Path | Checks | Takes |
| --- | --- | --- |
| `common/checker.py` | Nothing itself; runs a checker's rule table and prints findings | |
| `documentation/check_source_comments.py` | C++, GLSL, TypeScript and Python: license block, file header, block ceiling, `//` and `#` runs, separators, history words, `@date` | Files or folders |
| `documentation/check_docs_layout.py` | `docs/` entries, required documents, document names, work folders, status lines, design ceiling, links under `docs/`, reachability from `docs/README.md`, READMEs under `modules/<tier>/`, archive candidates | Repository root |
| `documentation/check_changelog.py` | Headings and entry shape, releases kept live, length, one sentence, nesting, cited places | Repository root |
| `layering/check_layering.py` | Declared tiers, upward includes, reaches into another module's `source/`; `support` headers including Engine, `engine/`, `modules/kernels/` or the tensor header; kernels including Engine, `engine/` or the tensor header; `engine.hpp` including an operation family; a dispatch seam including `engine.hpp`; operation sources calling an Engine member; a view header calling an Engine member other than `recorder` and `rng`; a graph insertion (a recording verb of `SushiRuntime::Recorder`) outside `TaskRecorder`; a task body that does not order its device work behind the events it is given. Entries in `K_KNOWN_VIOLATIONS` print as notes, a stale entry fails | Repository root |
| `versioning/check_version.py` | `version.hpp`, `.config/doxygen/Doxyfile`, `cli/pyproject.toml`, `sushi-module.toml`, `README.md` and `docs/reference/CHANGELOG.md` against `project()` in the root `CMakeLists.txt` | Repository root |
| `naming/sushiblas_0_1_0.tsv` | Nothing; lists the 94 renames of the v0.1.0 naming sweep (kind, old name, new name) for a consumer's rename tool | None |
| `licensing/write_license_block.py` | Nothing; writes the license block of `source-comments` into every tracked source file, or lists the files that lack it with `--report` | Repository root, `--project` |
| `tests/` | Unit tests for the checker rules, one file per checker except `check_changelog.py`: `test_check_docs_layout.py`, `test_check_layering.py`, `test_check_source_comments.py`, `test_check_version.py`, `test_write_license_block.py`; run `python -m unittest discover -s tools/tests` | |

`check_docs_layout.py` reads inline Markdown links, with or without a title, wrapped over a
line or in angle brackets. It does not read reference-style links or a link around an image.

`write_license_block.py` is the one tool here that writes files. It takes its licence lines
from `check_source_comments.py`, covers `.in` templates and `.inc` fragments by the file they become, `--closed` for a closed repository, `--skip GLOB` for generated
files and `--upstream PATH=LINE;LINE` for a ported file; a later run without it keeps those lines. A `--report` run that prints nothing
proves a repository's headers are in place.

Every checker takes `--report` and `--rule NAME`, exits 1 on findings, 0 when clean or
reporting, 2 on a bad path. Python 3.11, standard library only.

## Per-repository settings

`K_` constants at the top of each checker. The tier rules of `check_layering.py` run only under
a `modules/` folder, and report an empty `K_TIER_ORDER` until it lists the repository's tiers,
lowest first. `K_COPYRIGHT_LINE` and `K_LICENSE_LINES` in `check_source_comments.py` hold the
repository's licence lines; a closed repository sets `K_CLOSED` to `True`, which makes
`K_LICENSE_LINES` `("All rights reserved. No licence is granted.",)`.
