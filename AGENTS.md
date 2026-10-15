# Agent instructions

SushiBLAS is a C++17 tensor and BLAS library on SushiRuntime's task graph, with a Python CLI
under `cli/`. It follows the Sushi Systems skills.

Load before working here: `sushiskills`, `repository-layout`, `cpp-code-style`,
`python-code-style`, `source-comments`, `documentation`, `humanizer`, `commits`, `testing`,
`logging`, `project-tools`.

## Facts an agent needs

- Build, test and run through `sb` (`sb build`, `sb test`, `sb run`); never call cmake, ninja or
  ctest directly. `cli/README.md` describes every command.
- The library is built at C++17, as SushiRuntime is. A C++20 construct in a public header breaks
  the stack's ABI agreement.
- `Engine` is a composition root that names no operation family, an operation view is built
  from an `Engine` or a `TaskRecorder` and reaches the task graph only through the recorder, and
  a kernel knows nothing of the recorder or `Engine`. `tools/layering/check_tier_contents.py`
  and `tools/layering/check_engine_seams.py` check all three, and `docs/architecture/README.md`
  describes the layers.
- The source is one tree per module under `modules/<tier>/<module>/`. A module is declared with
  `sushiblas_add_module()` and names the modules its headers include in `DEPENDS` and the
  modules only its `source/` includes in `PRIVATE_DEPENDS`;
  `cmake/SushiBLASLayers.cmake` holds the tier order and refuses a dependency that points up.
  A header is included as `<SushiBLAS/<module>/<name>.hpp>`.
- Every operation has a SYCL kernel and a reference path, and an unsupported dtype throws. The
  BLAS, spatial and sparse families have a reference kernel beside the SYCL kernel; the functor
  families (elementwise, nonlinear, logic), the reductions and the random family run a host
  reference path. `docs/CONTRIBUTING.md`, section 5, states what a new operation carries.
- Enumerators are PascalCase and constants are `K_UPPER_SNAKE`.
  `tools/naming/sushiblas_0_1_0.tsv` lists every name the v0.1.0 sweep changed.
- `docs/README.md` indexes the manual. `docs/design/REMAINING_WORK.md` is the backlog and
  `docs/reference/KNOWN_ISSUES.md` the open defects.

## Check before reporting work as finished

```
python tools/documentation/check_docs_layout.py .
python tools/documentation/check_changelog.py .
python tools/documentation/check_source_comments.py .
python tools/documentation/check_docs_truth.py .
python tools/layering/check_layering.py .
python tools/layering/check_tier_contents.py .
python tools/layering/check_engine_seams.py .
python tools/testing/check_test_mirror.py .
python tools/versioning/check_version.py .
python -m unittest discover -s tools/tests
```
