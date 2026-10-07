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
- `Engine` knows no individual operation, an operation class knows the task graph only through
  `engine.get_graph()`, and a kernel knows nothing of `Engine`.
  `docs/architecture/ARCHITECTURE.md` describes the layers.
- Every operation has a reference kernel and a SYCL kernel, and an unsupported dtype throws.
  `docs/CONTRIBUTING.md`, section 5, states what a new operation carries.
- `docs/README.md` indexes the manual. `docs/design/REMAINING_WORK.md` is the backlog and
  `docs/reference/KNOWN_ISSUES.md` the open defects.

## Check before reporting work as finished

```
python tools/documentation/check_docs_layout.py .
python tools/documentation/check_changelog.py .
python tools/documentation/check_source_comments.py . --rule rule_license_block
python tools/layering/check_layering.py .
python tools/versioning/check_version.py .
python -m unittest discover -s tools/tests
```

The other rules of `check_source_comments.py` do not pass yet; the known issues page has the
count.
