# Manual read against the code

**Status:** Closed — 2026-10-09.

Nine agents read the manual and the module READMEs, sentence by sentence, against the code, and
corrected in place what the code made false. Each wrote only to its own files. Nothing was built
or run; the Python checkers and the CLI tests ran at the end.

## What was read

| Agent | Files | Sentences changed |
| --- | --- | --- |
| unit-tests | `tests/functional/unit/README.md` | 16 |
| support | `include/SushiBLAS/support/README.md` | 21 |
| kernels-root | `kernels/README.md` | 18 |
| engine-math | `include/SushiBLAS/engine/math/README.md`, `engine/logic/README.md` | 15 |
| src-and-top | `src/README.md`, `include/SushiBLAS/README.md` | 6 |
| engine | `include/SushiBLAS/engine/README.md` and the READMEs of `blas/`, `spatial/`, `sparse/` | 19 |
| kernel-headers-core-tests | `include/SushiBLAS/kernels/README.md`, `core/`, `experimental/`, and the READMEs of `tests/functional/regression/`, `integration/`, `tests/common/`, `package/`, `benchmark/`, `applications/examples/` | 23 |
| architecture | the twelve pages of `docs/architecture/`, `docs/CONTRIBUTING.md` | 12, and one paragraph deleted |
| manual-cli-tools | the root `README.md`, `docs/guides/`, `GLOSSARY.md`, `MAINPAGE.md`, `PUBLIC_SEAM.md`, the READMEs of `tests/`, `cli/`, `tools/` and `.config/` | 20 |

The counts are the rows of each agent's change table. The tables, with the line of code behind
each change, are in the transcripts of the session and are not copied here.

## What the changes were

- Enumerators and constants in their spelling before the naming sweep: `FORCE_REFERENCE`,
  `ROW_MAJOR`, `REAL`.
- Behaviour the code no longer has. The largest: the pages said the tiled SYRK and SYR2K throw
  `std::runtime_error` at execute time on a device that cannot host their tile, and the record
  call refuses first with `UnsupportedExecution`.
- Counts, sizes and types: which accumulation type a HALF sum uses, which operations take an
  integer tensor, how a batched BLAS driver steps between matrices.
- Sentences that said the tree had not been built or run.

## Found outside an agent's files, and corrected by the orchestrator

Comment-only changes in `tensor.hpp`, `logic/dispatch.hpp`, `reductions/dispatch.hpp`,
`device_policy.hpp`, `test_gemm.cpp` and `test_triangular_parallel.cpp`, and one sentence of `tests/common/README.md`.

## Not done

- The exception messages and about 80 comments spelled the modes `FORCE_REFERENCE` and
  `FORCE_SYCL`. The owner chose the enumerator spelling the same day; the change is in the
  tree and waits for a build.
- Every agent lists sentences it could not check from the tree: measurements, statements of
  history, claims about SushiAI, SushiRuntime, SushiStack, PyPI or a device. They were left as
  they stood.
- The architecture agent says it spot-checked some sentences and did not trace each to a line.

## Rule violation reported

One agent wrote two small text files to the system temp directory and not to the scratchpad, and
deleted them in the same command.
