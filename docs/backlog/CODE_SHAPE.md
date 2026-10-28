# Code shape

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-66 | P2 | open | Reword the old bare names in comments and messages | no comment or failure message spells an enumerator or a constant by its old bare name; `kLarge` remains in two tests |  |  | 2026-10-19 |
| RW-67 | P2 | open | Move the recorder's identity counter into a source file | two binaries that link the static library cannot share an identity |  |  | 2026-10-19 |
| RW-68 | P2 | open | Route every runtime dtype switch through one table | `constant.cpp`, `reductions/dispatch.hpp` and the rest use one table (C1) |  |  | 2026-10-19 |
| RW-69 | P2 | open | Give reduction kernels one home | every reduction kernel lives in one module (C9) |  |  | 2026-10-19 |
| RW-70 | P2 | open | Split the level 1 kernel headers per routine | each level 1 routine has its own header (C12) |  |  | 2026-10-19 |
| RW-71 | P2 | open | Split `test_regression.cpp` per defect | each defect has its own test file and every test states its tolerance (C17) |  |  | 2026-10-19 |
| RW-72 | P2 | open | Widen the reference tests of levels 1 and 2 and TRSM | they cover column-major and other dtypes, and the remaining single-case smoke tests carry the edge cases the five edited ones carry |  |  | 2026-10-19 |
| RW-73 | P2 | open | Give the eight older spatial operations an overlap check | each refuses a written tensor that overlaps a read one, as the `window_dot` triple does |  |  | 2026-10-19 |
| RW-74 | P2 | open | Bring the record-time errors to the error table of `cpp-code-style` | an operand that breaks a rule stops the process through `SB_VERIFY` with a replaceable handler, and `UnsupportedExecution` is returned by the next… |  |  | 2026-10-19 |
| RW-75 | P2 | open | Group the includes of every source file | every file groups its includes as the contributing guide states |  |  | 2026-10-19 |
| RW-76 | P2 | open | Include each header of a header-only module first and alone in a test | a header that forgot an include fails in its own test |  |  | 2026-10-19 |
| RW-77 | P2 | open | Spell the remote in `sushi-module.toml` as GitHub does | `sushi-module.toml` names `SushiSystems/SushiBLAS` |  |  | 2026-10-19 |
| RW-78 | P2 | open | Show the dependency rule on a kernel that drops its events | a test shows a kernel that accepts `deps` and never uses it, and the engine seam check reports it |  |  | 2026-10-19 |
| RW-104 | P2 | open | Carry out SushiBLAS's share of layered execution | tasks B1 to B13 of `docs/design/LAYERED_EXECUTION/README.md` are closed |  | `docs/design/LAYERED_EXECUTION/README.md` | 2026-10-20 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
