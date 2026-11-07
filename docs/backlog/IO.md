# Io

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-62 | P2 | open | Save through the task graph | `save` is asynchronous |  |  | 2026-10-19 |
| RW-63 | P2 | open | Route `Shared` storage through the queue for a print and a save | a print and a save of `Shared` storage read through the queue, not the host pointer |  |  | 2026-10-19 |
| RW-64 | P2 | open | Print a rank-0 tensor as its value | a rank-0 tensor prints its value and `to_vector` reads without zero-filling first |  |  | 2026-10-19 |
| RW-65 | P2 | open | Test the four CLI services | each service under `cli/sushiblas/services/` has tests (C18) |  |  | 2026-10-19 |
| RW-148 | P3 | open | Give io a header parser over bytes so the npy and sushi header refusals run small | The dtype, shape, layout, version and padding cases of the npy and sushi headers run Small_ on byte buffers; one Medium_ test per format reads a file |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
