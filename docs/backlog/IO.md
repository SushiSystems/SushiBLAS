# Io

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-62 | P2 | open | Save through the task graph | `save` is asynchronous |  |  | 2026-10-19 |
| RW-63 | P2 | open | Route `Shared` storage through the queue for a print and a save | a print and a save of `Shared` storage read through the queue, not the host pointer |  |  | 2026-10-19 |
| RW-64 | P2 | open | Print a rank-0 tensor as its value | a rank-0 tensor prints its value and `to_vector` reads without zero-filling first |  |  | 2026-10-19 |
| RW-65 | P2 | open | Test the four CLI services | each service under `cli/sushiblas/services/` has tests (C18) |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
