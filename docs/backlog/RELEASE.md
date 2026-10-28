# Release

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-89 | P2 | blocked | Pin the runtime | `cmake/SushiBLASConfig.cmake.in` finds SushiRuntime 0.4, `cmake/Runtime.cmake` names the same version, a sibling checkout's version is checked, and CI builds… | external: the SushiRuntime tag that carries LAYERED_EXECUTION R1 to R10, which B13 pins |  | 2026-10-19 |
| RW-90 | P2 | open | Keep the experimental module out of the install | `experimental` is `INTERNAL`, the package test fails on an installed tree that holds that folder or a document among its headers, and `sb test --package` passes |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
