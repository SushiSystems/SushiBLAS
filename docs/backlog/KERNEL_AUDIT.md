# Kernel audit

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-51 | P2 | open | Land the correctness fixes of the kernel audit | the six correctness items each land with the test the audit names |  | `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md` | 2026-10-19 |
| RW-53 | P2 | open | Land the optimisations of the kernel audit | each optimisation is its own `perf` commit with its kernel's tests green |  | `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md` | 2026-10-19 |
| RW-54 | P2 | open | Land the refactors of the kernel audit | each refactor item lands with its tests unchanged in result |  | `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md` | 2026-10-19 |
| RW-92 | P2 | blocked | Measure the optimisations of RW-53 in one session | the tree of commit `f5173a7` and the tree after RW-53 have run the benchmarks the optimisations touch, in at most an hour, and each change that moves no row… | RW-53 | `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md` | 2026-10-19 |
| RW-106 | P2 | open | Find why 13 bit-equality tests fail on the Ryzen 5 7600X CPU device | the cause is named and the 13 tests pass on `opencl:cpu` of that machine, or carry the `quarantined` label with an entry in `KNOWN_ISSUES.md` |  | `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md` | 2026-10-20 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
