# Command line

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-79 | P2 | blocked | Take build, test, run and clean from the shared registrar | the four commands come from sushicore (command line C1, C12) | sushicore RW-15 |  | 2026-10-19 |
| RW-82 | P2 | open | Link the colour, theme and exit code page from `cli/README.md` | `cli/README.md` links the one page sushicore writes (C14) |  |  | 2026-10-19 |
| RW-83 | P2 | open | State the default run target once | the default run target is written in one place (C18) |  |  | 2026-10-19 |
| RW-84 | P2 | open | Let `sb test --package` report a locked DLL | a locked DLL beside the consumer ends in a console error, not a traceback |  |  | 2026-10-19 |
| RW-166 | P2 | open | Never remove a build tree in a build; refuse one CMake cannot reuse | sb build keeps every tree, refuses another generator or source folder, and names sb clean |  |  | 2026-11-21 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
