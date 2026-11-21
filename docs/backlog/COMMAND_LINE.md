# Command line

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-79 | P2 | blocked | Take build, test, run and clean from the shared registrar | the four commands come from sushicore (command line C1, C12) | sushicore RW-15 |  | 2026-10-19 |
| RW-82 | P2 | open | Link the colour, theme and exit code page from `cli/README.md` | `cli/README.md` links the one page sushicore writes (C14) |  |  | 2026-10-19 |
| RW-83 | P2 | open | State the default run target once | the default run target is written in one place (C18) |  |  | 2026-10-19 |
| RW-84 | P2 | open | Let `sb test --package` report a locked DLL | a locked DLL beside the consumer ends in a console error, not a traceback |  |  | 2026-10-19 |
| RW-165 | P1 | open | Use the shared clean command and drop build --clean | sb clean says what it would remove and removes trees only with --yes, and sb build has no --clean |  | `cli/sushiblas/cli.py` | 2026-11-21 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
