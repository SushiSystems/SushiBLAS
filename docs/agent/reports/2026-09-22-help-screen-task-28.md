# Task 28: sb adopts sushicore's help screen

Branch `feature/sushicore_help_screen`, off `main`. Not pushed, merged or tagged. The checkout is left on that branch.

## Findings

- `sb` has 7 commands (`build`, `test`, `run`, `clean`, `doxygen`, `config`, `env`) on one Typer app with no sub-apps. There is no app with a single command and no callback, so every app reaches the help group. `no_args_is_help=True` stays; bare `sb` prints the grouped page and exits 2, as before.
- The CLI has no `cli/README.md`. The document that lists its commands is `docs/CLI_GUIDE.md` ("Command overview"), so the help-screen paragraph went there. The changelog is `docs/CHANGELOG.md`, and the new lines sit under `[Unreleased]`, in the dated one-line format of the existing `Build` and `Documentation` entries.
- Raw colour markup found: one, `cli/sushiblas/services/project.py:389`, `[dim]{exe}[/dim]`, now `[muted]`. None left over that the mapping did not cover.
- The plan's colour-name test cannot use `tokenize` STRING tokens on Python 3.12 and later: an f-string is split into FSTRING_START/MIDDLE/END, so `f"[dim]{exe}[/dim]"` was invisible and the test passed on the unfixed source. The test walks the `ast` for string constants instead. This is worth passing on to the `sr` and `sa` workers.

## Panels

- `Project`: `build`, `test`, `run`, `clean`. `docs/CLI_GUIDE.md` opens its overview table with "Build, test, run, and clean the C++ project".
- `Documentation`: `doxygen`. It has its own row in that table.
- `Diagnostics`: `config`, `env`. The guide and the source comment both call them diagnostics.

## Examples dropped

None dropped for naming a missing option. Each was checked against `cli.py`: `--type`, `--clean`, `-D` (`--define`), `--suite unit` (a `Suite` member), `--filter`, `--repeat`, `--sort`, `--all`, and the `--` forwarding of `run` (`allow_extra_args`). `SB_LOG_LEVEL` and `Unit_GEMMTest.*` come from `docs/CLI_GUIDE.md`. I removed the `Example: sb run sushiblas_example` line from the `run` docstring because the epilog now carries it.

## Files changed

- `cli/sushiblas/console.py`: added `current()`.
- `cli/sushiblas/cli.py`: `_help_group`, `cls=_help_group`, panels, epilogs.
- `cli/sushiblas/services/project.py`: `[dim]` to `[muted]`.
- `cli/pyproject.toml`: `sushicore>=0.4.0`.
- `cli/tests/test_help_screen.py`: new, 15 tests.
- `docs/CLI_GUIDE.md`: one paragraph after the command overview.
- `docs/CHANGELOG.md`: three lines.

## Commands

Baseline, from `D:/Projects/sushiblas/cli`, before any change:

```
$ PYTHONPATH=D:/Projects/sushicore python -m pytest tests -q -p no:cacheprovider
..............                                                           [100%]
14 passed in 0.37s
```

The new test file before the implementation. All 15 failed, the group and example ones because Typer's own screen was drawn (no `Project` heading, no `Examples`), the colour one on `services/project.py:389`:

```
15 failed in 0.31s
```

After the change:

```
$ PYTHONPATH=D:/Projects/sushicore python -m pytest tests -q -p no:cacheprovider
.............................                                            [100%]
29 passed in 0.25s
```

The baseline 14 are unchanged and 15 are added.

```
$ python -m py_compile sushiblas/cli.py sushiblas/console.py sushiblas/services/project.py tests/test_help_screen.py; echo "exit $?"
exit 0
```

In-process `CliRunner` run on a stand-in 90-column truecolour console (script in the scratchpad, `screens.py`), ANSI stripped. The root page, logo included:

```
$ sb --help (exit 0 )

     ▄▄▀▀▀▀▀▀▀▀▄      ▄█▀▀▀▀ ██  ██ ▄█▀▀▀▀ ██  ██ ██
   ▄▀▀▀▀▀▀▀▀▀██▀▀▄    ▀█▄▄▄  ██  ██ ▀█▄▄▄  ██▄▄██ ██
  ▄▀▀██▀▀▀▀▀██▀████       ██ ██  ██     ██ ██  ██ ██
  ███████████▀██████  ▀▀▀▀▀   ▀▀▀▀  ▀▀▀▀▀  ▀▀  ▀▀ ▀▀
  ██████████████████   ▄▄▄▄▄ ▄▄  ▄▄  ▄▄▄▄▄ ▄▄▄▄▄▄ ▄▄▄▄▄▄ ▄▄    ▄▄  ▄▄▄▄▄
  █████▀▀██▀▀████▀▀▀  ██     ▀█▄▄█▀ ██       ██   ██     ███▄▄███ ██
   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀     ▀▀▀█▄   ██    ▀▀▀█▄   ██   ██▀▀   ██ ██ ██  ▀▀▀█▄
     ▀▀▀▀▀▀▀▀▀▀▀      ▄▄▄▄█▀   ██   ▄▄▄▄█▀   ██   ██▄▄▄▄ ██    ██ ▄▄▄▄█▀


sb
SushiBLAS developer CLI — build, test, and run SushiBLAS.

Usage: sb [OPTIONS] COMMAND [ARGS]...

Project
  build  Configure and build the project.
  test   Run the test suite via CTest labels.
  run    Run a built executable.
  clean  Remove the build/ tree.

Documentation
  doxygen  Generate Doxygen documentation.

Diagnostics
  config  Print the resolved config and where each value came from.
  env     Print the environment cmake/ctest/run subprocesses execute under.

Options
  --help  Show this message and exit.
```

A leaf command:

```
$ sb build --help (exit 0 )
sb build
Configure and build the project.

Cache variables set with -D persist in the build tree, so dropping the flag on a later run
does not unset them; pass VAR= to clear one, or --clean to start over.

Usage: sb build [OPTIONS]

Options
  -t, --type [release|debug|relwithdebinfo]  Build type: release | debug | relwithdebinfo.
                                             [default: release]
  --clean                                    Delete the build tree before configuring.
  -D, --define VAR=VALUE                     Pass a CMake cache variable through to
                                             configure. Repeatable. Quote values
                                             containing ';', e.g. -D
                                             SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".
  --help                                     Show this message and exit.

Examples
  sb build
  sb build --type debug          Build with debug settings
  sb build --clean               Start from an empty build tree
  sb build -D SB_LOG_LEVEL=INFO  Pass a CMake cache variable through
```

`sb test --help`, `sb run --help` and `sb env --help` were run the same way and end in their Examples blocks.

## Not done

- `sb build`, `sb test` and every other command body were not run (cmake and ctest are off limits). Only `--help` was exercised.
- The real terminal, the `background = "dark"` glow and the pipx install were not exercised; the page was drawn on a stand-in console.
- No sub-app help was tested, because `sb` has no sub-app.
- `sushiblas-cli` stays at version 0.1.0; no release was cut.
