# CLI tests

The pytest suite of the `sushiblas` package. Run it with `python -m pytest -q` from `cli/`.

## conftest.py

`sushiblas` imports `sushicore`, the shared core under every Sushi CLI, at module load. The suite
cannot be collected without it.

These tests once ran against a hand-written stub of that package. The stub cost twice. It had to
grow every time sushicore did, and a test could end up pinning the imitation and not the
contract: one of them asserted on an attribute only the fake had. sushicore now ships inside the
SushiStack repository, so the real package is reachable, installed in the environment or sitting
next door, and the tests use it.

## test_console_lazy.py

`console.py` built the console at import time. That called `config_dir()`, which called
`find_project_root()`, which raised `SystemExit` outside a checkout. `cli.py` imports the console
transitively, so every `sb` invocation died with "Not inside a SushiBLAS project", including
`sb --help`, which Typer can answer with no project.

Four of the five CLIs fixed this independently and sushiengine's copy never did, so `se --help`
outside a checkout failed for as long as the copies existed. The lifecycle now lives in
`sushicore.cli_console.LazyConsole`. These tests pin that this CLI is wired to it, which is the
part a shared implementation cannot check on its own.

## test_device.py

`--device` is the gate a GPU run goes through, so a wrong selector runs the suite on a device
nobody asked for and reports it green. These tests pin every name of `K_DEVICE_SELECTOR`, the
`selector:` passthrough, the refusal of anything else and the rule that `--device` outranks a
selector in the environment. `test_project.py` checks that `sb test` and `sb run` hand the
resulting environment to the child.

## test_suites.py

`sb test` maps every suite to a `ctest -L` regex through `_SUITE_LABEL_REGEX`, with two
deliberate exceptions: `all`, which selects no label, and `package`, which runs no ctest. Both
are absent from the mapping, so both depend on `test()` branching before it indexes the mapping.
A missing branch is a `KeyError` at the moment the user runs the command. These tests exist to
prevent that.
