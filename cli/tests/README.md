# CLI tests

The pytest suite of the `sushiblas` package. Run it with `sb test --cli`. Every test file carries
one size marker, `small` or `medium`, and `conftest.py` fails collection when a test has none or
more than one, and gives each the timeout of its size.

## conftest.py

`sushiblas` imports `sushicore`, the shared core under every Sushi CLI, at module load. The suite
cannot be collected without it.

These tests once ran against a hand-written stub of that package. The stub cost twice. It had to
grow every time sushicore did, and a test could end up pinning the imitation and not the
contract: one of them asserted on an attribute only the fake had. sushicore ships as its own
package, so the real package is reachable, installed in the environment or sitting
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

## test_test_label.py

`sb test` keeps quarantined tests out of every selection that does not name them. These tests pin
the regexes and the marker expression each selection produces, so a stage that gates never picks
up a quarantined test by accident.
