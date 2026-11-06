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

## command_harness.py

A command test builds the real `ProjectService` and sushicore's `CMakeDriver` over a scratch
checkout, a `RecordingRunner` and a `RecordingConsole`, wraps them in a `CommandContext` and hands
runs the command inside `command_context.using(context)`. It then reads what the service gave cmake, ctest and the
programs from `runner.commands`, and what the command printed from the console. No test replaces a
function of `sushiblas`. The `commands` fixture in `conftest.py` makes one; the doubles that stand
in for the compiler configuration, the executable index and the build environment live in the same
module. Three tests still patch the standard library, because a removal that fails and a Ctrl+C in
a child cannot be scripted on the recording runner.

## test_console_lazy.py

`console.py` built the console at import time. That called `config_dir()`, which called
`find_project_root()`, which raised `SystemExit` outside a checkout. `cli.py` imports the console
transitively, so every `sb` invocation died with "Not inside a SushiBLAS project", including
`sb --help`, which Typer can answer with no project.

Four of the five CLIs fixed this independently and sushiengine's copy never did, so `se --help`
outside a checkout failed for as long as the copies existed. The lifecycle now lives in
`sushicore.cli_console.LazyConsole`. These tests pin that this CLI is wired to it, which is the
part a shared implementation cannot check on its own.

