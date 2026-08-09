# The SushiBLAS CLI

SushiBLAS ships with a small command-line tool that drives everything you do
day to day: building the C++ library, running its tests, launching the
example programs, and generating documentation. It is a thin wrapper around
CMake and CTest that reads your machine-specific toolchain paths from a
config file so you don't have to retype long compiler flags.

This guide explains every command in plain English. If you just want to get
the project compiling for the first time, the top-level [README.md](../README.md)
walks through setup.

## Installing the CLI

The CLI is a Python package that lives in the `cli/` folder. Install it once
and it puts two commands on your `PATH`:

- **`sb`** — short name.
- **`sushiblas`** — long name.

They are identical; use whichever you prefer. Every example below uses `sb`.

```bash
ss install-cli sushiblas      # always editable, against this checkout
```

This installs through **pipx** on every platform, into an isolated venv, then
injects the shared `sushicli` package — which is published to no index, so pipx
cannot resolve it as an ordinary dependency. The install is always editable and
there is no flag to make it otherwise: a frozen copy would stop tracking
`git pull`s on this checkout without ever saying so. You can also install the
package directly:

```bash
pipx install ./cli               # or:  pip install ./cli   (inside a venv/conda env)
```

To remove it later:

```bash
pipx uninstall sushiblas-cli
```

### How the CLI finds your compiler and runtime

SushiBLAS builds against the shared SushiStack SYCL toolchain — it selects no
toolchain of its own — and against a **SushiRuntime sibling checkout**. The
CLI reads both from:

- **`cli/config.toml`** — committed, shared defaults (the same for everyone).
- **`cli/config.local.toml`** — your machine's absolute paths (compiler,
  vcpkg, the SushiRuntime checkout, if it isn't the default sibling
  location). This file is gitignored, so your personal paths never get
  committed.

In the common case — a sibling `../sushiruntime` checkout with its bundled
toolchain already provisioned by `ss install`, or a SushiStack workspace —
no local config is needed at all. If you're ever unsure what the CLI actually
resolved, run `sb config`.

## Command overview

| Command | What it's for |
|---------|----------------|
| `sb build` / `test` / `run` / `clean` | Build, test, run, and clean the C++ project |
| `sb doxygen` | Generate the Doxygen API reference into `docs/api-site/html/` |
| `sb config` | Show the resolved configuration |
| `sb env` | Show the environment your builds run under |

Run any command with `--help` to see its options.

## `sb build`

Configures (if needed) and builds the library, examples, and test binary.

```bash
sb build                  # Release build (the default)
sb build --type debug     # Debug build
sb build --type relwithdebinfo
sb build --clean          # delete the build tree first, then build from scratch
```

The `--type` (`-t`) option accepts `release`, `debug`, or `relwithdebinfo`.
Flags can be combined, e.g. `sb build --type debug --clean`.

### Passing CMake options

`--define` (`-D`) forwards a cache variable to the configure step, and is
repeatable:

```bash
sb build -D SB_LOG_LEVEL=INFO           # raise the compiled-in log level
sb build -D SB_SYCL_TARGETS="spir64;nvptx64-nvidia-cuda"   # quote values with ';'
```

Cache variables persist in the build tree, so dropping the flag on a later run
does **not** unset them — pass `VAR=` to clear one, or `--clean` to start over.

The options SushiBLAS itself defines:

| Option | Default | What it does |
|--------|---------|--------------|
| `SB_LOG_LEVEL` | `auto` | Compiled-in log level: `auto`, `OFF`, `ERR`, `WARN`, `INFO`, `DEBUG`. `auto` means WARN for a release build and DEBUG for a debug build. |
| `SB_NATIVE_ARCH` | `OFF` | Build for the host CPU's own instruction set. |
| `SB_SYCL_TARGETS` | *(empty)* | AOT device targets; empty means SPIR-V/JIT only. |
| `SB_SYCL_DEVICE_FLAGS` | *(empty)* | Extra flags for the device compilation pass. |

`SB_LOG_LEVEL` is the one to reach for when debugging: a release build logs at
WARN, so `-D SB_LOG_LEVEL=INFO` turns the per-kernel-launch trace back on
without editing `include/SushiBLAS/core/logger.hpp`. Put it back with
`-D SB_LOG_LEVEL=auto`. Note that the level is a **public** compile definition —
it lives in an `inline constexpr` variable in a header every consumer compiles,
so the library and everything linking it must agree on it.

If the SushiRuntime sibling can't be found, `sb build` fails fast with a
clear message rather than letting CMake's own `add_subdirectory` error
surface — see `sb config` for what the CLI resolved as the runtime location,
and set `sushiruntime_dir` in `cli/config.local.toml` (or `SUSHIRUNTIME_DIR`
in the environment) to point it elsewhere.

## `sb test`

Runs the test suite through CTest. Tests are grouped by label, and you pick a
group with `--suite` (`-s`):

```bash
sb test                          # functional (unit+integration+regression) — the default
sb test --suite unit
sb test --suite integration
sb test --suite regression
sb test --suite all              # every CTest test
sb test --suite package          # the out-of-tree consumer (see below)
```

Useful options:

```bash
sb test --filter 'Gemm.*'     # ctest -R regex over 'Suite.Case' test names
sb test --repeat 5            # re-run each test up to 5x, stop on first failure
```

`--filter` (`-f`) matches `Suite.Case` test names (the `Unit_`/`Integration_`/
`Regression_`-prefixed GTest suite names CTest discovers). `--repeat` (`-r`)
is handy for hunting flaky tests (`ctest --repeat until-fail`).

For GoogleTest-level options CTest doesn't expose (per-case `--gtest_filter`,
shuffling, break-on-failure), run the binary directly with `sb run` and pass
flags after `--`.

### `--suite package`

The one suite that isn't a CTest label. It installs the package to
`build/package/prefix`, configures `tests/package/` into
`build/package/consumer` against it, builds it, stages the shared libraries
beside it and runs it — i.e. it consumes SushiBLAS the way SushiAI does, through
`find_package`, with no access to this source tree.

It is what catches three things nothing else does: a header that exists in the
tree but isn't installed, a development flag (`-Werror`, `-march=`, a sanitizer)
leaking through the exported interface onto a consumer's build, and a construct
that compiles at the library's C++17 but not at the consumer's C++20.

Two consequences worth knowing:

- **It is not included in `--suite all`.** A full install plus a second
  configure and build costs minutes and is usually not what you want mid-loop.
  `--suite all` prints a line saying it was skipped, so the gap is announced
  rather than silent.
- **`--filter` and `--repeat` are rejected with exit 2.** Both are CTest knobs
  and this suite runs no CTest; accepting them and quietly ignoring them would
  be worse than refusing.

Run it before publishing anything that adds a public header or touches
`src/CMakeLists.txt`'s install rules.

## `sb run`

Runs a built executable. With no target it runs the configured default
(`sushiblas_example`).

```bash
sb run                                # run the default target
sb run sushiblas_example              # run a specific binary by name
sb run --sort                         # interactively pick from the list of executables
```

The target name is matched exactly first, then by substring. Anything after
`--` is forwarded straight to the program — this is how you pass GoogleTest
flags to the test binary:

```bash
sb run sushiblas_tests -- --gtest_filter='Unit_GEMMTest.*'
sb run sushiblas_tests -- --gtest_shuffle --gtest_break_on_failure
```

## `sb clean` and `sb doxygen`

```bash
sb clean       # remove the build/ tree
sb doxygen     # generate the Doxygen API reference into docs/api-site/html/
```

It runs Doxygen against the repo-root `Doxyfile`, which is **committed** — for
a while it was not, and `.gitignore` listed it, so the command reported
`Doxyfile not found at repo root.` and exited 1 in every fresh checkout.
sushiruntime commits its own for the same reason.

Output lands in `docs/api-site/html/`, and warnings — every public symbol
missing a doc comment — in `docs/api-site/doxygen-warnings.log`. Both are
gitignored; the log is where the documentation debt is read off, and
`WARN_AS_ERROR` is deliberately `NO` so a missing `@param` cannot fail a docs
build.

Scope is `include/SushiBLAS` minus `Detail` and minus `experimental/`. The
quarantine is deliberate: decision D1 put the FFT and linalg headers outside
the umbrella header and not every declared op there has a definition, so
listing them would advertise an API that does not link.

Doxygen itself is an external tool, not vendored: install it (`winget install
DimitriVanHeesch.Doxygen` on Windows, `apt-get install doxygen graphviz` on
Linux, `brew install doxygen graphviz` on macOS) or point `doxygen_exe` in
`cli/config.local.toml` at an existing binary. The command prints the right
install line if it cannot find one. The stack's provisioned toolchain ships
one, so `ss install` is usually enough.

## `sb config` and `sb env` — diagnostics

```bash
sb config         # print the resolved config and where each value came from

sb env            # print the environment cmake/ctest/run subprocesses run under
sb env --all      # show every variable, not just the build-relevant ones
```

`sb config` prints the resolved SushiRuntime sibling directory, compiler, and
vcpkg root, plus which source (default / `config.toml` / `config.local.toml`
/ an `SB_*` environment variable) supplied each field — the first thing to
check when a build picks an unexpected compiler or can't find its runtime
dependency.

## Building without the CLI

The CLI is a convenience, not a requirement — it only runs CMake and CTest
for you. If you'd rather drive CMake yourself, see **Building with CMake
directly** in the top-level [README.md](../README.md#building-with-cmake-directly).
