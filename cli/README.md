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

- **`sb`**: short name.
- **`sushiblas`**: long name.

They are identical; use whichever you prefer. Every example below uses `sb`.

```bash
hub install-cli sushiblas      # always editable, against this checkout
```

This installs through **pipx** on every platform, into an isolated venv, and pipx resolves the
shared `sushicore` package from PyPI as an ordinary dependency. The install is always editable
and there is no flag to make it otherwise: a frozen copy would stop tracking `git pull`s on this
checkout without ever saying so.

To remove it later:

```bash
pipx uninstall sushiblas-cli
```

### How the CLI finds your compiler and runtime

SushiBLAS builds against the shared SushiStack SYCL toolchain (it selects no
toolchain of its own) and against a **SushiRuntime sibling checkout**. The
CLI reads both from:

- **`cli/config.toml`**: committed, shared defaults (the same for everyone).
- **`cli/config.local.toml`**: your machine's absolute paths (compiler,
  vcpkg, the SushiRuntime checkout, if it isn't the default sibling
  location). This file is gitignored, so your personal paths never get
  committed.

`sb setup` provisions the toolchain and libraries for this checkout and writes
`cli/config.local.toml`; it needs a SushiRuntime checkout beside this one. To
provision several checkouts at once, use `hub install` from a SushiStack
workspace. To see what the CLI resolved, run `sb config`.

A config file that is not valid TOML, or a `[cli] theme` that `sushicore` does
not know, stops the command with one line naming the problem and exit code 1.

## Command overview

| Command | What it's for |
|---------|----------------|
| `sb build` / `test` / `run` / `clean` | Build, test, run, and clean the C++ project |
| `sb docs` | Generate the Doxygen API reference into `build/docs/api-site/html/` |
| `sb docs bundle` | Pack the published manual and the API reference for docs.sushisystems.io |
| `sb config` | Show the resolved configuration |
| `sb env` | Show the environment your builds run under |
| `sb setup` | Provision what the build needs, then report readiness |
| `sb doctor` | Report whether this machine can build the module |
| `sb link` / `unlink` | Add this checkout to a SushiStack workspace, or remove it |

Run any command with `--help` to see its options. `sb --version` prints the
installed version of `sushiblas-cli`, and `sb --describe` prints the commands
and their options as JSON.

`sb --help` groups the commands under Project, Documentation and Diagnostics, and each
command's own help ends with examples. On a colour terminal the root screen shows the logo.
On a dark terminal add `[cli]` and `background = "dark"` to `cli/config.local.toml` to give
it a glow; `COLORFGBG` is read when the terminal sets it, and Windows Terminal does not.

## `sb build`

Configures and builds the library, examples, and test binary.

```bash
sb build                  # Release build (the default)
sb build --type debug     # Debug build
sb build --type relwithdebinfo
sb build --clean          # delete the build tree first, then build from scratch
sb build --benchmarks     # also build the benchmark programs (--no-benchmarks stops)
```

The `--type` (`-t`) option accepts `release`, `debug`, or `relwithdebinfo`.
Flags can be combined, e.g. `sb build --type debug --clean`.

There is one build tree. Building with a different `--type` removes it and
configures from an empty one, and `sb build` says so before it does. Ctrl+C
during a build, a test run or `sb run` ends the command with exit code 130.

### Passing CMake options

`--define` (`-D`) forwards a cache variable to the configure step, and is
repeatable:

```bash
sb build -D SB_ENABLE_LOGGING=OFF       # compile out every log call below ERROR
sb build -D SB_SYCL_TARGETS="spir64;nvptx64-nvidia-cuda"   # quote values with ';'
```

Every `-D` is checked for the `VAR=VALUE` form before anything else happens;
a malformed one exits 2 and leaves the build tree untouched, even with
`--clean`.

Cache variables persist in the build tree, so dropping the flag on a later run
does **not** unset them. Pass `VAR=` to clear one, or `--clean` to start over.
A change of `--type` starts over too, so pass the variables again with it.

The options SushiBLAS itself defines:

| Option | Default | What it does |
|--------|---------|--------------|
| `SB_ENABLE_LOGGING` | `ON` | Compile log calls below ERROR: TRACE and up in a debug build, DEBUG and up under `NDEBUG`. |
| `SB_BUILD_TESTS` | `ON` at the top level, `OFF` under a consumer | Build and register the test program of each module from its mirror under `tests/`. |
| `SB_BUILD_EXAMPLES` | `ON` at the top level, `OFF` under a consumer | Build the example programs under `applications/examples/`. |
| `SB_BUILD_BENCHMARKS` | `OFF` | Build the benchmark programs of the modules under `tests/`; `sb build --benchmarks` sets it. |
| `SB_SANITIZE` | *(empty)* | Sanitizers for host code, as `-fsanitize=` takes them; `sb build --sanitize` sets it. |
| `SB_NATIVE_ARCH` | `OFF` | Build for the host CPU's own instruction set. |
| `SB_SYCL_TARGETS` | *(empty)* | AOT device targets; empty means the targets SushiRuntime resolved. |
| `SB_SYCL_DEVICE_FLAGS` | *(empty)* | Extra flags for the device compilation pass. |

The level a record must reach is set at run time, not here: the `SB_LOG_LEVEL` environment
variable, as `SB_LOG_LEVEL=warn,blas_kernels=debug`, shows every kernel launch and its chosen
shape. `modules/foundation/core/README.md` describes the logger.

`--sanitize` instruments host code: `address-undefined` sets
`SB_SANITIZE=address,undefined`, `thread` sets `SB_SANITIZE=thread` and `none`
clears it. Left out, the last setting stays. The SYCL device compile is not
instrumented. SushiRuntime turns ThreadSanitizer on for its own debug and
RelWithDebInfo builds, so build `address-undefined` as a release build.

If the SushiRuntime sibling can't be found, `sb build` fails fast with a
clear message rather than letting CMake's own `add_subdirectory` error
surface. See `sb config` for what the CLI resolved as the runtime location,
and set `sushiruntime_dir` in `cli/config.local.toml` (or `SUSHIRUNTIME_DIR`
in the environment) to point it elsewhere.

## `sb test`

Runs the tests of the sizes you name through CTest. Every test carries one size
label, `small`, `medium` or `large`; [the tests README](../tests/README.md#sizes-are-labels)
says what each allows. `--label` (`-l`) picks the sizes and repeats:

```bash
sb test                          # small and medium, the default
sb test --label small
sb test --label small --label large
sb test --label quarantined      # the quarantined tests only
sb test --cli                    # the pytest tests of this CLI
sb test --package                # the out-of-tree consumer (see below)
```

A quarantined test runs only when `quarantined` is named; otherwise `sb test`
passes `-LE ^quarantined$` to ctest.

Useful options:

```bash
sb test --filter 'Gemm.*'     # ctest -R regex over 'Suite.Case' test names
sb test --repeat 5            # re-run each test up to 5x, stop on first failure
sb test --jobs 3              # run at most 3 tests at once
```

`--filter` (`-f`) matches `Suite.Case` test names. `--repeat` (`-r`) is handy
for hunting flaky tests (`ctest --repeat until-fail`). `--jobs` (`-j`) is passed
to ctest as `--parallel`; left out, ctest decides and `CTEST_PARALLEL_LEVEL`
holds.

For GoogleTest-level options CTest doesn't expose (per-case `--gtest_filter`,
shuffling, break-on-failure), run the binary directly with `sb run` and pass
flags after `--`.

### `--cli`

Runs `python -m pytest -m <expression>` in `cli/`, with the expression of the
labels given: `sb test --cli --label small` runs `(small) and not quarantined`.
`--filter`, `--repeat` and `--jobs` are ctest knobs, so they are rejected with
exit 2.

### `--package`

Runs no CTest test. It installs the package to
`build/package/prefix`, configures `tests/package/` into
`build/package/consumer` against it, builds it, stages the shared libraries
beside it and runs it: it consumes SushiBLAS the way SushiAI does, through
`find_package`, with no access to this source tree.

It is what catches three things nothing else does: a header that exists in the
tree but isn't installed, a development flag (`-Werror`, `-march=`, a sanitizer)
leaking through the exported interface onto a consumer's build, and a construct
that compiles at the library's C++17 but not at the consumer's C++20.

`--label`, `--filter`, `--repeat`, `--jobs` and `--cli` are rejected with exit 2: this
run has no CTest selection to apply them to.

Run it before publishing anything that adds a public header or touches
`modules/library/sushiblas/CMakeLists.txt`'s install rules.

## `sb bench`

Runs every benchmark program under `build/tests/`, in name order, after
`sb build --benchmarks`. A benchmark measures and does not pass or fail; the
command exits with the first non-zero code a program returned.

## `sb run`

Runs a built executable. With no target it runs `target_bin`, whose default
is the profile's `default_target` in `cli/sushiblas/config.py`; `sb config`
prints the value in effect.

```bash
sb run                                # run the default target
sb run quickstart_demo              # run a specific binary by name
sb run --sort                         # interactively pick from the list of executables
```

The target name is matched exactly first, then by substring. Anything after
`--` is forwarded straight to the program, which is how you pass GoogleTest
flags to the test binary:

```bash
sb run sushiblas_blas_tests -- --gtest_filter='Medium_GEMMTest.*'
sb run sushiblas_umbrella_tests -- --gtest_shuffle --gtest_break_on_failure
```

An option `sb run` does not know, written before `--`, is a usage error with
exit code 2. It is never taken as the target name or passed to the program.

## `sb clean` and `sb docs`

```bash
sb clean                         # remove the build/ tree
sb docs                          # generate the Doxygen API reference into build/docs/api-site/html/
sb docs bundle --release 1.2.3   # pack the documentation bundle into build/docs/bundle/
```

`sb clean` exits 1 and names the directory when the tree is still there
after the removal, for example because a running program holds a file in it.

`sb docs` is a command group that `sushicore.docs_bundle` registers. Without a
subcommand it generates the API reference and exits with Doxygen's code. `sb
docs bundle` is its one subcommand.

`sb docs` was `sb doxygen`. The old word still runs, hidden from the help, and
names the new one on a terminal; it is removed after 2027-03-25. It builds the
reference only and takes no subcommand.

Bare `sb docs` runs Doxygen against `.config/doxygen/Doxyfile`, from the repository root, and opens
the reference with `docs/reference/MAINPAGE.md`.

Output lands in `build/docs/api-site/html/`, and warnings (every public symbol
missing a doc comment) in `build/docs/api-site/doxygen-warnings.log`. Both are
gitignored; the log is where the documentation debt is read off, and
`WARN_AS_ERROR` is deliberately `NO` so a missing `@param` cannot fail a docs
build.

Scope is the installed headers under `modules/`, minus `Detail`, minus `experimental/` and
minus the four internal kernel modules.
Decision D1 put the FFT and linalg headers outside the umbrella header, and no
operation declared there has a definition, so listing them would
advertise an API that does not link. D1 is defined in
[the glossary](../docs/reference/GLOSSARY.md).

Doxygen itself is an external tool, not vendored: install it (`winget install
DimitriVanHeesch.Doxygen` on Windows, `apt-get install doxygen graphviz` on
Linux, `brew install doxygen graphviz` on macOS) or point `doxygen_exe` in
`cli/config.local.toml` at an existing binary. The command prints the right
install line if it cannot find one. `sb setup` installs one with the rest of
the build tools.

`sb docs bundle --release X.Y.Z` writes `docs-bundle-X.Y.Z.tar.gz` and its
`.sha256` file, the archive this repository publishes to docs.sushisystems.io.
`docs/publish.toml` names the manual sections that go in and the pages that stay
out. Because that file sets `api = true`, the command first runs the same
Doxygen build as bare `sb docs`, then copies the XML from `build/docs/api-site/xml/`.
`--release` is required and is three integers; leaving it out is a usage error
with exit code 2. `--out` names another output folder. A page without a
level-one heading, a page `docs/README.md` does not link, or a link to a file
that does not exist stops the command with one line and exit code 1.

## `sb config` and `sb env`: diagnostics

```bash
sb config         # print the resolved config and where each value came from

sb env            # print the environment the cmake, ctest and run subprocesses see
sb env --all      # show every variable, not just the build-relevant ones (-a)
```

Both commands come from `sushicore` and behave the same in every module CLI.

`sb config` prints the resolved SushiRuntime sibling directory, compiler, and
vcpkg root, plus which source (default / `config.toml` / `config.local.toml`
/ an `SB_*` environment variable) supplied each field. It is the first thing to
check when a build picks an unexpected compiler or can't find its runtime
dependency.

## `sb setup`, `sb doctor`, `sb link` and `sb unlink`

These four commands come from `sushicore` and behave the same in every module
CLI.

```bash
sb setup --dry-run            # show what would be installed and written
sb setup                      # install it, write cli/config.local.toml, run doctor
sb setup --toolchain oneapi   # also install a toolchain the fragments do not select
sb setup --no-gpu             # skip the toolkit for this machine's GPU
sb setup --yes                # answer yes to the LLVM download prompt

sb doctor                     # report every check
sb doctor --for build         # one group: build | test | infer | eval

sb link --workspace <path>    # record this checkout in a workspace
sb unlink --workspace <path>  # remove it again
```

`sb setup` reads three dependency fragments in order: the shared base that
ships with `sushicore`, SushiRuntime's `cli/sushistack.deps.toml`, then this
repository's. It installs what they declare under `~/.sushisystems`, or under
`SUSHISYSTEMS_HOME` when that is set. It exits 2 without installing anything
when it finds no SushiRuntime checkout; clone one beside this repository or
set `SUSHIRUNTIME_DIR`.

`sb doctor` exits non-zero when a required check fails and prints the command
that fixes it. `sb link` and `sb unlink` take the workspace from `--workspace`,
then `SUSHISTACK_HOME`, then the nearest ancestor directory that is a workspace.

## Building without the CLI

The CLI is a convenience, not a requirement: it only runs CMake and CTest
for you. If you'd rather drive CMake yourself, see **Building with CMake
directly** in the top-level [README.md](../README.md#building-with-cmake-directly).
