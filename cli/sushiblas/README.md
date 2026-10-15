# The sushiblas package

The Python package behind `sb`. The user guide is `cli/README.md`. This file records why the
modules are shaped as they are.

## config.py

Configuration is layered, low to high:

```
built-in defaults -> config.toml -> config.local.toml -> SB_* environment variables
```

SushiBLAS is a head of the stack. It consumes the shared SushiStack SYCL toolchain and vcpkg tree
and provisions neither, and it builds against a live SushiRuntime sibling checkout (D3, layered
CMake resolution), not a vendored copy. So this module selects no toolchain. Resolving the
compiler and vcpkg out of the shared tree is the job of `sushicore.stack_config.StackConfig`.
Locating the project and layering the files is the job of `sushicore.module_config.ModuleConfig`.
What is left here belongs to SushiBLAS: its identity, and the one sibling it builds against.

### PROFILE

`PROFILE` is how SushiBLAS describes itself to the shared machinery. It is stated once and read
by config loading, the console, executable discovery and diagnostics, so they cannot disagree
about what this module is.

`SUSHIRUNTIME_DIR` carries no `SB_` prefix. It names the same checkout for every CLI that builds
against it, so this CLI must read the same variable the others do.

### ENV_OVERRIDES

The map from field to environment variable is derived from the profile's prefix. It is a module
attribute because `sb config` reports which values came from the environment and needs the
reverse map. `sushicore.diag.Diagnostics` reads it under the name `_ENV_OVERRIDES`, so the module
keeps that name as an alias of `ENV_OVERRIDES` until sushicore reads the public one.

### Config.target_bin

The default run target is `PROFILE.default_target`, and `Config.target_bin` takes its default
from there. `config.toml` does not repeat it.

### Config.sushiruntime_dir

The field names the SushiRuntime sibling checkout. Empty means `../sushiruntime` relative to the
project root, which matches the default of `SUSHIRUNTIME_DIR` in `cmake/Runtime.cmake` when
`SUSHISTACK_HOME` is not set.

## console.py

The module is a thin wrapper around `sushicore`. The theme, icon and renderer logic and its
`[cli]` config schema live there, shared with every other Sushi CLI. sushicore's README says how
to change colours.

`sushicore.cli_console.LazyConsole` builds the console on first use. Building it needs this
repository's `cli/` directory, and locating that fails outside a checkout. A console built at
import time would make every `sb` invocation abort with "Not inside a SushiBLAS project",
including `sb --help`, which Typer can serve with no project at all.

The module assigns `__getattr__` (PEP 562), so attributes resolve on first access and importing
the module never touches the filesystem or the project layout.

## env.py

A parent process cannot `call vcvars64.bat` and inherit the result. The shell is run as a child
and its environment dumped; `sushicore.build_env` holds the mechanism and the cache.

SushiBLAS selects no SYCL toolchain of its own. It builds with the intel/llvm clang++ in the
shared SushiStack dependency tree. Its environment work is therefore exactly
`sushicore.build_env.StackBuildEnv`: load MSVC on Windows and put the shared toolchain's bin and
lib on PATH. The sibling checkouts in the cache key come from `PROFILE` and are not listed again.

## services/discovery.py

The walk over the build tree is shared; see `sushicore.discovery.ExecutableIndex`. This module
builds the index and contributes only which directories to prune, and it does not spell them out either. They are the
sibling checkouts `PROFILE` already names, because a module that builds a dependency in-tree
must not offer that dependency's executables as its own.

## Services and the command layer

`cli.py` parses arguments, prints the header and the outcome, and turns a
`services.errors.ProjectError` into one error line and that error's exit code. The services
return a result or raise; each error class carries its exit code. `ProjectService` is built per
command by `services/project_factory.py`, so importing the package builds no runner, driver or
index. It still prints progress lines through the console it receives, because sushicore's
`Runner` and `CMakeDriver`, which it wraps, print through the same console.

`services/diag.py` keeps `K_DIAGNOSTICS` at module level: `register_diagnostic_commands` takes
the object when the commands are declared.

## services/suite.py

### Suite

`unit`, `regression` and `integration` are CTest labels, assigned in `cmake/SushiBLASTests.cmake` from
the prefix of the GTest suite name. `functional` is the umbrella over the three.

`package` is not a CTest label and not part of `all`. It is an out-of-tree consumer of the
installed package, which has to be installed and configured on its own before it can run.

### K_SUITE_LABEL_REGEX

Each suite maps to a label-selection regex for `ctest -L`. `functional` is an alternation over
its sub-labels and not a label of its own, because a baked-in umbrella label does not survive
`gtest_discover_tests` on current CMake; `cmake/SushiBLASTests.cmake` has the details. `all` selects
everything and passes no `-L`. `package` is absent on purpose, since it runs no CTest.

## services/project.py

### validate_defines

Every `-D` is checked before `build` reads the configuration or touches the tree, so a typo
cannot cost a removed build tree. The flags are appended after every default the configure line
sets, so an explicit `-D` is never silently outranked by one of them.

### _toolchain_args

The project configure and the package consumer configure share these flags through one function.
The consumer configure is where a missing one is hardest to diagnose: clang-cl's CMake platform
module needs a resource compiler, and vcpkg's toolchain file puts hwloc and GoogleTest where
`find_package` sees them. `CMAKE_PREFIX_PATH` is left out because the two configures need
different values; the consumer must also see the install prefix.

### _remove_tree

Removal is checked after the fact. `shutil.rmtree` with `ignore_errors` leaves a tree in place
when a running program holds one of its files, and `clean` and `build` raise
`TreeNotRemovedError` instead of reporting success.

### test

`--filter` and `--repeat` are ctest knobs, and the package suite runs no ctest. The command
refuses them with exit code 2, which is better than accepting a flag and doing nothing with it.

`all` means every CTest test, which is not every test this repository has. The command prints a
line that says so, so the name does not quietly under-deliver.

### _test_package

It is the one test of what `find_package(SushiBLAS)` hands a third party, which is why it is not
a CTest label: the consumer configures from an install prefix with no access to this tree. It
compiles the public headers at C++20 while the library builds at C++17;
`tests/package/CMakeLists.txt` says why. Both intermediates live under `build/`, so `sb clean`
reclaims them. On Windows the shared libraries are copied beside the consumer, standing in for
the deployment a real consumer does; without them the binary exits with 0xC0000135 before `main`.

The install starts from an empty prefix. A stale prefix would let a header deleted since the last
run keep satisfying an include.

The consumer configure passes `--no-warn-unused-cli`. The consumer is CXX-only and never invokes
vcpkg's own installer, so a couple of the shared toolchain flags go unread there. Silencing that
one warning keeps a clean run clean; anything else cmake has to say still comes through.
