# Plan: repository layout to the current rules

Branch `refactor/layout`. No release, no tag. Work is inline, one commit per wave.

## Decisions

- **Include form: flat.** `<SushiBLAS>/<module>/<name>.hpp`, the form the repository already
  uses. Module names are unique across the repository, so the flat form is predictable. A tiered
  form would rewrite about 550 include lines in every consumer's tree; the flat form changes none
  of the existing module paths. The `repository-layout` skill accepts either form, but a
  repository uses one for all modules.
- **Consumers.** SushiAI includes no `SushiBLAS/support` header. Its include counts by module
  (tensor 93, engine 89, core 85, blas 19, elementwise 13, random 9, reductions 5, spatial 4) do
  not change under the flat form. The removed module name is `support` only.
- **No release.** The owner has decided that nothing is published now.

## Scope

1. Remove the forbidden module `support` and split its headers into modules named by
   responsibility (wave 1).
2. Remove the forbidden folder `tests/common` and move its helpers into internal modules
   (wave 2).
3. Replace the kind folders under `tests/` (`unit`, `integration`, `regression`, `benchmark`)
   with the mirror `tests/<tier>/<module>/test_<name>.cpp`. Kinds become labels (wave 3).
4. Move module-local `tests/` folders under the same mirror (wave 3).

## Wave 1: the `support` split

| New module | Tier | Headers | Same-tier dependencies |
|---|---|---|---|
| `arithmetic` | foundation | `complex_arithmetic.hpp`, `safe_math.hpp`, `device_math.hpp`, `accumulator.hpp` | none |
| `launch` | foundation | `device_policy.hpp`, `launch_1d.hpp` | none |
| `reduction` | foundation | `reduce_1d.hpp`, `reduce_axis.hpp` | arithmetic, launch |
| `reference` | foundation | `reference_accumulator.hpp`, `serial_1d.hpp`, `serial_reduce.hpp` | reduction |
| `substitution` | foundation | `running_solution.hpp` | arithmetic |
| `orthonormal` | foundation | `orthonormal_sweep.hpp` | none |
| `sparse_assembly` | foundation | `sparse_assembly.hpp` | reference |
| `geometry` (existing) | foundation | `spatial_geometry.hpp` merged in | none |

Steps:
- `cmake/SushiBLASLayers.cmake`: add the new names to `SUSHIBLAS_MODULE_LAYERS`, remove
  `support`, and list the new same-tier edges in `SUSHIBLAS_SAME_LAYER_EDGES`. The existing
  `support core` and `geometry support` entries are rewritten to the new names.
- Each new module gets `CMakeLists.txt` with `sushiblas_add_module(NAME ... LAYER foundation
  DEPENDS ...)`, its `README.md` (rule 5), and its headers under `include/SushiBLAS/<module>/`.
- Every module that includes one of the moved headers lists the new module in its `DEPENDS`
  (rule 6). The list is computed from the include lines, not guessed.
- The `support` tests move to `tests/foundation/<new module>/`; they stay in the root tree
  until wave 3 if that is simpler.

## Waves 2 to 4

Described when wave 1 is verified. The `tests/common` split and the mirror move need the
dependency of each helper, which is read from its includes at that point.

## Verification

- Every wave: the repository checkers under `tools/` run and pass; the layering checker
  reports no upward edge.
- Build: the owner's `sr` build at the end, as he asked. No build is run before that.
- CI: the branch is pushed and a pull request opened when wave 3 is done.

## Status

- Wave 0 (this plan): written.
- Wave 1 (`9e42d03`): done.
- Wave 2 (`fb62a8a`): done.
- Wave 3: done. Every test sits in `tests/<tier>/<module>/`, each module builds one program
  through `sushiblas_add_test_program()`, and the benchmark helpers are the internal module
  `benchmark_harness`. The shared `check_test_mirror.py`, run from a scratch copy, reports no
  orphan test, no stray folder and no private include.
- Wave 4: blocked on two owner decisions. The shared `check_layering.py` exempts an umbrella
  only in a module named `umbrella`, so the 158 test includes of `<SushiBLAS/SushiBLAS.h>`
  from below the library tier are findings until `SushiBLAS.h` moves to such a module. The
  shared `check_test_mirror.py` reports 185 public headers with no test file.
