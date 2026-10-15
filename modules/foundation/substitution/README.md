# substitution

`RunningSolution`, which carries the solved elements of a triangular substitution unrounded between steps. Its headers are included as `<SushiBLAS/substitution/<name>.hpp>`.

It depends on [`arithmetic`](../arithmetic/README.md), SYCL and the standard library.

| File | Declares |
| --- | --- |
| `running_solution.hpp` | `RunningSolution`, which carries a substitution's solution unrounded |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `running_solution.hpp`

### `RunningSolution`

A triangular substitution reads the elements it has already solved. `RunningSolution` carries
them in `Accumulator<T>` between steps and stores each one rounded to `T` in the right-hand side
as well, so a later row reads the unrounded value. For every type but HALF `Accumulator<T>` is
`T` and the solution lives in the right-hand side itself. For HALF it needs a workspace of one
`float` per element, which `K_NEEDS_WORKSPACE` reports and
`Internal::running_solution_bytes` sizes at record time.
