# index_kernels

The reference and SYCL kernels of the row index operations: rows copied by index, rows summed into
a table, one element picked per row and one element written per row. The module is internal: its
headers are not installed. `modules/kernels/README.md` holds the notes of each kernel.

It depends on `arithmetic`, `launch`, `reference` and `substitution`. Its headers are included as
`<SushiBLAS/index_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `index.hpp` | Includes every index kernel, reference and SYCL |
| `spans.hpp` | `MatrixSpan`, the extents and strides of a rank-2 operand, and `is_in_range`, the test every kernel applies to an index |
| `reference/along.hpp` | The reference kernels of `take_along` and `put_along` |
| `reference/rows.hpp` | The reference kernels of `take_rows` and `add_rows` |
| `sycl/along.hpp` | The SYCL kernels of `take_along` and `put_along` |
| `sycl/rows.hpp` | The SYCL kernels of `take_rows` and `add_rows` |

Every kernel is a template over the element type and the index type, INT32 or INT64. The
operations that use them are in [`gather`](../../operations/gather/README.md).

## Behaviour

An index outside its range selects nothing, on both paths and in every build type: a gather writes
zero where it would have read, and `add_rows` skips the row. `is_in_range` in `spans.hpp` is the one
test, so the two sides cannot differ on it, and it compares in `int64_t`, so an INT32 index of -1
and an INT64 one behave alike. No kernel asserts on an index; a caller that wants a named error
checks on the host.

`take_rows`, `take_along` and `put_along` write each element once and add nothing, so the reference
and the SYCL kernel agree bit for bit. `add_rows` runs one work-item per column of the table. The
work-item sets each element of its column to `beta` times the old one, or to zero when `beta` is
exactly zero without reading the table, then walks every row `r` upwards and adds `rows[r, d]` to
the element `index[r]` names. An element has one writer and receives its terms in ascending `r`,
so a repeated index gives the same bytes on every run. The device keeps the sum in
`Accumulator<T>` through `RunningSolution`, which needs a workspace of one `Accumulator<T>` per
element of the table's span for HALF only, and rounds to `T` on each store. The reference sums
each element in a `ReferenceSum`, in the same order, so the two paths agree to the precision of
the device sum. The launch is as wide as the table has columns, so a table of few columns uses
little of the device.
