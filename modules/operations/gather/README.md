# gather

`RowIndexOps`: four operations that select by an integer index. `take_rows` copies table rows,
`add_rows` sums rows back into a table, `take_along` picks one element per row and `put_along`
writes one element per row. The index is the label itself, so no sparse matrix is built and the
size of the table sets no width limit.

It depends on `core`, `tensor`, `engine` and `operands`; its kernels are the private module
[`index_kernels`](../../kernels/index_kernels/README.md). Its headers are included as
`<SushiBLAS/gather/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `gather.hpp` | `RowIndexOps`, which records the four operations |
| `index_operands.hpp` | The record-time checks the four operations share |

`source/` holds one translation unit per operation: `take_rows.cpp`, `add_rows.cpp`,
`take_along.cpp` and `put_along.cpp`.

## `gather.hpp`

```cpp
SushiBLAS::RowIndexOps rows(recorder);
rows.take_rows(table, index, picked_rows);
rows.add_rows(picked_rows, index, table_gradient, 0.0);
```

| | `take_rows` | `add_rows` | `take_along` | `put_along` |
| --- | --- | --- | --- | --- |
| Shapes | `table [V, D]`, `index [R]`, `rows [R, D]` | `rows [R, D]`, `index [R]`, `table [V, D]` | `values [R, C]`, `index [R]`, `picked [R]` | `picked [R]`, `index [R]`, `values [R, C]` |
| Element types | Every one but UINT32 | HALF, FLOAT32, FLOAT64 | Every one but UINT32 | Every one but UINT32 |
| Index out of range | The row is zeros | The row is skipped | The element is zero | The row is zeros |
| Written | Every element of `rows` | Every element of `table` | Every element of `picked` | Every element of `values` |
| Work-items | One per element of `rows` | One per column of `table` | One per row | One per element of `values` |

The index is a rank-1 INT32 or INT64 tensor, zero-based. The range is `[0, V)` for the row
operations and `[0, C)` for the element operations, and a negative index is out of range. An index
outside it selects nothing, on both paths and in every build type; nothing is reported, because
the content of an index is not known when the task is recorded.

`add_rows` computes `table[v, d] = beta * table[v, d]` plus every `rows[r, d]` with
`index[r] == v`, added in ascending `r`. An index that repeats therefore gives the same bytes on
every run. A `beta` of exactly zero does not read `table`, and the task declares `table` as
written; any other `beta` declares it as updated. The device sums in `Accumulator<T>`, so HALF is
carried in FLOAT32 through a scratch buffer of one element per element of the table's span; the
reference sums in `ReferenceSum`. The two paths agree to the precision of the device sum, and bit
for bit when every partial sum is exact in the element type.

The matrices are rank 2 and are addressed through their own strides, so both layouts work and so
does a view into an arena, at any `storage_offset`. The index is read when the task runs, so a leaf
refilled between replays is honoured.

Every check runs when the task is recorded and throws `std::invalid_argument` naming the operation
and the operand: a rank that is not the one the table above gives, an index of another dtype, an
extent that does not match, a dtype that differs from the data operand's, a written operand whose
elements share an address, and a written operand that overlaps a read one. An element type the
device cannot compute in throws `UnsupportedExecution`. The operation names are `index.take_rows`,
`index.add_rows`, `index.take_along` and `index.put_along`.

## `index_operands.hpp`

The checks live in one header so the four sources state each rule once. They read tensors and
name no kernel type, so the header does not depend on `index_kernels`. `spanned_elements` gives the
scratch size of `add_rows`.
