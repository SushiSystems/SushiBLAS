# io

`IO`: the host transfer of a tensor, the `.sushi`, raw binary and `.npy` readers and writers, the
text rendering, and `IoStatus`, the result of a file operation.

It depends on `core`, `tensor` and `engine`. Its headers are included as
`<SushiBLAS/io/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `io.hpp` | IO, which saves, loads and displays tensors |
| `io_status.hpp` | IoStatus, the result of a file operation of IO |

`source/` holds 1 translation unit.
