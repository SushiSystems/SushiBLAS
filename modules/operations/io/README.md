# io

`IO`: the host transfer of a tensor, the `.sushi`, raw binary and `.npy` readers and writers, the
text rendering, and `IOStatus`, the result of a file operation.

It depends on `core`, `tensor` and `engine`. Its headers are included as
`<SushiBLAS/io/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `file_header.hpp` | The readers and checks of the .npy and .sushi headers, which work on bytes and text and touch no file |
| `file_remover.hpp` | `IOFiles::FileRemover`, the port a failed save deletes its partial file through, and `DiskFileRemover` |
| `io.hpp` | IO, which saves, loads and displays tensors |
| `io_status.hpp` | IOStatus, the result of a file operation of IO |
| `partial_file.hpp` | `IOFiles::remove_partial_file`, which deletes the file a failed save left through a `FileRemover` and warns when it cannot |

`source/` holds 4 translation units: `file_header.cpp`, `io.cpp` (the transfers), `io_file_stream.cpp` (the checked stream reads and writes the transfers share) and `print.cpp` (the text rendering). `io_file_stream.hpp` and `io_tensor_access.hpp` are private to them.
