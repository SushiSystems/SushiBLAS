# Examples

Two programs that link the library the way a user would. `applications/examples/CMakeLists.txt` builds them.

| File | What it runs |
| --- | --- |
| `example.cpp` | One 1024 x 1024 GEMM on shared-memory tensors with a column-major engine |
| `io_example.cpp` | `to_string`, `save_npy` and `load_npy` on a 2 x 3 tensor, then a complex tensor printed |

`sb run` starts `sushiblas_example` by default; see `cli/README.md`.
