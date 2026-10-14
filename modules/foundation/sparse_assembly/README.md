# sparse_assembly

The host-side assembly of CSR arrays from triplets, and its input checks. Its headers are included as `<SushiBLAS/sparse_assembly/<name>.hpp>`.

It depends on [`core`](../core/README.md), [`reference`](../reference/README.md), SYCL and the standard library.

| File | Declares |
| --- | --- |
| `sparse_assembly.hpp` | The host-side assembly of CSR arrays from triplets, and its input checks |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `sparse_assembly.hpp`

Host code that knows no tensor and no engine. `Sparse::assemble_csr` turns COO triplets into the
three arrays of a CSR matrix, `HostCsr`: each (row, column) pair once, by ascending column in a
row, with the values of a repeated pair added in input order through `Kernels::ReferenceSum`.
`require_distinct_positions` checks the stored positions of a sparse vector. Both throw
`std::runtime_error`, as does the check that an index type can hold a count. `SparseBuilder` is
the caller.
