# blas_test_oracle

The values a BLAS test measures a kernel against. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`. Its headers are included as `<SushiBLAS/blas_test_oracle/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `blas_truth.hpp` | The double-precision truth a BLAS parity test measures both kernels against |
| `blas_flags.hpp` | The mappings from a test's bool parameters to the BLAS flag enumerations |
| `blas_structured.hpp` | The conversions from a dense test matrix to CBLAS band and packed storage |
| `dense_oracle.hpp` | The plain dense loops a kernel test takes its expected values from: products, triangles, bands and rank updates |

## blas_flags.hpp

The BLAS suites are parameterised over `bool` values (transposed or not, upper or lower), and the
public API takes `Core::Transpose`, `Core::Uplo`, `Core::Diag` and `Core::Side`. The four
functions here map one to the other: `true` is `Trans`, `Upper`, `Unit` and `Left`. A test names
the flag it means at the call site and keeps its parameter list.

## blas_truth.hpp

The reference kernel sums in `double` and the SYCL kernel in the element type, FLOAT32 for HALF, so the two do not
agree bit for bit on a rounded sum, and a comparison of one against the other cannot say which is
off. Both are measured against a third value instead.

`TruthSum` adds products of doubles and keeps the rounding error of each product and each
addition, so its sum is what twice the working precision would give. `value()` returns the sum
together with the magnitude behind it, the sum of the absolute values of the products.
`rounding_bound(roundings, unit, magnitude)` is the largest error that many rounded operations
can leave on a sum of that magnitude, and `unit_roundoff<Real>()` is the unit of the element
type. A test asserts that each kernel lies within the bound of the truth.

## blas_structured.hpp

Builds the operands of the band and packed tests from a dense matrix given row by row:
`in_layout` orders it for a layout, `to_band` and `to_triangle_band` lay it out as CBLAS band
storage, and `to_packed` as CBLAS packed storage of one triangle. The expected result of a band or
packed routine is then the result of its dense sibling on the same matrix.
