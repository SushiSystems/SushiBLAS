# reference

The reference path behind `ForceReference`: the wide type and running sum a reference kernel accumulates in, and the serial host loops and folds that mirror the launch and reduction shapes. Its headers are included as `<SushiBLAS/reference/<name>.hpp>`.

It depends on [`core`](../core/README.md), [`reduction`](../reduction/README.md), SYCL and the standard library.

| File | Declares |
| --- | --- |
| `reference_accumulator.hpp` | `ReferenceAccumulatorOf` and `ReferenceSum`, the type and the running sum of a reference kernel |
| `serial_1d.hpp` | The serial host loops that are the reference path of those three launch shapes |
| `serial_reduce.hpp` | The serial host folds that are the reference path of both reductions |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `reference_accumulator.hpp`

### `ReferenceAccumulatorOf`

A reference kernel sums a real element type in `double` and a complex one in
`std::complex<double>`, whatever the element type. The reference is the specification the SYCL
kernel is compared against, so its sum is wider than the one under test wherever a wider type
exists.

### `ReferenceSum`

The running sum of a reference kernel. For HALF, FLOAT32 and COMPLEX32 the wide type already
holds more bits than the element, and the sum is a plain `double` sum. For FLOAT64 and COMPLEX64
the wide type is the element type itself, so `ReferenceSum` keeps the rounding error of every
addition with TwoSum and adds it back when the sum is read. The two differences TwoSum takes are
exact only under strict IEEE arithmetic, which the library builds with.

## `serial_1d.hpp`

`Serial::map`, `Serial::zip_inplace` and `Serial::indexed` are the host loops behind
`ForceReference` for every kernel that launches through `launch_1d.hpp`. Each takes the functor
its `Launch` sibling would launch and calls it once per element, in index order, on one thread.
There is no second implementation of an operation to keep in step: the reference path and the
SYCL path differ only in who calls the functor. The functor contract is the device one, so a
functor that runs on the device runs here.

## `serial_reduce.hpp`

The host folds behind `ForceReference` for `reduce`, `reduce_indexed` and `reduce_axis`:
`Serial::reduce`, `Serial::reduce_indexed` and `Serial::reduce_axis`, each one ascending walk.

### `Serial::Fold`

Folds on the host in the accumulator the SYCL path uses, in the type the SYCL path folds in
(`Reduce::AccumulationOf<A>`). The specialisation for `Reduce::CompensatedSum` is the exception:
a sum is folded in `Kernels::ReferenceSum`, that is, in `double` whatever the element type. So
`finalize` is handed a `double` on the host path of a sum, and must return its value in the type
it is handed; `Dispatch::Detail::require_finalize_keeps_width` refuses at compile time one that
does not. The result is converted to the stored type once, on the store.
