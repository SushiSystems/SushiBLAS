# Signal headers

`signal.hpp` declares `SignalOps`, the operations that act on each row of a signal on its own. A
row is the last dimension of a tensor and the leading dimensions batch rows. It depends on
`tensor` and `engine`, and its source on `signal_kernels`. Its header is included as
`<SushiBLAS/signal/signal.hpp>`.

`SignalOps` is built from a `TaskRecorder` and reaches the task graph only through it. The FFT
is `TransformsOps` of [the experimental module](../experimental/README.md); the two share no
name and no header.

## `signal.hpp`

### `SignalOps::convolve_1d`

`out[n] = sum over k of signal[n + K - 1 - k] * taps[k]`, in valid mode: a row of `L` samples and
`K` taps give `L - K + 1` samples, and the taps are applied in reverse, so this is a convolution and
not a correlation. `signal` has rank 1 or more and its last dimension is the row; the leading
dimensions are batched rows, and `out` has the same leading dimensions. `taps` is rank 1 with
`1 <= K <= L`. A caller that carries a filter's history prepends the last `K - 1` samples of the
previous block to the row, which is how SushiEngine's audio FIR uses it.

**Options.** A call may end with an `OperationOptions`, the type `IndexOps::for_each` and `BlendOps::blend_transforms` take: the
name the node reports under and a `SushiRuntime::Determinism`. A call with none keeps the name
`signal.convolve_1d` and `Unstated`. An empty name, or a class outside the enum, is refused at
record time. The family includes `<SushiBLAS/engine/operation_options.hpp>`.

All three operands must be dense and row-major, so a row is contiguous; a view or a column-major
tensor of rank 2 or more is refused. `out` may share no byte with `signal` or `taps`. The domain is
real: HALF, FLOAT32 and FLOAT64.

Each output sample is one sum of `K` products in ascending `k` order. The SYCL kernel forms it in
`Accumulator<T>` and runs one work-item per output sample, a grid-stride loop over the
`Device::plan_1d` grid, so two runs on one device give the same bits. The reference forms it in
`ReferenceSum`, in double, as the other reductions do, so the two modes agree within the bound that
`tests/operations/signal/README.md` states and not to the bit. The kernel is not tiled: a filter
of tens of taps rereads the same few cache lines, and tiling is a change to make with a
measurement.
