# Blend headers

`blend.hpp` declares `BlendOps`, the operations that move each point by a weighted blend of
indexed transforms. It depends on `tensor` and `engine`, and its source on `blend_kernels`. Its
header is included as `<SushiBLAS/blend/blend.hpp>`.

`BlendOps` is built from a `TaskRecorder` and reaches the task graph only through it. It is not
part of `SpatialOps`, which stays the operations over an NHWC grid; the owner ruled so on
2026-10-24, and `docs/design/LAYERED_EXECUTION/README.md` holds the plan.

## `blend.hpp`

### `BlendOps::blend_transforms`

`out[p] = sum over j of weights[p][j] * transforms[indices[p][j]] * points[p]`. Linear blend
skinning is this operation with `points` the bind-pose vertices, `transforms` the joint matrices,
and `indices` and `weights` the influences of each vertex.

| Operand | Shape | Dtype |
| --- | --- | --- |
| `points` | `[P, 3]` or `[P, 4]` | HALF, FLOAT32 or FLOAT64 |
| `transforms` | `[T, 4, 4]` or `[T, 3, 4]` | the dtype of `points` |
| `indices` | `[P, J]` | INT32 or INT64 |
| `weights` | `[P, J]` | the dtype of `points` |
| `out` | the shape of `points` | the dtype of `points` |

A transform multiplies a column vector, so row `i` of the result is `sum over c of M[i][c] * v[c]`.
A point with three values is a position, taken as `(x, y, z, 1)`, and `out` holds three values: the
translation column applies and the fourth row of a 4×4 transform is not read, so nothing is divided
by `w`. A point with four values is homogeneous and `out` holds four, `w` included: `w = 0` leaves
a direction, such as a normal, free of the translation, and `w = 1` is a position. A 3×4 transform
has the implicit last row `(0, 0, 0, 1)`, so it carries `w` through unchanged. A 4×4 transform is
linear in all four values, and a projective last row shows in the blended `w`.

The weights need not sum to one; a skinning caller normalises them, and a morph or attachment
blend may not. `J` is the same for every point. A point with fewer influences pads with weight 0.

All five operands must be dense and row-major; a view or a column-major tensor is refused. `P`, `T`
and `J` must be at least 1, and `out` may share no byte with an input. The domain is real.

**Indices.** The index values live in device memory and may not exist yet when the call is
recorded, so the call cannot read them. It checks the dtype and the shape of `indices`, and the
kernels check each value as they read it: an index outside `[0, T)` adds no term for that
influence, in both modes, and no transform is read through it. Nothing throws at run time. A caller
that wants to find such an index checks its own data before the call.

**Order and type.** Each output value is one sum over `j` in ascending order, of `weights[p][j]`
times the row product `sum over c of M[i][c] * v[c]`, with `c` ascending. The SYCL kernel forms
both in `Accumulator<T>` (HALF in FLOAT32), runs one work-item per point, a grid-stride loop over
the `Device::plan_1d` grid, and rounds once to `T` on the store, so two runs on one device give the
same bits. The reference forms them in `ReferenceSum`, in double, so the two modes agree within the
bound that `tests/operations/blend/README.md` states and not to the bit. The kernel is not tuned:
a point reads its 3 or 4 values and, per influence, a row-major matrix, and a different layout or
a tile of transforms in local memory is a change to make with a measurement.
