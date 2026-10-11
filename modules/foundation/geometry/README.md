# geometry

Small fixed-size geometric value types and the functions on them: a 3-component vector, a
rotation quaternion and a column-major 4x4 matrix. The module is header-only and lives in
`SushiBLAS::Geometry`. Its headers are included as `<SushiBLAS/geometry/<name>.hpp>`.

A geometry header includes the standard library, `<SushiBLAS/support/device_math.hpp>` and other
geometry headers, and nothing else. `device_math.hpp` includes SYCL only when the translation unit
is already a SYCL one, so every header here compiles with a plain C++17 compiler that has neither
SYCL nor SushiRuntime on its include path.

| File | Declares |
| --- | --- |
| `vector3.hpp` | `Vector3T<T>` with `+`, `-` and `* scalar`, and `dot`, `cross`, `length`, `normalize` |
| `quaternion.hpp` | `QuaternionT<T>`, `quaternion_axis_angle`, `mul`, `conjugate`, `rotate`, `normalize`, `dot` |
| `matrix4.hpp` | `Matrix4T<T>` and its product `mul` |
| `interpolation.hpp` | `lerp` for scalars and vectors, `nlerp` and `slerp` for rotations |
| `transform.hpp` | `translation`, `scaling`, `perspective`, `look_at`, `matrix4_from_quaternion`, `compose_transform`, `quaternion_from_matrix`, `affine_inverse`, `decompose_transform` |

The types and functions came from SushiEngine's `blas_placeholder.hpp`, which held them until this
library owned them. The arithmetic is the same operations in the same order, so a consumer that
moves from the stand-in to this module gets the same bits.

## The element type

Every type and function is a template on the element type `T`, and the module names no precision
of its own: there is no alias that picks `double`. A consumer declares the aliases it wants, for
example `using Vector3 = SushiBLAS::Geometry::Vector3T<double>;`. One build can then hold a
`Vector3T<double>` for camera and transform maths beside a `Vector3T<float>` for a solver that runs
in a narrower precision.

The functions that take only scalars, `perspective` and the scalar `lerp`, deduce `T` from every
argument. A call that mixes `float` and `double` arguments does not compile; the caller converts
first.

## What a kernel can rely on

`Vector3T`, `QuaternionT` and `Matrix4T` are trivially copyable and standard layout, and each holds
its elements and no padding: 3, 4 and 16 times `sizeof(T)`. A SYCL kernel captures one by value,
and a buffer of them has the layout of a buffer of `T`. A `static_assert` beside each type states
this for `float` and `double`. The default member initialisers make a default-constructed vector
zero, a quaternion the identity rotation and a matrix the identity; they do not affect the copy.

`length` and both `normalize` overloads take their square root from `SushiBLAS::Math`, which is
`std::sqrt` on the host and `sycl::sqrt` on the device. The functions that call `std::sin`,
`std::cos`, `std::acos`, `std::tan` or `std::sqrt` directly are `quaternion_axis_angle`, `slerp`,
`perspective` and `quaternion_from_matrix`; they are host functions as they stand.

## `vector3.hpp`

### `normalize`

A vector whose length is not above zero is returned unchanged. That covers the zero vector and a
vector so small that the sum of its squares underflows to zero. The other vectors are multiplied
by `1 / length`, one division and three multiplications.

## `quaternion.hpp`

### `QuaternionT`

`(x, y, z)` is the vector part and `w` the scalar part. `mul(a, b)` is the Hamilton product, the
rotation `b` followed by `a`.

### `rotate`

With `qv = (q.x, q.y, q.z)`, the rotated vector is `v + 2w (qv x v) + 2 (qv x (qv x v))`. The
function computes `t = 2 (qv x v)` once and returns `v + w t + qv x t`: two cross products and no
matrix.

### `normalize`

A quaternion whose length is not above zero becomes the identity rotation, where the vector
overload returns its argument. A zero quaternion is no rotation at all, and the identity is the
one answer a caller can keep composing with.

## `matrix4.hpp`

### `Matrix4T`

Storage is column-major: the element at row `r`, column `c` is `m[c * 4 + r]`, so the translation
of an affine matrix is `m[12]`, `m[13]`, `m[14]`. `mul(a, b)` applies `b` first.

## `interpolation.hpp`

### `nlerp`

`nlerp` blends the four components linearly and normalises the result. It costs less than `slerp`
and, for the small angle between two neighbouring samples of an animation track, gives the same
picture. A quaternion and its negation are the same rotation, so `nlerp` first negates `b` when
`dot(a, b)` is below zero; the blend then takes the short arc.

### `slerp`

`slerp` turns at a constant angular rate along the short arc, for the crossfades of two inputs
where the uneven rate of `nlerp` shows. It negates `b` as `nlerp` does. When `dot(a, b)` is above
`1 - 1e-6` the two rotations are nearly parallel, `sin(omega)` in the denominator is close to
zero, and `slerp` returns `nlerp(a, b, t)`.

## `transform.hpp`

### `perspective`

The projection is right-handed and reverse-Z with an infinite far plane: the near plane maps to
clip depth 1 and infinity to 0, since `ndc.z = near / -z_view`. `m[5]` is `-f`, which flips Y so
that world up is screen up under Vulkan's inverted clip Y. Reverse-Z spreads the precision of a
floating-point depth buffer almost evenly over the range, so one camera can hold a 5 cm object
and one 10^7 m away in the same frame without z-fighting. The pipeline has to clear depth to 0
and compare with `GREATER_OR_EQUAL`. `far_plane` is a parameter so that call sites written for a
finite far plane compile; the function does not read it.

### `quaternion_from_matrix`

The inverse of `matrix4_from_quaternion`. It reads the upper-left 3x3 and ignores the translation
and the last row. The 3x3 must be orthonormal, with scale removed, as `decompose_transform` leaves
it. The function is Shepperd's method: it picks one of four formulas by the trace and the largest
diagonal element, so the square root is always taken of a value of at least 1 and the divisor `s`
is at least 2. The result is normalised.

### `affine_inverse`

Inverts the upper-left 3x3 by its adjugate and sends the translation through the negated inverse.
The last row is taken to be `(0, 0, 0, 1)`. Because the 3x3 is inverted in full, a matrix with
non-uniform scale or shear inverts correctly; the transpose short cut is right only for a pure
rotation. A determinant of exactly zero returns the identity.

### `decompose_transform`

The inverse of `compose_transform` for a matrix that is a translation, a rotation and a scale.
The translation is the fourth column. The scale of an axis is the length of its column. The
rotation is read from the columns divided by their scales, through `quaternion_from_matrix`.

A negative determinant means the matrix holds a reflection, which no rotation can express. The x
scale takes the minus sign, so the rotation that is left is right-handed. A column of length zero
is not divided; that axis of the rotation stays the identity axis. Shear is not recovered: a
sheared matrix comes back as a translation, a rotation and a scale that do not compose to it.

## Placement

A function sits in the header of the type it is about. Two placements were a choice:

- `rotate` takes a quaternion and a vector and is in `quaternion.hpp`, because it is the
  quaternion's action on a vector and `vector3.hpp` does not know quaternions.
- `mul` for two matrices is in `matrix4.hpp` with the type. `transform.hpp` holds only the
  functions that give a matrix a geometric meaning.
