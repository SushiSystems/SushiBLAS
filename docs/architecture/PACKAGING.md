# 10. Packaging

SushiBLAS is found with `find_package`: `SushiBLAS::sushiblas` is a `STATIC` library, aliased
in-tree and exported under the same name, with `install(EXPORT)`, `export(EXPORT)` and a
generated `SushiBLASConfig.cmake` (`SameMinorVersion` compatibility before 1.0). The configure
step checks `modules/foundation/core/include/SushiBLAS/core/version.hpp` against the CMake project version. `tests/package/`
is an out-of-tree consumer; see [the integration guide](../guides/INTEGRATION.md).

Because `sushiblas` is a `STATIC` library, its build-policy targets (`sushiblas_warnings`,
`sushiblas_perf`, `sushiblas_sanitizers`, composed by `sushiblas_apply_build_policy()` in
`cmake/SushiBLASBuildPolicy.cmake`) put flag values on the real target through
`target_compile_options`, `target_compile_definitions`, `target_link_options` and
`$<TARGET_PROPERTY:...>`, never through `target_link_libraries`. For a static library,
`target_link_libraries(... PRIVATE policyTarget)` records the policy target's name in
`INTERFACE_LINK_LIBRARIES` as `$<LINK_ONLY:...>`, which breaks the export with "requires target X
that is not in any export set". Flag values record no target name.

The CMake modules that define SushiBLAS functions or targets carry the program's full name:
the nine `SushiBLAS*.cmake` files under `cmake/`. `Vcpkg.cmake`, `Toolchain.cmake` and `SyclTarget.cmake` do not,
because their logic gives the same result whichever sibling repository's copy loads first.
`SyclTarget.cmake` prefixes the two functions it defines, `sushiblas_runtime_device_targets` and
`sushiblas_apply_sycl`.
[CONTRIBUTING.md, section 4](../CONTRIBUTING.md#build-system-names) states the naming rule.

## What is installed

The install copies the include root of every module that is not internal, so a consumer of the
package can include the umbrella header, any header it names, the `dispatch.hpp` of each family
and the seam kernels of `blas_seam_kernels`. `blas_kernels`, `random_kernels`, `sparse_kernels`
and `spatial_kernels` are declared `INTERNAL` in their `CMakeLists.txt` and are left out
(`cmake/SushiBLASModule.cmake`).

## The header-only modules without the library

`cmake/SushiBLASHeaderModules.cmake` is for a consumer that wants the header-only foundation
modules and not the library. It includes that file from a SushiBLAS checkout and calls
`sushiblas_declare_header_modules()`, which declares three interface targets:

| Target | Gives |
| --- | --- |
| `SushiBLAS::Geometry` | The vector, quaternion and matrix types of `geometry` and `Math`; needs no SYCL toolchain and no SushiRuntime |
| `SushiBLAS::FixedOrderFold` | The fixed fold order and the combiners of `graph_primitives`; standard library only |
| `SushiBLAS::GraphPrimitives` | `reduce`, `segmented_reduce`, `prefix_scan` and `counting_sort`; the consumer brings its own SushiRuntime target |
