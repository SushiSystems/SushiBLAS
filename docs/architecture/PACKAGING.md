# 10. Packaging

SushiBLAS is found with `find_package`: `SushiBLAS::sushiblas` is a `STATIC` library, aliased
in-tree and exported under the same name, with `install(EXPORT)`, `export(EXPORT)` and a
generated `SushiBLASConfig.cmake` (`SameMinorVersion` compatibility before 1.0). The configure
step checks `include/SushiBLAS/version.hpp` against the CMake project version. `tests/package/`
is an out-of-tree consumer; see [the integration guide](../guides/INTEGRATION.md).

Because `sushiblas` is a `STATIC` library, its build-policy targets (`sushiblas_warnings`,
`sushiblas_perf`, `sushiblas_sanitizers`, composed by `sushiblas_apply_build_policy()` in
`cmake/SushiBLASBuildPolicy.cmake`) put flag values on the real target through
`target_compile_options`, `target_compile_definitions`, `target_link_options` and
`$<TARGET_PROPERTY:...>`, never through `target_link_libraries`. For a static library,
`target_link_libraries(... PRIVATE policyTarget)` records the policy target's name in
`INTERFACE_LINK_LIBRARIES` as `$<LINK_ONLY:...>`, which breaks the export with "requires target X
that is not in any export set". Flag values record no target name.

Five CMake modules that define SushiBLAS functions or targets (`SushiBLASBuildPolicy.cmake`,
`SushiBLASCompilerWarnings.cmake`, `SushiBLASOptimization.cmake`, `SushiBLASSanitizers.cmake`,
`SushiBLASLogging.cmake`)
carry the program's full name. `Vcpkg.cmake`, `Toolchain.cmake` and `SyclTarget.cmake` do not,
because their logic gives the same result whichever sibling repository's copy loads first.
`SyclTarget.cmake` prefixes the two functions it defines, `sushiblas_runtime_device_targets` and
`sushiblas_apply_sycl`.
[CONTRIBUTING.md, section 4](../CONTRIBUTING.md#build-system-names) states the naming rule.
