# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# --- Define SushiBLAS Library ---
# The archive has no source of its own: every object in it comes from a module.
add_library(sushiblas STATIC)

# --- Modules ---
# What sushiblas_add_module() recorded (cmake/SushiBLASModule.cmake): the object
# libraries this archive is composed from and the include roots of the modules. The
# library reads the two lists and links no module target, so the exported target
# names nothing outside its export set.
get_property(SB_MODULE_OBJECT_TARGETS GLOBAL PROPERTY SUSHIBLAS_MODULE_OBJECT_TARGETS)
get_property(SB_INSTALLED_INCLUDE_ROOTS GLOBAL PROPERTY SUSHIBLAS_INSTALLED_INCLUDE_ROOTS)
foreach(module_objects IN LISTS SB_MODULE_OBJECT_TARGETS)
    target_sources(sushiblas PRIVATE "$<TARGET_OBJECTS:${module_objects}>")
endforeach()
foreach(include_root IN LISTS SB_INSTALLED_INCLUDE_ROOTS)
    target_include_directories(sushiblas PUBLIC "$<BUILD_INTERFACE:${include_root}>")
endforeach()

# A namespaced alias so an in-tree consumer (add_subdirectory) and an out-of-tree one
# (find_package) spell the dependency identically - mirrors SushiRuntime::SushiRuntime
# exactly (SB-6).
add_library(SushiBLAS::sushiblas ALIAS sushiblas)

# --- Include Directories ---
# The installed modules' include roots are the public surface a consumer compiles
# against: BUILD_INTERFACE while building this tree, INSTALL_INTERFACE afterwards, where
# they are one folder. The INTERNAL modules' roots are not on this target at all; a test
# that needs one links that module's interface target.
target_include_directories(sushiblas PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# --- Compile Policy ---
# Warnings, architecture targeting and sanitizers, PRIVATE to this tree (SB-6).
sushiblas_apply_build_policy(sushiblas)

# SB_ENABLE_LOGGING, PUBLIC (see cmake/SushiBLASLogging.cmake) so examples, tests and
# out-of-tree consumers compile the same log calls this archive was compiled with.
sushiblas_apply_logging(sushiblas)

# --- Link Libraries ---
# Link against dependencies. PUBLIC so consumers (like the app) inherit them.
# SushiRuntime is always linked via its namespaced target, which cmake/Runtime.cmake finds as
# a package.
# This is also how SushiBLAS inherits SushiRuntime's `sushiruntime_deterministic_fp`
# contract transitively - SushiBLAS does not need (or own) a copy of that
# guarantee, since it links SushiRuntime::SushiRuntime PUBLIC.
# Not linked directly: IntelSYCL::SYCL_CXX. It's an imported target defined
# inside sushiruntime's own directory scope (non-GLOBAL, per its Sycl.cmake),
# invisible by name to sushiblas's sibling scope - but SushiRuntime::SushiRuntime
# already carries it PUBLIC, so it reaches sushiblas transitively regardless.
target_link_libraries(sushiblas PUBLIC
    SushiRuntime::SushiRuntime
)
# The public headers of the profile and profiling modules include SushiProfiler, and the
# archive holds the profiling module's objects.
if(SB_ENABLE_PROFILER)
    target_link_libraries(sushiblas PUBLIC SushiProfiler::SushiProfiler)
endif()

# --- Properties ---
# C++17, matching SushiRuntime and the root CMakeLists.txt; see the note there for why
# the two must agree rather than merely both being modern.
set_target_properties(sushiblas PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
)

# What the archive hands a consumer for SYCL (cmake/SyclTarget.cmake): the SYCL headers and the
# device link options, and no device compile option. A consumer's unit that instantiates a
# SushiBLAS kernel template takes the compile options of SushiBLAS::device_code.
sushiblas_publish_sycl(sushiblas)

# --- Install & export (SB-6) ---
#
# What makes SushiBLAS consumable out of tree via find_package(SushiBLAS).
# Mirrors SushiRuntime's own install/export shape exactly. Everything the
# build applies beyond this - warnings, -Werror, sanitizers, -march - is
# PRIVATE and never reaches here, which is the boundary cmake/SushiBLASBuildPolicy.cmake
# exists to draw.
install(TARGETS sushiblas
    EXPORT SushiBLASTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# The two interface targets of cmake/SyclTarget.cmake: sushiblas links the first, and a consumer
# names the second for its device units.
install(TARGETS sushiblas_sycl_headers sushiblas_device_code
    EXPORT SushiBLASTargets
)

foreach(include_root IN LISTS SB_INSTALLED_INCLUDE_ROOTS)
    install(DIRECTORY "${include_root}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
endforeach()

install(EXPORT SushiBLASTargets
    FILE SushiBLASTargets.cmake
    NAMESPACE SushiBLAS::
    DESTINATION "${SB_INSTALL_CMAKEDIR}"
)

# The same targets, usable from a build tree without installing. A consumer
# can point CMAKE_PREFIX_PATH at the build directory during development.
export(EXPORT SushiBLASTargets
    FILE "${PROJECT_BINARY_DIR}/SushiBLASTargets.cmake"
    NAMESPACE SushiBLAS::
)

