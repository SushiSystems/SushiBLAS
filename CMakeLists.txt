cmake_minimum_required(VERSION 3.20)

# SushiBLAS — a head; SushiRuntime is the plugged-in component (the battery).
# Mirrors SushiEngine's own root CMakeLists.txt ordering exactly (both are
# head modules over the same backbone): Vcpkg + Toolchain run BEFORE
# project() to shape the toolchain file and pick the default SYCL compiler
# ahead of project()'s probe; everything else (the runtime dependency, the
# SYCL target policy) runs AFTER. See cmake/ for each stage.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Pre-project: toolchain shaping -------------------------------------------
include(Vcpkg)      # (Windows) locate vcpkg + set the toolchain file
include(Toolchain)  # pick the default SYCL compiler (clang++) before the probe

project(SushiBLAS VERSION 0.1.0 LANGUAGES CXX C)

# --- Post-project: runtime dependency (D3 layered resolution, SB-0), SYCL policy
include(Runtime)     # add_subdirectory(sushiruntime) -> SushiRuntime::SushiRuntime,
                      # add_sycl_to_target(), SR_SYCL_TOOLCHAIN
include(SyclTarget)  # sushiblas_apply_sycl() helper

# Compile policy. BuildPolicy composes the warning, architecture, and
# sanitizer targets and exposes sushiblas_apply_build_policy(<target>), which
# every target in the tree calls. A function rather than a directory-scope
# add_compile_options() so the in-tree-only flags never leak into the
# installed export set (see cmake/SushiBLASBuildPolicy.cmake). Removed here: the old
# global add_compile_options(-march=native / /arch:AVX2 / /fp:fast) block —
# baking the build machine's ISA and fast-math into every target
# unconditionally is exactly the anti-pattern SB-6 packaging exists to
# close. -fp:fast in particular silently reassociates and contracts FP ops,
# which would have fought SushiRuntime's own deterministic-FP contract that
# sushiblas already inherits transitively.
include(SushiBLASBuildPolicy)

# The compiled-in log level (SB_LOG_LEVEL) -> sushiblas_apply_log_level(<target>).
# Separate from BuildPolicy because it is the one compile definition that must be
# PUBLIC: the level lives in an inline constexpr variable in a public header, so a
# consumer that disagreed with the library about it would be an ODR violation.
include(SushiBLASLogging)

# --- Policies ---
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# --- Configuration ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Install layout & package location (SB-6) ---
include(GNUInstallDirs)
set(SB_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SushiBLAS"
    CACHE STRING "Where the SushiBLAS CMake package files are installed")

# The version literals in include/SushiBLAS/version.hpp are hand-written, so
# they are greppable and need no generated header on the include path. That
# trade is only safe if they cannot silently disagree with the version the
# build and the package advertise — so the disagreement is a configure-time
# error, here, rather than a consumer discovering it later. Mirrors
# sushiruntime's own version-drift check exactly.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/SushiBLAS/version.hpp"
     SB_VERSION_STRING_LINE
     REGEX "^#define SUSHIBLAS_VERSION_STRING ")
string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _sb_unused "${SB_VERSION_STRING_LINE}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR
        "Version mismatch: project() says ${PROJECT_VERSION}, "
        "include/SushiBLAS/version.hpp says ${CMAKE_MATCH_1}. "
        "Update both — they are the same fact stated twice.")
endif()

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(examples)

# --- Package files (SB-6) ---
# Written after add_subdirectory(src) so the sushiblas/SushiRuntime::SushiRuntime
# targets it references already exist.
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SushiBLASConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfig.cmake"
    INSTALL_DESTINATION "${SB_INSTALL_CMAKEDIR}"
)

# SameMinorVersion: the project is pre-1.0 and its API is not frozen, so a
# consumer asking for 0.2 must not be handed 0.3. Revisit at the v1.0.0
# freeze, where SameMajorVersion becomes the honest compatibility statement.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfigVersion.cmake"
    DESTINATION "${SB_INSTALL_CMAKEDIR}"
)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
