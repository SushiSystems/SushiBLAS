# SushiBLASSyclHeaders.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Finds the include directory of the SYCL toolchain that compiles this project, the folder that
# holds <sycl/sycl.hpp>. -fsycl puts it on the search path of the unit it compiles; a unit
# compiled without -fsycl, which still includes the SYCL headers through SushiBLAS's and
# SushiRuntime's public headers, needs it named. cmake/SyclTarget.cmake calls this while
# SushiBLAS builds, and the installed SushiBLASConfig.cmake calls it with the consumer's
# compiler, since a path of the build machine cannot go into the export.

# sushiblas_find_sycl_include_dir(<out_var>)
#   Sets <out_var> to <root>/include when the C++ compiler is <root>/bin/<compiler> and
#   <root>/include/sycl/sycl.hpp exists, and to an empty string otherwise.
function(sushiblas_find_sycl_include_dir out_var)
    set(found "")
    get_filename_component(compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(compiler_root "${compiler_bin}" DIRECTORY)
    if(EXISTS "${compiler_root}/include/sycl/sycl.hpp")
        set(found "${compiler_root}/include")
    endif()
    set(${out_var} "${found}" PARENT_SCOPE)
endfunction()
