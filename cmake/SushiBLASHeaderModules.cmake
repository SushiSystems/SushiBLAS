# SushiBLASHeaderModules.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Declares the header-only modules of SushiBLAS as targets a consumer links without
# building the library. A consumer includes this file from a SushiBLAS checkout and calls
# sushiblas_declare_header_modules(); it needs no SYCL toolchain and no SushiRuntime for
# SushiBLAS::Geometry, and brings its own SushiRuntime target for SushiBLAS::GraphPrimitives.
# The install carries this file and SushiBLASConfig.cmake includes and calls it, so a package
# consumer gets the same targets over the installed include folder.
#
#   SushiBLAS::Geometry         vector, quaternion and matrix types, and Math
#   SushiBLAS::FixedOrderFold   the fixed fold order and the combiners, standard library only
#   SushiBLAS::GraphPrimitives  reduce, scan and sort recorded on a SushiRuntime task graph;
#                               prefix_scan and counting_sort are defined in the library

# In a checkout every module has its own include folder under modules/foundation. In an install
# the modules' headers are one folder, include/, three levels above lib/cmake/SushiBLAS.
set(_SUSHIBLAS_HEADER_MODULES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../modules/foundation")
set(_SUSHIBLAS_HEADER_MODULES_INSTALLED_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../../../include")

function(sushiblas_declare_header_modules)
    get_filename_component(root "${_SUSHIBLAS_HEADER_MODULES_ROOT}" ABSOLUTE)
    get_filename_component(installed_include "${_SUSHIBLAS_HEADER_MODULES_INSTALLED_INCLUDE}" ABSOLUTE)
    if(EXISTS "${root}/geometry/include")
        set(geometry_includes "${root}/geometry/include" "${root}/arithmetic/include")
        set(fold_includes "${root}/graph_primitives/include")
    else()
        set(geometry_includes "${installed_include}")
        set(fold_includes "${installed_include}")
    endif()

    if(NOT TARGET SushiBLAS::Geometry)
        add_library(SushiBLAS::Geometry INTERFACE IMPORTED GLOBAL)
        target_include_directories(SushiBLAS::Geometry INTERFACE ${geometry_includes})
    endif()

    if(NOT TARGET SushiBLAS::FixedOrderFold)
        add_library(SushiBLAS::FixedOrderFold INTERFACE IMPORTED GLOBAL)
        target_include_directories(SushiBLAS::FixedOrderFold INTERFACE ${fold_includes})
    endif()

    if(NOT TARGET SushiBLAS::GraphPrimitives)
        add_library(SushiBLAS::GraphPrimitives INTERFACE IMPORTED GLOBAL)
        target_link_libraries(SushiBLAS::GraphPrimitives INTERFACE SushiBLAS::FixedOrderFold)
    endif()
endfunction()
