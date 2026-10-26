# SushiBLASHeaderModules.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Declares the header-only modules of SushiBLAS as targets a consumer links without
# building the library. A consumer includes this file from a SushiBLAS checkout and calls
# sushiblas_declare_header_modules(); it needs no SYCL toolchain and no SushiRuntime for
# SushiBLAS::Geometry, and brings its own SushiRuntime target for SushiBLAS::GraphPrimitives.
#
#   SushiBLAS::Geometry         vector, quaternion and matrix types, and Math
#   SushiBLAS::FixedOrderFold   the fixed fold order and the combiners, standard library only
#   SushiBLAS::GraphPrimitives  reduce, scan and sort recorded on a SushiRuntime task graph;
#                               prefix_scan and counting_sort are defined in the library

set(_SUSHIBLAS_HEADER_MODULES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../modules/foundation")

function(sushiblas_declare_header_modules)
    get_filename_component(root "${_SUSHIBLAS_HEADER_MODULES_ROOT}" ABSOLUTE)

    if(NOT TARGET SushiBLAS::Geometry)
        add_library(SushiBLAS::Geometry INTERFACE IMPORTED GLOBAL)
        target_include_directories(SushiBLAS::Geometry INTERFACE
            "${root}/geometry/include"
            "${root}/arithmetic/include")
    endif()

    if(NOT TARGET SushiBLAS::FixedOrderFold)
        add_library(SushiBLAS::FixedOrderFold INTERFACE IMPORTED GLOBAL)
        target_include_directories(SushiBLAS::FixedOrderFold INTERFACE
            "${root}/graph_primitives/include")
    endif()

    if(NOT TARGET SushiBLAS::GraphPrimitives)
        add_library(SushiBLAS::GraphPrimitives INTERFACE IMPORTED GLOBAL)
        target_link_libraries(SushiBLAS::GraphPrimitives INTERFACE SushiBLAS::FixedOrderFold)
    endif()
endfunction()
