# SushiBLASModule.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASModule.cmake — sushiblas_add_module(), the one way a module becomes a target.
#
# Single responsibility: turn a folder under modules/<tier>/ into its targets, check its
# declared dependencies against cmake/SushiBLASLayers.cmake, and record what the library
# target needs from it.
#
# A module has two targets. `sushiblas_<name>` is an INTERFACE target that carries the
# module's include root and its dependencies; it is what another module names. A module
# with sources also has `sushiblas_<name>_objects`, an OBJECT library the `sushiblas`
# library is composed from. Nothing links the object target, so a test can link both the
# library and a module's interface target without its objects reaching the link twice.
#
# The library does not link a module target. It reads three global properties instead,
# so the exported SushiBLAS::sushiblas names no target outside its export set.

include(SushiBLASLayers)

# The three global properties a module appends to:
#   SUSHIBLAS_MODULE_OBJECT_TARGETS    the object libraries the library is composed from
#   SUSHIBLAS_INSTALLED_INCLUDE_ROOTS  the include roots of the modules that are installed
#   SUSHIBLAS_INTERNAL_INCLUDE_ROOTS   the include roots of the INTERNAL modules

# Refuses a dependency that points up the tier order, or sideways without being listed.
function(sushiblas_check_module_edge consumer dependency)
    sushiblas_module_layer("${dependency}" dependency_layer)
    if("${dependency_layer}" STREQUAL "")
        message(FATAL_ERROR
            "sushiblas_add_module(${consumer}): depends on '${dependency}', which is not in "
            "SUSHIBLAS_MODULE_LAYERS (cmake/SushiBLASLayers.cmake).")
    endif()
    sushiblas_module_layer("${consumer}" consumer_layer)
    sushiblas_layer_index("${consumer_layer}" consumer_index)
    sushiblas_layer_index("${dependency_layer}" dependency_index)
    if(dependency_index GREATER consumer_index)
        message(FATAL_ERROR
            "sushiblas_add_module(${consumer}): upward dependency ${consumer} "
            "(${consumer_layer}) -> ${dependency} (${dependency_layer}). Move the shared "
            "code down a tier.")
    endif()
    if(dependency_index EQUAL consumer_index)
        sushiblas_same_layer_edge_allowed("${consumer}" "${dependency}" allowed)
        if(NOT allowed)
            message(FATAL_ERROR
                "sushiblas_add_module(${consumer}): ${consumer} -> ${dependency} stays inside "
                "the ${consumer_layer} tier and is not in SUSHIBLAS_SAME_LAYER_EDGES "
                "(cmake/SushiBLASLayers.cmake).")
        endif()
    endif()
endfunction()

# Declares the module in the current folder.
#
#   sushiblas_add_module(
#       NAME <module> LAYER <tier>
#       [INTERNAL]
#       [SOURCES <file under source/>...]
#       [DEPENDS <module>...]
#       [PRIVATE_DEPENDS <module>...]
#       [EXTERNAL <target>...])
#
# INTERNAL keeps the module's headers out of the install; the modules of the library can
# include them and a consumer cannot. EXTERNAL names targets from outside this tree.
# DEPENDS names the modules the public headers include, and every module that names this one
# sees them too. PRIVATE_DEPENDS names the modules only source/ includes; they reach the
# object library and no one else, so they need SOURCES.
function(sushiblas_add_module)
    cmake_parse_arguments(ARG "INTERNAL" "NAME;LAYER" "SOURCES;DEPENDS;PRIVATE_DEPENDS;EXTERNAL"
        ${ARGN})
    if("${ARG_NAME}" STREQUAL "" OR "${ARG_LAYER}" STREQUAL "")
        message(FATAL_ERROR "sushiblas_add_module(): NAME and LAYER are required.")
    endif()
    if(NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR
            "sushiblas_add_module(${ARG_NAME}): unexpected argument(s) "
            "'${ARG_UNPARSED_ARGUMENTS}'.")
    endif()
    sushiblas_module_layer("${ARG_NAME}" manifest_layer)
    if(NOT "${ARG_LAYER}" STREQUAL "${manifest_layer}")
        message(FATAL_ERROR
            "sushiblas_add_module(${ARG_NAME}): declares LAYER '${ARG_LAYER}' and "
            "SUSHIBLAS_MODULE_LAYERS says '${manifest_layer}' "
            "(cmake/SushiBLASLayers.cmake).")
    endif()
    if(ARG_PRIVATE_DEPENDS AND NOT ARG_SOURCES)
        message(FATAL_ERROR
            "sushiblas_add_module(${ARG_NAME}): PRIVATE_DEPENDS needs SOURCES; a module "
            "without sources has nothing to compile against them.")
    endif()
    foreach(dependency IN LISTS ARG_PRIVATE_DEPENDS)
        if(dependency IN_LIST ARG_DEPENDS)
            message(FATAL_ERROR
                "sushiblas_add_module(${ARG_NAME}): ${dependency} is in both DEPENDS and "
                "PRIVATE_DEPENDS.")
        endif()
    endforeach()
    foreach(dependency IN LISTS ARG_DEPENDS ARG_PRIVATE_DEPENDS)
        sushiblas_check_module_edge("${ARG_NAME}" "${dependency}")
    endforeach()

    set(target "sushiblas_${ARG_NAME}")
    set(include_root "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE "$<BUILD_INTERFACE:${include_root}>")
    foreach(dependency IN LISTS ARG_DEPENDS)
        target_link_libraries(${target} INTERFACE "sushiblas_${dependency}")
    endforeach()
    if(ARG_EXTERNAL)
        target_link_libraries(${target} INTERFACE ${ARG_EXTERNAL})
    endif()
    sushiblas_apply_logging(${target} INTERFACE)

    if(ARG_INTERNAL)
        set_property(GLOBAL APPEND PROPERTY SUSHIBLAS_INTERNAL_INCLUDE_ROOTS "${include_root}")
        # One target for every INTERNAL module's headers, which a test program links to
        # include a kernel the installed package does not ship.
        if(NOT TARGET sushiblas_internal_headers)
            add_library(sushiblas_internal_headers INTERFACE)
        endif()
        target_link_libraries(sushiblas_internal_headers INTERFACE ${target})
    else()
        set_property(GLOBAL APPEND PROPERTY SUSHIBLAS_INSTALLED_INCLUDE_ROOTS "${include_root}")
    endif()

    if(ARG_SOURCES)
        set(module_sources "")
        foreach(source IN LISTS ARG_SOURCES)
            list(APPEND module_sources "${CMAKE_CURRENT_SOURCE_DIR}/source/${source}")
        endforeach()
        set(objects "${target}_objects")
        add_library(${objects} OBJECT ${module_sources})
        # SushiRuntime is linked by every object library, whatever its headers include: its
        # PUBLIC options carry the deterministic floating-point contract, and every
        # translation unit of the library has to be compiled under it.
        target_link_libraries(${objects} PUBLIC ${target} SushiRuntime::SushiRuntime)
        foreach(dependency IN LISTS ARG_PRIVATE_DEPENDS)
            target_link_libraries(${objects} PRIVATE "sushiblas_${dependency}")
        endforeach()
        target_include_directories(${objects} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source")
        set_target_properties(${objects} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
        sushiblas_apply_build_policy(${objects})
        sushiblas_apply_sycl(${objects} SOURCES ${module_sources})
        set_property(GLOBAL APPEND PROPERTY SUSHIBLAS_MODULE_OBJECT_TARGETS ${objects})
    endif()
endfunction()
