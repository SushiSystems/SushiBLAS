# SushiBLASLayers.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASLayers.cmake — the tier order and the tier of every module.
#
# Single responsibility: state which tier each module under modules/ belongs to and in
# which direction a dependency may point. cmake/SushiBLASModule.cmake reads these tables
# and refuses a module whose declared dependencies break them.

# Lowest tier first. A module may depend on a module of a lower tier.
set(SUSHIBLAS_LAYER_ORDER
    foundation
    data
    kernels
    engine
    operations
    library)

# Module name, then its tier. A module is added here before its folder is.
set(SUSHIBLAS_MODULE_LAYERS
    core                foundation
    arithmetic          foundation
    launch              foundation
    reduction           foundation
    reference           foundation
    substitution        foundation
    orthonormal         foundation
    sparse_assembly     foundation
    geometry            foundation
    graph_primitives    foundation
    blas_test_oracle    foundation
    serial_reductions   foundation
    test_runner         foundation
    sparse_fixtures     foundation
    typed_suite_naming  foundation
    host_device         foundation
    small_integers      foundation
    tensor              data
    sparse_storage      data
    tensor_expectations data
    blas_seam_kernels   kernels
    blas_kernels        kernels
    random_kernels      kernels
    sparse_kernels      kernels
    spatial_kernels     kernels
    signal_kernels      kernels
    blend_kernels       kernels
    packed_product_harness kernels
    engine              engine
    operands            engine
    functor_map         engine
    cluster             engine
    recording_graph     engine
    engine_fixture      engine
    execution_mode_sweep engine
    graph_probe         engine
    blas                operations
    elementwise         operations
    index               operations
    nonlinear           operations
    reductions          operations
    random              operations
    logic               operations
    sparse              operations
    spatial             operations
    signal              operations
    blend               operations
    io                  operations
    experimental        operations
    benchmark_harness   operations
    umbrella            library
    sushiblas           library)

# Consumer, then dependency: the edges allowed between two modules of one tier. An edge
# inside a tier that is not listed here is refused.
set(SUSHIBLAS_SAME_LAYER_EDGES
    launch              core
    reduction           arithmetic
    reduction           launch
    reference           core
    reference           reduction
    substitution        arithmetic
    sparse_assembly     core
    sparse_assembly     reference
    geometry            arithmetic
    sparse_storage      tensor
    blas_kernels        blas_seam_kernels
    sparse_kernels      blas_seam_kernels
    sparse_kernels      blas_kernels
    operands            engine
    functor_map         engine
    cluster             engine
    blas_test_oracle    core
    packed_product_harness blas_seam_kernels
    test_runner         core
    sparse_fixtures     core
    sparse_fixtures     arithmetic
    tensor_expectations tensor
    recording_graph     engine
    engine_fixture      engine
    engine_fixture      recording_graph
    graph_probe         engine_fixture
    execution_mode_sweep engine
    graph_probe         engine
    benchmark_harness   io)

# Gives the position of a tier in SUSHIBLAS_LAYER_ORDER, or -1.
function(sushiblas_layer_index layer output_variable)
    list(FIND SUSHIBLAS_LAYER_ORDER "${layer}" index)
    set(${output_variable} ${index} PARENT_SCOPE)
endfunction()

# Gives the tier SUSHIBLAS_MODULE_LAYERS places a module in, or the empty string.
function(sushiblas_module_layer module output_variable)
    set(${output_variable} "" PARENT_SCOPE)
    list(LENGTH SUSHIBLAS_MODULE_LAYERS entry_count)
    set(index 0)
    while(index LESS entry_count)
        math(EXPR layer_slot "${index} + 1")
        list(GET SUSHIBLAS_MODULE_LAYERS ${index} listed_module)
        if("${listed_module}" STREQUAL "${module}")
            list(GET SUSHIBLAS_MODULE_LAYERS ${layer_slot} listed_layer)
            set(${output_variable} "${listed_layer}" PARENT_SCOPE)
            return()
        endif()
        math(EXPR index "${index} + 2")
    endwhile()
endfunction()

# Says whether SUSHIBLAS_SAME_LAYER_EDGES lists the edge consumer -> dependency.
function(sushiblas_same_layer_edge_allowed consumer dependency output_variable)
    set(${output_variable} FALSE PARENT_SCOPE)
    list(LENGTH SUSHIBLAS_SAME_LAYER_EDGES entry_count)
    set(index 0)
    while(index LESS entry_count)
        math(EXPR dependency_slot "${index} + 1")
        list(GET SUSHIBLAS_SAME_LAYER_EDGES ${index} listed_consumer)
        list(GET SUSHIBLAS_SAME_LAYER_EDGES ${dependency_slot} listed_dependency)
        if("${listed_consumer}" STREQUAL "${consumer}"
                AND "${listed_dependency}" STREQUAL "${dependency}")
            set(${output_variable} TRUE PARENT_SCOPE)
            return()
        endif()
        math(EXPR index "${index} + 2")
    endwhile()
endfunction()
