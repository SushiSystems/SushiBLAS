# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The tensor side of SushiRuntime's distributed layer: packs, their sharing and their tasks.
# Added only when SB_ENABLE_DISTRIBUTED is on, which needs SushiRuntime built with
# SR_ENABLE_DISTRIBUTED, the condition under which SushiRuntime::distributed exists.
if(NOT TARGET SushiRuntime::distributed)
    message(FATAL_ERROR
        "SB_ENABLE_DISTRIBUTED is on and SushiRuntime has no distributed target: build "
        "SushiRuntime with SR_ENABLE_DISTRIBUTED=ON.")
endif()

sushiblas_add_module(
    NAME cluster
    LAYER engine
    SOURCES
        bound_pack.cpp
        cluster_binding.cpp
        pack_layout.cpp
        remote_task.cpp
        tensor_pack.cpp
    DEPENDS
        core
        engine
        tensor
    EXTERNAL
        SushiRuntime::distributed)

sushiblas_add_test_program(
    MODULE cluster
    TESTS
        test_bound_pack.cpp
        test_cluster_binding.cpp
        test_pack_layout.cpp
        test_remote_task.cpp
        test_tensor_pack.cpp)
