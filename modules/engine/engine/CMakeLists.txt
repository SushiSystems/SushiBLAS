# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME engine
    LAYER engine
    SOURCES
        engine.cpp
    DEPENDS
        core
        launch
        tensor)

sushiblas_add_test_program(
    MODULE engine
    TESTS
        test_engine/test_reduce_plans.cpp
        test_engine/test_run_failure.cpp
        test_engine/test_runtime_shapes.cpp
        test_engine/test_tensor_const_access.cpp
        test_engine/test_tensor_factory.cpp
        test_engine/test_tensor_view.cpp
        test_engine/test_unsupported_execution.cpp
        test_execution_mode.cpp
        test_operand_check.cpp
        test_rng_stream.cpp
        test_scratch_shelf.cpp
        test_task_handle.cpp
        test_task_recorder.cpp)
