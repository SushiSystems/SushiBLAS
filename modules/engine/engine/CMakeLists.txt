# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME engine
    LAYER engine
    SOURCES
        engine.cpp
        engine_profile.cpp
        operation_options.cpp
    DEPENDS
        core
        cost
        launch
        tensor
    PRIVATE_DEPENDS
        profile)

sushiblas_add_test_program(
    MODULE engine
    TESTS
        test_dispatch.cpp
        test_late_count_hold.cpp
        test_operand_set.cpp
        test_operation_options.cpp
        test_profile_hooks.cpp
        test_scratch.cpp
        test_host_scratch_source.cpp
        test_scratch_block_source.cpp
        test_runtime_scratch_source.cpp
        test_scratch_pool.cpp
        test_scratch_shelf.cpp
        test_task.cpp
        test_task_cost.cpp
        test_graph_task_sink.cpp
        test_host_task_sink.cpp
        test_task_description.cpp
        test_tensor_profile.cpp
    DEVICE_TESTS
        test_engine/test_engine_contract.cpp
        test_engine/test_reduce_plans.cpp
        test_engine/test_run_failure.cpp
        test_engine/test_runtime_shapes.cpp
        test_engine/test_tensor_const_access.cpp
        test_engine/test_tensor_factory.cpp
        test_engine/test_tensor_view.cpp
        test_engine/test_unsupported_execution.cpp
        test_execution_mode.cpp
        test_kernel_pair_dispatcher.cpp
        test_operand_check.cpp
        test_rng_stream.cpp
        test_task_handle.cpp
        test_task_recorder.cpp
        test_task_sink.cpp
    COMPILE_CHECKS
        test_engine_log_category.cpp)
