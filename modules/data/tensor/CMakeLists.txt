# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME tensor
    LAYER data
    SOURCES
        tensor_factory.cpp
    DEPENDS
        core)

sushiblas_add_test_program(
    MODULE tensor
    TESTS
        test_block_maker.cpp
        test_host_block.cpp
        test_host_data_type.cpp
        test_tensor/test_shape_only_tensor.cpp
        test_runtime_block.cpp
        test_storage/test_storage.cpp
        test_storage/test_storage_block_contract.cpp
        test_storage_block.cpp
        test_tensor/test_tensor.cpp
        test_tensor_factory/test_tensor_factory.cpp
        test_tensor_factory/test_tensor_factory_contract.cpp
    COMPILE_CHECKS
        test_tensor_log_category.cpp)
