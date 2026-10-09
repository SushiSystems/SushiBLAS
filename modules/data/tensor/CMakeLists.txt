# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME tensor
    LAYER data
    DEPENDS
        core)

sushiblas_add_module_tests(
    MODULE tensor
    SOURCES
        test_storage_offset.cpp
        test_tensor_const_access.cpp
        test_tensor_factory.cpp
        test_tensor_view.cpp)
