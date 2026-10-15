# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME functor_map
    LAYER engine
    DEPENDS
        core
        launch
        reference
        tensor
        engine)

sushiblas_add_test_program(
    MODULE functor_map
    TESTS
        test_read_binding.cpp)
