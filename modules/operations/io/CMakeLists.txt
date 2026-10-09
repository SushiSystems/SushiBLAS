# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME io
    LAYER operations
    SOURCES
        io.cpp
    DEPENDS
        core
        tensor
        engine)

sushiblas_add_module_tests(
    MODULE io
    SOURCES
        test_host_transfer.cpp)
