# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME geometry
    LAYER foundation
    DEPENDS
        arithmetic)

sushiblas_add_test_program(
    MODULE geometry
    TESTS
        test_interpolation.cpp
        test_matrix4.cpp
        test_quaternion.cpp
        test_spatial_geometry.cpp
        test_transform.cpp
        test_vector3.cpp)
