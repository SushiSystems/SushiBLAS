# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME nonlinear
    LAYER operations
    SOURCES
        elu.cpp
        gelu.cpp
        leaky_relu.cpp
        relu.cpp
        sigmoid.cpp
        silu.cpp
        softplus.cpp
        tanh.cpp
    DEPENDS
        core
        arithmetic
        tensor
        engine
        functor_map)

sushiblas_add_test_program(
    MODULE nonlinear
    TESTS
        test_dispatch.cpp
        test_nonlinear/test_complex_nonlinear.cpp
        test_nonlinear/test_elu.cpp
        test_nonlinear/test_gelu.cpp
        test_nonlinear/test_leaky_relu.cpp
        test_nonlinear/test_relu.cpp
        test_nonlinear/test_sigmoid.cpp
        test_nonlinear/test_silu.cpp
        test_nonlinear/test_softplus.cpp
        test_nonlinear/test_tanh.cpp)
