# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME random
    LAYER operations
    SOURCES
        bernoulli.cpp
        constant.cpp
        discrete_uniform.cpp
        exponential.cpp
        he_normal.cpp
        he_uniform.cpp
        log_normal.cpp
        normal.cpp
        orthogonal.cpp
        poisson.cpp
        set_seed.cpp
        shuffle.cpp
        truncated_normal.cpp
        uniform.cpp
        xavier_normal.cpp
        xavier_uniform.cpp
    DEPENDS
        core
        support
        tensor
        random_kernels
        engine)

sushiblas_add_module_tests(
    MODULE random
    SOURCES
        test_constant.cpp
        test_distributions.cpp
        test_initializers.cpp
        test_normal.cpp
        test_orthogonal.cpp
        test_random_large_lambda.cpp
        test_random_reference.cpp
        test_shuffle.cpp
        test_uniform.cpp)
