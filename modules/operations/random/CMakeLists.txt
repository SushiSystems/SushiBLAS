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
        launch
        tensor
        engine
    PRIVATE_DEPENDS
        orthonormal
        random_kernels)

sushiblas_add_test_program(
    MODULE random
    TESTS
        test_random_log_category.cpp
        test_dispatch.cpp
        test_random/test_constant.cpp
        test_random/test_distributions.cpp
        test_random/test_initializers.cpp
        test_random/test_normal.cpp
        test_random/test_orthogonal.cpp
        test_random/test_random_large_lambda.cpp
        test_random/test_random_reference.cpp
        test_random/test_shuffle.cpp
        test_random/test_uniform.cpp
    BENCHMARKS
        benchmark_random.cpp)
