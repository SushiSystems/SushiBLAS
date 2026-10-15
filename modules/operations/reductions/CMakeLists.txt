# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME reductions
    LAYER operations
    SOURCES
        argmax.cpp
        argmin.cpp
        asum.cpp
        count_nonfinite.cpp
        logsumexp.cpp
        max.cpp
        max_along.cpp
        mean.cpp
        mean_spatial.cpp
        min.cpp
        min_along.cpp
        nrm2.cpp
        prod.cpp
        std.cpp
        sum.cpp
        sum_along.cpp
        var.cpp
    DEPENDS
        core
        launch
        reduction
        reference
        tensor
        engine)

sushiblas_add_test_program(
    MODULE reductions
    SIZE medium
    TESTS
        test_arg_reduction.cpp
        test_centred_reduction.cpp
        test_dispatch.cpp
        test_reductions/test_axis_reductions.cpp
        test_reductions/test_count_nonfinite.cpp
        test_reductions/test_max_min_nan.cpp
        test_reductions/test_mean_spatial.cpp
        test_reductions/test_reduce_segments.cpp
        test_reductions/test_reduction_determinism.cpp
        test_reductions/test_reductions.cpp
        test_reductions/test_scratch_buffer.cpp
        test_reductions/test_sum_edge_cases.cpp
    BENCHMARKS
        benchmark_reductions.cpp)
