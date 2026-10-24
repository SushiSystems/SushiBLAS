# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME elementwise
    LAYER operations
    SOURCES
        abs.cpp
        acos.cpp
        acosh.cpp
        add.cpp
        asin.cpp
        asinh.cpp
        atan.cpp
        atanh.cpp
        ceil.cpp
        clamp.cpp
        convert.cpp
        cos.cpp
        cosh.cpp
        div.cpp
        exp.cpp
        floor.cpp
        fmod.cpp
        log.cpp
        max.cpp
        min.cpp
        mul.cpp
        neg.cpp
        ordered_sum.cpp
        pow.cpp
        reciprocal.cpp
        remainder.cpp
        round.cpp
        sin.cpp
        sinh.cpp
        sqrt.cpp
        square.cpp
        sub.cpp
        tan.cpp
    DEPENDS
        core
        arithmetic
        tensor
        engine
        functor_map
    PRIVATE_DEPENDS
        launch
        reference)

sushiblas_add_test_program(
    MODULE elementwise
    TESTS
        test_elementwise_log_category.cpp
        test_dispatch.cpp
        test_elementwise/test_abs.cpp
        test_elementwise/test_acos.cpp
        test_elementwise/test_acosh.cpp
        test_elementwise/test_add.cpp
        test_elementwise/test_asin.cpp
        test_elementwise/test_asinh.cpp
        test_elementwise/test_atan.cpp
        test_elementwise/test_atanh.cpp
        test_elementwise/test_ceil.cpp
        test_elementwise/test_clamp.cpp
        test_elementwise/test_convert.cpp
        test_elementwise/test_cos.cpp
        test_elementwise/test_cosh.cpp
        test_elementwise/test_div.cpp
        test_elementwise/test_exp.cpp
        test_elementwise/test_floor.cpp
        test_elementwise/test_fmod.cpp
        test_elementwise/test_integer_arithmetic.cpp
        test_elementwise/test_log.cpp
        test_elementwise/test_max.cpp
        test_elementwise/test_min.cpp
        test_elementwise/test_mul.cpp
        test_elementwise/test_neg.cpp
        test_elementwise/test_ordered_sum.cpp
        test_elementwise/test_pow.cpp
        test_elementwise/test_reciprocal.cpp
        test_elementwise/test_remainder.cpp
        test_elementwise/test_round.cpp
        test_elementwise/test_sin.cpp
        test_elementwise/test_sinh.cpp
        test_elementwise/test_sqrt.cpp
        test_elementwise/test_square.cpp
        test_elementwise/test_sub.cpp
        test_elementwise/test_tan.cpp
        test_wrapping_math.cpp)
