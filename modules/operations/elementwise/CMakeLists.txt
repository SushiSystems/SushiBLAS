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
        launch
        reference
        tensor
        engine
        functor_map)

sushiblas_add_module_tests(
    MODULE elementwise
    SOURCES
        test_abs.cpp
        test_acos.cpp
        test_acosh.cpp
        test_add.cpp
        test_asin.cpp
        test_asinh.cpp
        test_atan.cpp
        test_atanh.cpp
        test_ceil.cpp
        test_clamp.cpp
        test_convert.cpp
        test_cos.cpp
        test_cosh.cpp
        test_div.cpp
        test_exp.cpp
        test_floor.cpp
        test_fmod.cpp
        test_integer_arithmetic.cpp
        test_log.cpp
        test_mul.cpp
        test_neg.cpp
        test_pow.cpp
        test_reciprocal.cpp
        test_remainder.cpp
        test_round.cpp
        test_sin.cpp
        test_sinh.cpp
        test_sqrt.cpp
        test_square.cpp
        test_sub.cpp
        test_tan.cpp)
