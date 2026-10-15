/**************************************************************************/
/* test_orthonormal_sweep.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_orthonormal_sweep.cpp
 * @brief Tests the vector layout, the sweep count and the tiny-norm threshold of the sweep.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_OrthonormalSweep, AVectorSetStartsEmpty)
        {
            const Kernels::VectorSet set{};

            EXPECT_EQ(set.count, int64_t{0});
            EXPECT_EQ(set.length, int64_t{0});
            EXPECT_EQ(set.elem_stride, int64_t{0});
            EXPECT_EQ(set.vec_stride, int64_t{0});
        }

        TEST(Unit_OrthonormalSweep, TheSweepRunsTwice)
        {
            static_assert(Kernels::K_ORTHONORMAL_SWEEPS == 2, "Two passes of Gram-Schmidt.");

            EXPECT_EQ(Kernels::K_ORTHONORMAL_SWEEPS, 2);
        }

        TEST(Unit_OrthonormalSweep, TheTinyNormIsTheSquareOfOneEMinusTwelve)
        {
            static_assert(std::is_same<decltype(Kernels::tiny_norm_sq<float>()), float>::value,
                          "The threshold has the element type.");

            EXPECT_DOUBLE_EQ(Kernels::tiny_norm_sq<double>(), 1e-24);
            EXPECT_FLOAT_EQ(Kernels::tiny_norm_sq<float>(), 1e-24f);
            EXPECT_GT(Kernels::tiny_norm_sq<float>(), 0.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
