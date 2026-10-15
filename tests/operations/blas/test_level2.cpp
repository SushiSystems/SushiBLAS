/**************************************************************************/
/* test_level2.cpp                                                        */
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
 * @file test_level2.cpp
 * @brief Tests that the Level 2 facet of BLASOps records gemv and ger with exact results.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/level2.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_base_of_v<Level2, BLASOps>, "BLASOps exposes the Level 2 facet.");
        static_assert(!std::is_constructible_v<Level2, TaskRecorder&>,
                      "Only the facade builds the Level 2 facet.");

        /** @brief Runs Level 2 calls through the facade on the fixture's engine. */
        class Medium_Level2 : public SushiBLASTest
        {

        };

        TEST_F(Medium_Level2, GemvAndItsTransposeGiveExactValues)
        {
            const Tensor a = create_initialized_tensor<double>({2, 3}, {1, 2, 3, 4, 5, 6});
            const Tensor x = create_initialized_tensor<double>({3}, {1, -1, 2});
            const Tensor u = create_initialized_tensor<double>({2}, {1, 1});
            Tensor y = create_initialized_tensor<double>({2}, {0, 0});
            Tensor w = create_initialized_tensor<double>({3}, {0, 0, 0});
            BLASOps blas(*engine);
            Level2& level2 = blas;

            level2.gemv(a, x, y);
            level2.gemv(a, u, w, Core::Transpose::Trans);
            ASSERT_TRUE(engine->execute().has_value());

            EXPECT_EQ(read_tensor<double>(y), (std::vector<double>{5, 11}));
            EXPECT_EQ(read_tensor<double>(w), (std::vector<double>{5, 7, 9}));
        }

        TEST_F(Medium_Level2, GerAddsTheOuterProduct)
        {
            const Tensor x = create_initialized_tensor<double>({2}, {1, 2});
            const Tensor y = create_initialized_tensor<double>({2}, {3, -1});
            Tensor a = create_initialized_tensor<double>({2, 2}, {1, 1, 1, 1});
            BLASOps blas(*engine);
            Level2& level2 = blas;

            level2.ger(x, y, a, 2.0);
            ASSERT_TRUE(engine->execute().has_value());

            EXPECT_EQ(read_tensor<double>(a), (std::vector<double>{7, -1, 13, -3}));
        }
    } // namespace Test
} // namespace SushiBLAS
