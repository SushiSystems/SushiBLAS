/**************************************************************************/
/* test_level1.cpp                                                        */
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
 * @file test_level1.cpp
 * @brief Tests that the Level 1 facet of BLASOps records copy, scal and swap with exact results.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/level1.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_base_of_v<Level1, BLASOps>, "BLASOps exposes the Level 1 facet.");
        static_assert(!std::is_constructible_v<Level1, TaskRecorder&>,
                      "Only the facade builds the Level 1 facet.");

        /** @brief Runs Level 1 calls through the facade on the fixture's engine. */
        class Medium_Level1 : public SushiBLASTest
        {

        };

        TEST_F(Medium_Level1, CopyScaleAndSwapGiveExactValues)
        {
            const Tensor x = create_initialized_tensor<float>({3}, {1.0f, -2.0f, 3.0f});
            Tensor y = create_initialized_tensor<float>({3}, {0.0f, 0.0f, 0.0f});
            Tensor z = create_initialized_tensor<float>({3}, {7.0f, 8.0f, 9.0f});
            BLASOps blas(*engine);
            Level1& level1 = blas;

            level1.copy(x, y);
            level1.scal(2.0f, y);
            level1.swap(y, z);
            ASSERT_TRUE(engine->execute().has_value());

            EXPECT_EQ(read_tensor<float>(z), (std::vector<float>{2.0f, -4.0f, 6.0f}));
            EXPECT_EQ(read_tensor<float>(y), (std::vector<float>{7.0f, 8.0f, 9.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
