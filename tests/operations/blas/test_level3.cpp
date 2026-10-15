/**************************************************************************/
/* test_level3.cpp                                                        */
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
 * @file test_level3.cpp
 * @brief Tests that the Level 3 facet of BLASOps records gemm with exact results.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/level3.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_base_of_v<Level3, BLASOps>, "BLASOps exposes the Level 3 facet.");
        static_assert(!std::is_constructible_v<Level3, TaskRecorder&>,
                      "Only the facade builds the Level 3 facet.");

        /** @brief Runs Level 3 calls through the facade on the fixture's engine. */
        class Medium_Level3 : public SushiBLASTest
        {

        };

        TEST_F(Medium_Level3, GemmWithScalarsGivesExactValues)
        {
            const Tensor a = create_initialized_tensor<float>({2, 2}, {1, 2, 3, 4});
            const Tensor b = create_initialized_tensor<float>({2, 2}, {0, 1, 1, 0});
            Tensor c = create_initialized_tensor<float>({2, 2}, {1, 1, 1, 1});
            BLASOps blas(*engine);
            Level3& level3 = blas;

            level3.gemm(a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 2.0, -1.0);
            ASSERT_TRUE(engine->execute().has_value());

            EXPECT_EQ(read_tensor<float>(c), (std::vector<float>{3, 1, 7, 5}));
        }
    } // namespace Test
} // namespace SushiBLAS
