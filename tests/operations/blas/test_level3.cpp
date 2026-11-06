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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
            BLASOps blas(recorder());
            Level3& level3 = blas;

            level3.gemm(a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans, 2.0, -1.0);
            ASSERT_TRUE(recording->run().has_value());

            EXPECT_EQ(read_tensor<float>(c), (std::vector<float>{3, 1, 7, 5}));
        }
    } // namespace Test
} // namespace SushiBLAS
