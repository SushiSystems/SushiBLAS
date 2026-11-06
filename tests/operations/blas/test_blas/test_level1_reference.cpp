/**************************************************************************/
/* test_level1_reference.cpp                                              */
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
 * @file test_level1_reference.cpp
 * @brief Tests the BLAS Level 1 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 16 FLOAT32 roundings of a value of order one,
        /// for every comparison in this file.
        constexpr double K_TOLERANCE = absolute_tolerance<float>(16);

        /**
         * @brief Runs every Level 1 call of a test on the reference kernels.
         */
        class Medium_Level1ReferenceTest : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and forces the reference path. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    set_mode(ExecutionMode::ForceReference);
                }
        };

        TEST_F(Medium_Level1ReferenceTest, RotAppliesThePlaneRotation)
        {
            const std::vector<float> K_EXPECTED_X = {3.0f, 4.4f};
            const std::vector<float> K_EXPECTED_Y = {1.0f, 0.8f};

            auto x = create_initialized_tensor<float>({2}, {1.0f, 2.0f});
            auto y = create_initialized_tensor<float>({2}, {3.0f, 4.0f});

            SushiBLAS::BLASOps(recorder()).rot(x, y, 0.6, 0.8);
            recording->run().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED_X, K_TOLERANCE);
            EXPECT_TENSOR_NEAR(y, K_EXPECTED_Y, K_TOLERANCE);
        }
    } // namespace Test
} // namespace SushiBLAS
