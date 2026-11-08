/**************************************************************************/
/* test_tensor_comparison.cpp                                             */
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
 * @file test_tensor_comparison.cpp
 * @brief Tests that the tensor comparisons accept a match and report a mismatch, NaN included.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_copy.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The comparison with no room for rounding.
        constexpr double K_EXACTLY = 0.0;

        /// @brief Returns a FLOAT32 tensor of host memory holding values.
        static Tensor float_tensor(const std::vector<float>& values)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const int64_t dims[1] = {static_cast<int64_t>(values.size())};
            Tensor tensor = factory.create(SushiRuntime::span<const int64_t>(dims, 1));
            copy_host_to_tensor(tensor, values.data(), values.size() * sizeof(float));
            return tensor;
        }

        /// @brief Returns a value that lies half the default tolerance above two.
        static float two_within_the_default_tolerance()
        {
            return 2.0f + static_cast<float>(K_DEFAULT_TOLERANCE / 2.0);
        }

        TEST(Small_TensorComparison, ExpectTensorAcceptsValuesWithinTheTolerance)
        {
            const Tensor tensor = float_tensor({1.0f, 2.0f, 3.0f});

            EXPECT_TENSOR_NEAR(tensor,
                               (std::vector<float>{1.0f, two_within_the_default_tolerance(), 3.0f}),
                               K_DEFAULT_TOLERANCE);
        }

        TEST(Small_TensorComparison, AssertTensorNearAcceptsEqualTensorsAtNoTolerance)
        {
            const Tensor tensor = float_tensor({1.0f, 2.0f, 3.0f});

            SB_ASSERT_TENSOR_NEAR(tensor, float_tensor({1.0f, 2.0f, 3.0f}), K_EXACTLY, float);
        }

        TEST(Small_TensorComparison, ReportsAValueOutsideTheToleranceWithItsIndex)
        {
            const Tensor tensor = float_tensor({1.0f, 2.0f});

            EXPECT_NONFATAL_FAILURE(expect_tensor(tensor, std::vector<float>{1.0f, 2.5f}),
                                    "Mismatch (value) at index 1");
        }

        TEST(Small_TensorComparison, ANaNExpectationAcceptsANaN)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();

            expect_tensor_equal(float_tensor({nan, 1.0f}), std::vector<float>{nan, 1.0f});
        }

        TEST(Small_TensorComparison, ANaNExpectationRefusesANumber)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();

            EXPECT_NONFATAL_FAILURE(expect_tensor_equal(float_tensor({0.0f, 1.0f}),
                                                        std::vector<float>{nan, 1.0f}),
                                    "Expected NaN");
        }

        TEST(Small_TensorComparison, AnInfinityExpectationAcceptsTheSameInfinity)
        {
            const float infinity = std::numeric_limits<float>::infinity();

            expect_tensor_equal(float_tensor({infinity}), std::vector<float>{infinity});
        }

        TEST(Small_TensorComparison, AnInfinityExpectationRefusesTheOppositeInfinity)
        {
            const float infinity = std::numeric_limits<float>::infinity();

            EXPECT_NONFATAL_FAILURE(expect_tensor_equal(float_tensor({-infinity}),
                                                        std::vector<float>{infinity}),
                                    "Infinity mismatch");
        }
    } // namespace Test
} // namespace SushiBLAS
