/**************************************************************************/
/* test_mean_spatial.cpp                                                  */
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
 * @file test_mean_spatial.cpp
 * @brief Tests ReductionOps::mean_spatial and the single-axis reductions that share its span.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include "support/tolerance.hpp"
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The batch extent of the NHWC case.
            constexpr int64_t K_N = 2;

            /// @brief The height of the NHWC case.
            constexpr int64_t K_H = 3;

            /// @brief The width of the NHWC case.
            constexpr int64_t K_W = 4;

            /// @brief The channel count of the NHWC case.
            constexpr int64_t K_C = 5;

            /// @brief The side of the square spatial extent of the long case.
            constexpr int64_t K_LONG_SIDE = 512;

            /// @brief Returns the values i % 17 - 8 for the elements of the NHWC case.
            std::vector<float> nhwc_source()
            {
                std::vector<float> src(static_cast<std::size_t>(K_N * K_H * K_W * K_C));
                for (std::size_t i = 0; i < src.size(); ++i)
                    src[i] = static_cast<float>(i % 17) - 8.0f;
                return src;
            }

            /// @brief Returns the values i % 7 - 3 for the elements of the long case.
            std::vector<float> long_source()
            {
                std::vector<float> src(static_cast<std::size_t>(K_LONG_SIDE * K_LONG_SIDE));
                for (std::size_t i = 0; i < src.size(); ++i)
                    src[i] = static_cast<float>(i % 7) - 3.0f;
                return src;
            }

            /**
             * @brief Expects a mean to lie within the rounding bound of the mean of src in double.
             * @param got The mean the code under test produced.
             * @param sum The sum of the values folded, in double.
             * @param magnitude The sum of their absolute values, in double.
             * @param count The number of values folded.
             */
            ::testing::AssertionResult mean_is_within_bound(float got, double sum, double magnitude,
                                                            std::size_t count)
            {
                const double elements = static_cast<double>(count);
                return TestSupport::is_within(TestSupport::measure_sum_error<float>(
                    static_cast<double>(got), sum / elements, magnitude / elements, count));
            }

            /** @brief Expects got to be the mean of every value of src within the rounding bound. */
            void expect_mean_of_all(float got, const std::vector<float>& src)
            {
                double sum = 0.0;
                double magnitude = 0.0;
                for (float value : src)
                {
                    sum += static_cast<double>(value);
                    magnitude += std::fabs(static_cast<double>(value));
                }
                EXPECT_TRUE(mean_is_within_bound(got, sum, magnitude, src.size()));
            }

            /**
             * @brief Expects each (n, c) result to be the mean of its H * W values of src.
             * @param got N * C results, row-major.
             */
            void expect_nhwc_means(const std::vector<float>& got, const std::vector<float>& src)
            {
                ASSERT_EQ(got.size(), static_cast<std::size_t>(K_N * K_C));
                for (int64_t n = 0; n < K_N; ++n)
                {
                    for (int64_t c = 0; c < K_C; ++c)
                    {
                        double sum = 0.0;
                        double magnitude = 0.0;
                        for (int64_t h = 0; h < K_H; ++h)
                        {
                            for (int64_t w = 0; w < K_W; ++w)
                            {
                                const std::size_t idx = static_cast<std::size_t>(
                                    ((n * K_H + h) * K_W + w) * K_C + c);
                                sum += static_cast<double>(src[idx]);
                                magnitude += std::fabs(static_cast<double>(src[idx]));
                            }
                        }
                        EXPECT_TRUE(mean_is_within_bound(got[static_cast<std::size_t>(n * K_C + c)],
                                                         sum, magnitude,
                                                         static_cast<std::size_t>(K_H * K_W)))
                            << "at (n, c) = (" << n << ", " << c << ")";
                    }
                }
            }
        }

        class Medium_MeanSpatialTest : public SushiBLASTest {};

        /**
         * @brief Tests that mean_spatial folds the two middle axes into one result per (n, c).
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_MeanSpatialTest, FoldsTheTwoMiddleAxesOfAnNhwcTensor)
        {
            const std::vector<float> src = nhwc_source();
            auto x = engine->create_tensor({K_N, K_H, K_W, K_C});
            auto out = engine->create_tensor({K_N, K_C});
            fill_tensor(x, src);

            SushiBLAS::ReductionOps(recorder()).mean_spatial(x, out);
            recording->run().value();

            expect_nhwc_means(read_tensor<float>(out), src);
        }

        /**
         * @brief Tests that a tensor with N = 1 and C = 1 still reduces correctly.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_MeanSpatialTest, ASingleSegmentPairIsStillCorrect)
        {
            auto x = engine->create_tensor({1, 2, 2, 1});
            auto out = engine->create_tensor({1, 1});
            fill_tensor(x, {1.0f, 2.0f, 3.0f, 4.0f});

            SushiBLAS::ReductionOps(recorder()).mean_spatial(x, out);
            recording->run().value();

            // 10 / 4 is exact in float.
            expect_tensor_equal(out, {2.5f});
        }

        /**
         * @brief Tests that one fold of 262144 elements takes the cooperative shape on the test device.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_MeanSpatialTest, ALongExtentTakesTheCooperativeShape)
        {
            const Device::Profile device = Device::profile_for(TestEnvironment::queue());

            EXPECT_TRUE(Reduce::plan_axis(device, 1,
                                          static_cast<std::size_t>(K_LONG_SIDE * K_LONG_SIDE))
                            .cooperative());
        }

        /**
         * @brief Tests the mean of a fold of 262144 elements, the length the cooperative shape serves.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_MeanSpatialTest, TheMeanOfALongExtentIsCorrect)
        {
            const std::vector<float> src = long_source();
            auto x = engine->create_tensor({1, K_LONG_SIDE, K_LONG_SIDE, 1});
            auto out = engine->create_tensor({1, 1});
            fill_tensor(x, src);

            SushiBLAS::ReductionOps(recorder()).mean_spatial(x, out);
            recording->run().value();

            expect_mean_of_all(read_tensor<float>(out)[0], src);
        }
    } // namespace Test
} // namespace SushiBLAS
