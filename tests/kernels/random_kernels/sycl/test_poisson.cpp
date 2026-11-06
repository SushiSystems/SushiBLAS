/**************************************************************************/
/* test_poisson.cpp                                                       */
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
 * @file test_poisson.cpp
 * @brief Tests the Poisson parameters, both draw methods and their agreement on the host device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/poisson.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Poisson = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 1234567u;

        /// @brief Draws per lambda in the moment checks.
        constexpr uint64_t K_DRAWS = 4000;

        /// @brief Standard errors of the mean of K_DRAWS draws that the mean may stray.
        constexpr double K_STANDARD_ERRORS = 5.0;

        /// @brief Draws per lambda in the comparison of the device with the host.
        constexpr std::size_t K_DEVICE_DRAWS = 256;

        TEST(Small_Poisson, SmallLambdaKeepsOnlyKnuthsConstant)
        {
            const Poisson::PoissonParams small = Poisson::make_poisson_params(2.0);

            EXPECT_DOUBLE_EQ(small.exp_neg_lambda, std::exp(-2.0));
            EXPECT_EQ(small.b, 0.0);
        }

        TEST(Small_Poisson, LargeLambdaFillsTheHatOfPTRS)
        {
            const Poisson::PoissonParams large = Poisson::make_poisson_params(100.0);

            EXPECT_DOUBLE_EQ(large.log_lambda, std::log(100.0));
            EXPECT_DOUBLE_EQ(large.b, 0.931 + 2.53 * 10.0);
            // Hoermann's 0.9277 - 3.6224 / (b - 2) at b = 26.231.
            EXPECT_NEAR(large.v_r, 0.7782055507407866, absolute_tolerance<double>(8));
        }

        TEST(Small_Poisson, TheHatIsFilledFromTheThresholdUpAndNotJustBelowIt)
        {
            const double below = std::nextafter(Poisson::K_POISSON_PTRS_MIN_LAMBDA, 0.0);

            EXPECT_EQ(Poisson::make_poisson_params(below).b, 0.0);
            EXPECT_GT(Poisson::make_poisson_params(Poisson::K_POISSON_PTRS_MIN_LAMBDA).b, 0.0);
        }

        /// @brief One mean of the distribution, named for the row of the draws.
        struct LambdaRow
        {
                const char* name;
                double lambda;
        };

        /// @brief Returns two means below the PTRS threshold and two above it.
        static std::vector<LambdaRow> lambda_rows()
        {
            return {{"Lambda0Point5", 0.5}, {"Lambda4", 4.0}, {"Lambda30", 30.0}, {"Lambda250", 250.0}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string lambda_row_name(const ::testing::TestParamInfo<LambdaRow>& info)
        {
            return info.param.name;
        }

        /// @brief Returns the first draws of an element stream at a mean, as the host draws them.
        static std::vector<double> host_counts(double lambda, std::size_t count)
        {
            const Poisson::PoissonParams params = Poisson::make_poisson_params(lambda);
            std::vector<double> counts;
            for (std::size_t element = 0; element < count; ++element)
                counts.push_back(Poisson::draw_poisson(K_SEED, 0, element, params));
            return counts;
        }

        /// @brief Draws counts at one mean.
        class Small_PoissonDraws : public ::testing::TestWithParam<LambdaRow>
        {
        };

        TEST_P(Small_PoissonDraws, BothMethodsDrawCountsWithTheRightMean)
        {
            const double lambda = GetParam().lambda;
            const std::vector<double> counts = host_counts(lambda, K_DRAWS);

            const double mean = std::accumulate(counts.begin(), counts.end(), 0.0) / static_cast<double>(K_DRAWS);

            EXPECT_NEAR(mean, lambda, K_STANDARD_ERRORS * std::sqrt(lambda / static_cast<double>(K_DRAWS)));
        }

        TEST_P(Small_PoissonDraws, EveryDrawIsAWholeNonNegativeCount)
        {
            const std::vector<double> counts = host_counts(GetParam().lambda, K_DRAWS);

            EXPECT_TRUE(std::all_of(counts.begin(), counts.end(),
                                    [](double count) { return count >= 0.0 && count == std::floor(count); }));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PoissonDraws, ::testing::ValuesIn(lambda_rows()), lambda_row_name);

        /** @brief Draws on the host CPU queue. */
        class Medium_PoissonOnDevice : public HostDeviceTest, public ::testing::WithParamInterface<LambdaRow>
        {

        };

        TEST_P(Medium_PoissonOnDevice, AKernelDrawsTheCountsTheHostDraws)
        {
            const Poisson::PoissonParams params = Poisson::make_poisson_params(GetParam().lambda);
            SharedArray<double> device(queue(), K_DEVICE_DRAWS);
            double* out = device.get();

            queue().parallel_for(sycl::range<1>(K_DEVICE_DRAWS), [=](sycl::id<1> i)
            {
                out[i[0]] = Poisson::draw_poisson(K_SEED, 0, i[0], params);
            }).wait();

            EXPECT_EQ(device.values(), host_counts(GetParam().lambda, K_DEVICE_DRAWS));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_PoissonOnDevice,
                                 ::testing::Values(LambdaRow{"Lambda3", 3.0}, LambdaRow{"Lambda60", 60.0}),
                                 lambda_row_name);
    } // namespace Test
} // namespace SushiBLAS
