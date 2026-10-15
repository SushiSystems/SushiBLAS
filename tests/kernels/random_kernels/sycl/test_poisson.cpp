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
 * @file test_poisson.cpp
 * @brief Tests the Poisson parameters, both draw methods and their agreement on the host device.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/poisson.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Poisson = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 1234567u;

        /// @brief Draws per lambda in the moment checks.
        constexpr uint64_t K_DRAWS = 4000;

        TEST(Medium_Poisson, SmallLambdaKeepsOnlyKnuthsConstant)
        {
            const Poisson::PoissonParams small = Poisson::make_poisson_params(2.0);

            EXPECT_DOUBLE_EQ(small.exp_neg_lambda, std::exp(-2.0));
            EXPECT_EQ(small.b, 0.0);
        }

        TEST(Medium_Poisson, LargeLambdaFillsTheHatOfPTRS)
        {
            const Poisson::PoissonParams large = Poisson::make_poisson_params(100.0);

            EXPECT_DOUBLE_EQ(large.log_lambda, std::log(100.0));
            EXPECT_DOUBLE_EQ(large.b, 0.931 + 2.53 * 10.0);
            EXPECT_GT(large.v_r, 0.0);
        }

        TEST(Medium_Poisson, BothMethodsDrawCountsWithTheRightMean)
        {
            for (const double lambda : {0.5, 4.0, 30.0, 250.0})
            {
                SCOPED_TRACE(lambda);
                const Poisson::PoissonParams params = Poisson::make_poisson_params(lambda);
                double sum = 0.0;
                for (uint64_t element = 0; element < K_DRAWS; ++element)
                {
                    const double count = Poisson::draw_poisson(K_SEED, 0, element, params);
                    ASSERT_GE(count, 0.0);
                    ASSERT_EQ(count, std::floor(count));
                    sum += count;
                }
                // Five standard errors of the mean of K_DRAWS draws.
                const double tolerance = 5.0 * std::sqrt(lambda / static_cast<double>(K_DRAWS));
                EXPECT_NEAR(sum / static_cast<double>(K_DRAWS), lambda, tolerance);
            }
        }

        /** @brief Draws on the host CPU queue. */
        class Medium_PoissonOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_PoissonOnDevice, AKernelDrawsTheCountsTheHostDraws)
        {
            constexpr std::size_t count = 256;
            for (const double lambda : {3.0, 60.0})
            {
                SCOPED_TRACE(lambda);
                const Poisson::PoissonParams params = Poisson::make_poisson_params(lambda);
                SharedArray<double> device(queue(), count);
                double* out = device.get();

                queue().parallel_for(sycl::range<1>(count), [=](sycl::id<1> i)
                {
                    out[i[0]] = Poisson::draw_poisson(K_SEED, 2, i[0], params);
                }).wait();

                for (std::size_t i = 0; i < count; ++i)
                    EXPECT_EQ(device.get()[i], Poisson::draw_poisson(K_SEED, 2, i, params));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
