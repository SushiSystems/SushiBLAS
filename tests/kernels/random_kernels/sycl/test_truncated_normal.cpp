/**************************************************************************/
/* test_truncated_normal.cpp                                              */
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
 * @file test_truncated_normal.cpp
 * @brief Tests the normal distribution function and the truncated draws it inverts.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/truncated_normal.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Truncated = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 777u;

        TEST(Medium_TruncatedNormal, TheDistributionFunctionHasItsKnownValues)
        {
            EXPECT_DOUBLE_EQ(Truncated::standard_normal_cdf(0.0), 0.5);
            EXPECT_NEAR(Truncated::standard_normal_cdf(1.0), 0.8413447460685429, 1e-15);
            EXPECT_NEAR(Truncated::standard_normal_cdf(-2.0), 0.02275013194817921, 1e-16);
        }

        TEST(Medium_TruncatedNormal, EveryDrawLiesInsideItsBounds)
        {
            for (uint64_t block = 0; block < 200; ++block)
            {
                for (const double value :
                     Truncated::draw_truncated_standard_normal<double>(K_SEED, 0, block, -1.0,
                                                                       0.5))
                {
                    EXPECT_GE(value, -1.0);
                    EXPECT_LE(value, 0.5);
                }
            }
        }

        TEST(Medium_TruncatedNormal, ANarrowWindowDrawsNearItsCentre)
        {
            for (const float value :
                 Truncated::draw_truncated_standard_normal<float>(K_SEED, 1, 0, -0.01f, 0.01f))
                EXPECT_NEAR(value, 0.0f, 0.01f);
        }

        /** @brief Draws on the host CPU queue. */
        class Medium_TruncatedNormalOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_TruncatedNormalOnDevice, AKernelDrawsWithinTheBoundsTheHostUses)
        {
            constexpr std::size_t blocks = 64;
            SharedArray<double> device(queue(), 2 * blocks);
            double* out = device.get();

            queue().parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> i)
            {
                const auto values =
                    Truncated::draw_truncated_standard_normal<double>(K_SEED, 3, i[0], -2.0, 1.0);
                out[2 * i[0]] = values[0];
                out[2 * i[0] + 1] = values[1];
            }).wait();

            for (std::size_t block = 0; block < blocks; ++block)
            {
                const auto host =
                    Truncated::draw_truncated_standard_normal<double>(K_SEED, 3, block, -2.0, 1.0);
                // erfc may round differently on the device; the bisection then ends a step away.
                EXPECT_NEAR(device.get()[2 * block], host[0], 1e-12);
                EXPECT_NEAR(device.get()[2 * block + 1], host[1], 1e-12);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
