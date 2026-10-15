/**************************************************************************/
/* test_philox.cpp                                                        */
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
 * @file test_philox.cpp
 * @brief Tests Philox4x32-10 against its known answers and the draws built on it, host and device.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/philox.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Philox = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 20261015u;

        TEST(Unit_Philox, MatchesTheKnownAnswersOfRandom123)
        {
            using Block = std::array<uint32_t, 4>;
            const uint32_t ones = 0xFFFFFFFFu;

            EXPECT_EQ(Philox::philox4x32_10(0, 0, 0, 0, 0, 0),
                      (Block{0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u}));
            EXPECT_EQ(Philox::philox4x32_10(ones, ones, ones, ones, ones, ones),
                      (Block{0x408F276Du, 0x41C83B0Eu, 0xA20BC7C6u, 0x6D5451FDu}));
        }

        TEST(Unit_Philox, UniformDrawsLieInTheirHalfOpenRange)
        {
            EXPECT_EQ(Philox::uniform01_float(0u), 0.0f);
            EXPECT_LT(Philox::uniform01_float(0xFFFFFFFFu), 1.0f);
            EXPECT_LT(Philox::uniform01_double(0xFFFFFFFFu, 0xFFFFFFFFu), 1.0);

            for (uint64_t block = 0; block < 100; ++block)
            {
                for (const float value : Philox::uniform_block<float>(K_SEED, 0, block, -2.0f,
                                                                       3.0f))
                {
                    EXPECT_GE(value, -2.0f);
                    EXPECT_LT(value, 3.0f);
                }
            }
        }

        TEST(Unit_Philox, ABlockGivesFourFloatsOrTwoDoubles)
        {
            static_assert(Philox::K_DRAWS_PER_BLOCK<float> == 4, "Four floats per block.");
            static_assert(Philox::K_DRAWS_PER_BLOCK<double> == 2, "Two doubles per block.");

            EXPECT_EQ(Philox::uniform01_block<double>(Philox::philox_block(K_SEED, 0, 0)).size(),
                      std::size_t{2});
        }

        TEST(Unit_Philox, TheDrawIsAFunctionOfSeedOffsetAndBlock)
        {
            EXPECT_EQ(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 3, 7));
            EXPECT_NE(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 4, 7));
            EXPECT_NE(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 3, 8));
        }

        TEST(Unit_Philox, AnIndexStaysBelowItsBound)
        {
            EXPECT_EQ(Philox::draw_index(K_SEED, 0, 5, 1), uint64_t{0});
            EXPECT_EQ(Philox::draw_index(K_SEED, 0, 5, 0), uint64_t{0});
            for (uint64_t index = 0; index < 200; ++index)
                EXPECT_LT(Philox::draw_index(K_SEED, 0, index, 13), uint64_t{13});
        }

        TEST(Unit_Philox, NormalDrawsAreFinite)
        {
            for (uint64_t block = 0; block < 100; ++block)
                for (const double value : Philox::normal_block<double>(K_SEED, 0, block, 1.0, 2.0))
                    EXPECT_TRUE(std::isfinite(value));
        }

        /** @brief Draws on the host CPU queue. */
        class Device_Philox : public HostDeviceTest
        {

        };

        TEST_F(Device_Philox, AKernelDrawsTheBitsTheHostDraws)
        {
            constexpr std::size_t blocks = 64;
            SharedArray<float> device(queue(), 4 * blocks);
            float* out = device.get();

            queue().parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> i)
            {
                const auto values = Philox::normal_block<float>(K_SEED, 9, i[0], 0.0f, 1.0f);
                for (std::size_t k = 0; k < 4; ++k)
                    out[4 * i[0] + k] = values[k];
            }).wait();

            std::vector<float> host(4 * blocks);
            for (std::size_t block = 0; block < blocks; ++block)
            {
                const auto values = Philox::normal_block<float>(K_SEED, 9, block, 0.0f, 1.0f);
                for (std::size_t k = 0; k < 4; ++k)
                    host[4 * block + k] = values[k];
            }

            // The integer stream is exact; the normal transform agrees to a few ulps.
            for (std::size_t i = 0; i < host.size(); ++i)
                EXPECT_NEAR(device.get()[i], host[i], 1e-5f * (1.0f + std::fabs(host[i])));
        }
    } // namespace Test
} // namespace SushiBLAS
