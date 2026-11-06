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
 * @file test_philox.cpp
 * @brief Tests Philox4x32-10 against its known answers and the draws built on it, host and device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/philox.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace Philox = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 20261015u;

        /// @brief The rounded operations on the longest path of a float normal draw: the logarithm, the root, the sine or cosine and the scaling.
        constexpr std::size_t K_NORMAL_OPERATIONS = 16;

        /// @brief Returns blocks from the first to ones past the 32-bit counter word and the last.
        static std::vector<uint64_t> block_rows()
        {
            return {0, 1, 2, 7, 63, 99, 4096, uint64_t{1} << 32, std::numeric_limits<uint64_t>::max()};
        }

        /// @brief Names a block row by its number, which gtest takes as the test's suffix.
        static std::string block_row_name(const ::testing::TestParamInfo<uint64_t>& info)
        {
            return "Block" + std::to_string(info.param);
        }

        TEST(Small_Philox, MatchesTheKnownAnswersOfRandom123)
        {
            using Block = std::array<uint32_t, 4>;
            const uint32_t ones = 0xFFFFFFFFu;

            EXPECT_EQ(Philox::philox4x32_10(0, 0, 0, 0, 0, 0),
                      (Block{0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u}));
            EXPECT_EQ(Philox::philox4x32_10(ones, ones, ones, ones, ones, ones),
                      (Block{0x408F276Du, 0x41C83B0Eu, 0xA20BC7C6u, 0x6D5451FDu}));
        }

        TEST(Small_Philox, TheLowestAndHighestWordsMapToTheEndsOfTheUnitRange)
        {
            EXPECT_EQ(Philox::uniform01_float(0u), 0.0f);
            EXPECT_LT(Philox::uniform01_float(0xFFFFFFFFu), 1.0f);
            EXPECT_LT(Philox::uniform01_double(0xFFFFFFFFu, 0xFFFFFFFFu), 1.0);
        }

        TEST(Small_Philox, ARangeOfOnePointFillsEveryDrawWithIt)
        {
            const auto values = Philox::uniform_block<float>(K_SEED, 0, 3, 1.5f, 1.5f);

            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 1.5f; }));
        }

        /// @brief Draws the uniform values of one block.
        class Small_PhiloxUniformBlocks : public ::testing::TestWithParam<uint64_t>
        {
        };

        TEST_P(Small_PhiloxUniformBlocks, UniformDrawsLieInTheirHalfOpenRange)
        {
            constexpr float K_LOW = -2.0f;
            constexpr float K_HIGH = 3.0f;

            const auto values = Philox::uniform_block<float>(K_SEED, 0, GetParam(), K_LOW, K_HIGH);

            EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                                    [](float value) { return value >= K_LOW && value < K_HIGH; }));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PhiloxUniformBlocks, ::testing::ValuesIn(block_rows()),
                                 block_row_name);

        TEST(Small_Philox, TheDrawIsAFunctionOfSeedOffsetAndBlock)
        {
            EXPECT_EQ(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 3, 7));
            EXPECT_NE(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 4, 7));
            EXPECT_NE(Philox::philox_block(K_SEED, 3, 7), Philox::philox_block(K_SEED, 3, 8));
        }

        TEST(Small_Philox, ABoundOfZeroOrOneGivesZero)
        {
            EXPECT_EQ(Philox::draw_index(K_SEED, 0, 5, 1), uint64_t{0});
            EXPECT_EQ(Philox::draw_index(K_SEED, 0, 5, 0), uint64_t{0});
        }

        /// @brief One index and the bound it is drawn under.
        struct IndexRow
        {
                const char* name;
                uint64_t index;
                uint64_t bound;
        };

        /// @brief Returns small and large indices under small and large bounds.
        static std::vector<IndexRow> index_rows()
        {
            return {{"First_Under13", 0, 13},
                    {"Second_Under13", 1, 13},
                    {"Seventh_Under13", 7, 13},
                    {"TwoHundredth_Under13", 199, 13},
                    {"Fifth_Under2", 5, 2},
                    {"Fifth_UnderALargePrime", 5, 1000000007u},
                    {"LastIndex_Under13", std::numeric_limits<uint64_t>::max(), 13}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string index_row_name(const ::testing::TestParamInfo<IndexRow>& info)
        {
            return info.param.name;
        }

        /// @brief Draws one index under one bound.
        class Small_PhiloxIndices : public ::testing::TestWithParam<IndexRow>
        {
        };

        TEST_P(Small_PhiloxIndices, AnIndexStaysBelowItsBound)
        {
            const IndexRow& row = GetParam();

            EXPECT_LT(Philox::draw_index(K_SEED, 0, row.index, row.bound), row.bound);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PhiloxIndices, ::testing::ValuesIn(index_rows()),
                                 index_row_name);

        /// @brief Draws the normal values of one block.
        class Small_PhiloxNormalBlocks : public ::testing::TestWithParam<uint64_t>
        {
        };

        TEST_P(Small_PhiloxNormalBlocks, NormalDrawsAreFinite)
        {
            const auto values = Philox::normal_block<double>(K_SEED, 0, GetParam(), 1.0, 2.0);

            EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                                    [](double value) { return std::isfinite(value); }));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PhiloxNormalBlocks, ::testing::ValuesIn(block_rows()),
                                 block_row_name);

        namespace
        {
            /// @brief Returns the normal draws of the first blocks, four floats per block, as the host draws them.
            std::vector<float> host_normal_draws(std::size_t blocks)
            {
                std::vector<float> host;
                for (std::size_t block = 0; block < blocks; ++block)
                {
                    const auto values = Philox::normal_block<float>(K_SEED, 9, block, 0.0f, 1.0f);
                    host.insert(host.end(), values.begin(), values.end());
                }
                return host;
            }

            /// @brief Returns the normal draws of the first blocks, four floats per block, as a kernel draws them.
            std::vector<float> device_normal_draws(sycl::queue& queue, std::size_t blocks)
            {
                SharedArray<float> device(queue, 4 * blocks);
                float* out = device.get();

                queue.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> i)
                {
                    const auto values = Philox::normal_block<float>(K_SEED, 9, i[0], 0.0f, 1.0f);
                    for (std::size_t k = 0; k < 4; ++k)
                        out[4 * i[0] + k] = values[k];
                }).wait();

                return device.values();
            }

            /**
             * @brief Expects each device value to match the host value within the rounding of the normal transform.
             *
             * The integer stream is exact; the logarithm, the root and the sine or cosine of the
             * transform round on the device as they may on the host, so the bound is relative to
             * the value, with one unit added for a value near zero.
             */
            void expect_within_the_normal_transform(const std::vector<float>& device,
                                                    const std::vector<float>& host)
            {
                ASSERT_EQ(device.size(), host.size());
                for (std::size_t i = 0; i < host.size(); ++i)
                {
                    const double bound = static_cast<double>(K_NORMAL_OPERATIONS) *
                                         TestSupport::unit_roundoff<float>() *
                                         (1.0 + std::fabs(host[i]));
                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(device[i], host[i], bound)))
                        << "element " << i;
                }
            }
        } // namespace

        /** @brief Draws on the host CPU queue. */
        class Medium_PhiloxOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_PhiloxOnDevice, AKernelDrawsTheBitsTheHostDraws)
        {
            constexpr std::size_t blocks = 64;

            expect_within_the_normal_transform(device_normal_draws(queue(), blocks),
                                               host_normal_draws(blocks));
        }
    } // namespace Test
} // namespace SushiBLAS
