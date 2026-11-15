/**************************************************************************/
/* test_rng_position_tensor.cpp                                           */
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
 * @file test_rng_position_tensor.cpp
 * @brief Tests that write_rng_position fills a position tensor and refuses a tensor that cannot hold one.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/random/rng_position_tensor.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the seed the tests write.
            constexpr std::uint64_t K_SEED = 0x0123456789ABCDEFULL;

            /// @brief Holds the offset the tests write.
            constexpr std::uint64_t K_OFFSET = 0xFEDCBA9876543210ULL;

            /// @brief Holds the value an element carries before a write; neither K_SEED nor K_OFFSET.
            constexpr std::uint64_t K_UNWRITTEN = 77;
        } // namespace

        /** @brief Builds position tensors on the fixture's engine. */
        class Medium_RngPositionTensor : public SushiBLASTest
        {
            protected:
                /** @brief Returns a Shared INT64 tensor of count elements, each holding K_UNWRITTEN. */
                [[nodiscard]] Tensor unwritten_position(int64_t count)
                {
                    Tensor position = engine->create_tensor({count}, Core::DataType::Int64);
                    for (int64_t i = 0; i < count; ++i)
                        position.data_as<std::uint64_t>()[i] = K_UNWRITTEN;
                    return position;
                }
        };

        /** @brief Tests that the seed lands in element 0 and the offset in element 1, as raw bits. */
        TEST_F(Medium_RngPositionTensor, WritesTheSeedThenTheOffsetAsRawBits)
        {
            Tensor position = unwritten_position(K_RNG_POSITION_ELEMENTS);

            write_rng_position(position, RNGPosition{K_SEED, K_OFFSET});

            EXPECT_EQ(position.data_as<std::uint64_t>()[0], K_SEED);
            EXPECT_EQ(position.data_as<std::uint64_t>()[1], K_OFFSET);
        }

        /** @brief Tests that a view at a non-zero offset takes the write and its neighbours do not. */
        TEST_F(Medium_RngPositionTensor, WritesThroughAViewIntoALargerAllocation)
        {
            Tensor larger = unwritten_position(4);
            Tensor view = larger.slice(0, 2, 4);

            write_rng_position(view, RNGPosition{K_SEED, K_OFFSET});

            EXPECT_EQ(larger.data_as<std::uint64_t>()[0], K_UNWRITTEN);
            EXPECT_EQ(larger.data_as<std::uint64_t>()[1], K_UNWRITTEN);
            EXPECT_EQ(larger.data_as<std::uint64_t>()[2], K_SEED);
            EXPECT_EQ(larger.data_as<std::uint64_t>()[3], K_OFFSET);
        }

        /** @brief Tests that pinned host memory takes the write, since the host and the device both address it. */
        TEST_F(Medium_RngPositionTensor, WritesIntoHostResidentMemory)
        {
            Tensor position = engine->create_tensor({K_RNG_POSITION_ELEMENTS},
                                                    Core::DataType::Int64,
                                                    SushiRuntime::Residency::Host);

            write_rng_position(position, RNGPosition{K_SEED, K_OFFSET});

            EXPECT_EQ(position.data_as<std::uint64_t>()[0], K_SEED);
            EXPECT_EQ(position.data_as<std::uint64_t>()[1], K_OFFSET);
        }

        /** @brief Tests that a FLOAT32 tensor of two elements is refused. */
        TEST_F(Medium_RngPositionTensor, RefusesAPositionThatIsNotInt64)
        {
            Tensor position = engine->create_tensor({K_RNG_POSITION_ELEMENTS},
                                                    Core::DataType::Float32);

            EXPECT_THROW(write_rng_position(position, RNGPosition{K_SEED, K_OFFSET}),
                         std::invalid_argument);
        }

        /** @brief Tests that INT64 tensors of one and of three elements are refused. */
        TEST_F(Medium_RngPositionTensor, RefusesAnExtentOtherThanTwoElements)
        {
            Tensor one = unwritten_position(1);
            Tensor three = unwritten_position(3);

            EXPECT_THROW(write_rng_position(one, RNGPosition{K_SEED, K_OFFSET}),
                         std::invalid_argument);
            EXPECT_THROW(write_rng_position(three, RNGPosition{K_SEED, K_OFFSET}),
                         std::invalid_argument);
            EXPECT_EQ(three.data_as<std::uint64_t>()[0], K_UNWRITTEN);
        }

        /** @brief Tests that a rank-2 tensor of two elements is refused. */
        TEST_F(Medium_RngPositionTensor, RefusesARankOtherThanOne)
        {
            Tensor position = engine->create_tensor({1, K_RNG_POSITION_ELEMENTS},
                                                    Core::DataType::Int64);

            EXPECT_THROW(write_rng_position(position, RNGPosition{K_SEED, K_OFFSET}),
                         std::invalid_argument);
        }

        /** @brief Tests that device-only memory, which the host cannot address, is refused. */
        TEST_F(Medium_RngPositionTensor, RefusesDeviceOnlyMemory)
        {
            Tensor position = engine->create_tensor({K_RNG_POSITION_ELEMENTS},
                                                    Core::DataType::Int64,
                                                    SushiRuntime::Residency::Device);

            EXPECT_THROW(write_rng_position(position, RNGPosition{K_SEED, K_OFFSET}),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
