/**************************************************************************/
/* test_local_tile.cpp                                                    */
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
 * @file test_local_tile.cpp
 * @brief Tests the local tile demand per dtype and the record-time check that a device hosts it.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas/local_tile.hpp>
#include <SushiBLAS/core/errors.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief A tile that stages sixteen elements in a four-item work-group. */
        struct SmallTile
        {
            /** @brief Returns the demand of the tile for element type T. */
            template<typename T>
            [[nodiscard]] static constexpr Device::LocalTileDemand demand() noexcept
            {
                return Device::LocalTileDemand{16 * sizeof(T), 4};
            }
        };

        /** @brief A tile no device hosts: it stages a terabyte of local memory. */
        struct HugeTile
        {
            /** @brief Returns the demand of the tile for element type T. */
            template<typename T>
            [[nodiscard]] static constexpr Device::LocalTileDemand demand() noexcept
            {
                return Device::LocalTileDemand{std::size_t{1} << 40, 4};
            }
        };

        /** @brief Holds the demand of a tile per element type, which needs no device. */
        class Small_LocalTileDemand : public ::testing::Test
        {

        };

        TEST_F(Small_LocalTileDemand, TheSizedDemandFollowsTheElementType)
        {
            EXPECT_EQ(Internal::local_tile_demand<SmallTile>(Core::DataType::Float64).local_bytes,
                      16 * sizeof(double));
            EXPECT_EQ(
                Internal::local_tile_demand<SmallTile>(Core::DataType::Complex32).local_bytes,
                16 * sizeof(std::complex<float>));
        }

        TEST_F(Small_LocalTileDemand, TheDemandIsZeroForIntegers)
        {
            EXPECT_EQ(Internal::local_tile_demand<SmallTile>(Core::DataType::Int32).local_bytes,
                      std::size_t{0});
        }

        /** @brief Holds the record-time decision on a profile value, with no device and no graph. */
        class Small_LocalTileCheck : public ::testing::Test
        {
            protected:
                /** @brief Returns the profile of a device with 64 KiB of local memory. */
                [[nodiscard]] static Device::Profile profile_of_a_device()
                {
                    Device::Profile profile;
                    profile.local_mem_bytes = std::size_t{64} * 1024;
                    profile.max_work_group_size = 256;
                    return profile;
                }
        };

        TEST_F(Small_LocalTileCheck, ATileTheDeviceHostsIsAccepted)
        {
            EXPECT_TRUE(Internal::tile_is_accepted<SmallTile>(
                ExecutionMode::Auto, Core::DataType::Float32, profile_of_a_device()));
        }

        TEST_F(Small_LocalTileCheck, ATileTheDeviceCannotHostIsNotAccepted)
        {
            EXPECT_FALSE(Internal::tile_is_accepted<HugeTile>(
                ExecutionMode::Auto, Core::DataType::Float32, profile_of_a_device()));
        }

        TEST_F(Small_LocalTileCheck, ATileTheDeviceCannotHostIsRefusedWithUnsupportedExecution)
        {
            EXPECT_THROW(Internal::require_local_tile<HugeTile>(
                             ExecutionMode::Auto, Core::DataType::Float32, profile_of_a_device(),
                             "op"),
                         UnsupportedExecution);
        }

        TEST_F(Small_LocalTileCheck, ATileTheDeviceCannotHostIsAcceptedWhenTheReferenceRuns)
        {
            EXPECT_TRUE(Internal::tile_is_accepted<HugeTile>(
                ExecutionMode::ForceReference, Core::DataType::Float32, profile_of_a_device()));
        }
    } // namespace Test
} // namespace SushiBLAS
