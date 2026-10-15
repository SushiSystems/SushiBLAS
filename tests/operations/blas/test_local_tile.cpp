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
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

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

        /** @brief Builds the output matrices on the fixture's engine. */
        class Medium_LocalTile : public SushiBLASTest
        {

        };

        TEST_F(Medium_LocalTile, TheDemandFollowsTheElementTypeAndIsZeroForIntegers)
        {
            EXPECT_EQ(Internal::local_tile_demand<SmallTile>(Core::DataType::Float64).local_bytes,
                      16 * sizeof(double));
            EXPECT_EQ(
                Internal::local_tile_demand<SmallTile>(Core::DataType::Complex32).local_bytes,
                16 * sizeof(std::complex<float>));
            EXPECT_EQ(Internal::local_tile_demand<SmallTile>(Core::DataType::Int32).local_bytes,
                      std::size_t{0});
        }

        TEST_F(Medium_LocalTile, ATileTheDeviceCannotHostIsRefusedUnlessTheReferenceRuns)
        {
            const Tensor c = engine->create_tensor({4, 4});
            GraphProbe automatic;
            GraphProbe reference(ExecutionMode::ForceReference);

            EXPECT_NO_THROW(Internal::require_local_tile<SmallTile>(automatic.recorder(), c, "op"));
            EXPECT_ANY_THROW(Internal::require_local_tile<HugeTile>(automatic.recorder(), c, "op"));
            EXPECT_NO_THROW(Internal::require_local_tile<HugeTile>(reference.recorder(), c, "op"));
        }
    } // namespace Test
} // namespace SushiBLAS
