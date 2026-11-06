/**************************************************************************/
/* test_gemm_tile_table.cpp                                               */
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
 * @file test_gemm_tile_table.cpp
 * @brief Tests that the measured GEMM tile table holds only entries a chooser can match.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_table.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmTileTable, NoEntryNamesAnUnknownArchitecture)
        {
            for (const Kernels::GemmTileEntry& entry : Kernels::K_GEMM_TILE_TABLE)
                EXPECT_NE(entry.architecture, Device::Architecture::Unknown);
        }

        TEST(Small_GemmTileTable, NoTwoEntriesShareAKey)
        {
            const auto& table = Kernels::K_GEMM_TILE_TABLE;
            for (std::size_t i = 0; i < table.size(); ++i)
                for (std::size_t j = i + 1; j < table.size(); ++j)
                    EXPECT_FALSE(table[i].architecture == table[j].architecture &&
                                 table[i].dtype == table[j].dtype &&
                                 table[i].size_class == table[j].size_class)
                        << "entries " << i << " and " << j;
        }
    } // namespace Test
} // namespace SushiBLAS
