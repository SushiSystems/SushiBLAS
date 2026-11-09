/**************************************************************************/
/* test_gemm_tile_catalog.cpp                                             */
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
 * @file test_gemm_tile_catalog.cpp
 * @brief Tests the sizes and the stable names of the tiled GEMM's catalog shapes.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmTileCatalog, BalancedKeepsTheFloat32GeometryOfTheSingleShapeKernel)
        {
            using Balanced = Kernels::GemmTile<float, Kernels::GemmTileRole::Balanced>;

            EXPECT_EQ(Balanced::K_WG_ITEMS, 64);
            EXPECT_EQ(Balanced::K_BLOCK_M, 64);
            EXPECT_EQ(Balanced::K_BLOCK_N, 64);
            EXPECT_EQ(Balanced::K_TILE_K, 16);
            EXPECT_EQ(Balanced::K_TILE_A_LD, 17);
            EXPECT_EQ(Balanced::K_TILE_B_LD, 66);
            EXPECT_EQ(Balanced::local_mem_bytes(), std::size_t{(64 * 17 + 16 * 66) * 4});
        }

        TEST(Small_GemmTileCatalog, WideDoublesTheRowsAndTheDepthOfBalanced)
        {
            using Wide = Kernels::GemmTile<float, Kernels::GemmTileRole::Wide>;

            EXPECT_EQ(Wide::K_WG_ITEMS, 128);
            EXPECT_EQ(Wide::K_BLOCK_M, 128);
            EXPECT_EQ(Wide::K_BLOCK_N, 64);
            EXPECT_EQ(Wide::K_TILE_K, 32);
            EXPECT_EQ(Wide::K_TILE_A_LD, 33);
        }

        TEST(Small_GemmTileCatalog, CompactHalvesTheBalancedBlockForEveryDtype)
        {
            using Kernels::GemmTile;
            using Kernels::GemmTileRole;

            EXPECT_EQ((GemmTile<float, GemmTileRole::Compact>::K_BLOCK_M), 32);
            EXPECT_EQ((GemmTile<double, GemmTileRole::Balanced>::K_BLOCK_M), 32);
            EXPECT_EQ((GemmTile<double, GemmTileRole::Compact>::K_BLOCK_M), 16);
            EXPECT_EQ((GemmTile<std::complex<double>, GemmTileRole::Balanced>::K_BLOCK_N), 32);
            EXPECT_EQ((GemmTile<std::complex<double>, GemmTileRole::Compact>::K_BLOCK_N), 16);
        }

        /** @brief Runs a catalog case once per tile role. */
        class Small_GemmTileCatalogRoles : public ::testing::TestWithParam<Kernels::GemmTileRole>
        {

        };

        TEST_P(Small_GemmTileCatalogRoles, TheRoleNameReadsBackToItsRole)
        {
            EXPECT_EQ(Kernels::gemm_tile_role_from(Kernels::to_string(GetParam())), GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmTileCatalogRoles,
                                 ::testing::Values(Kernels::GemmTileRole::Balanced,
                                                   Kernels::GemmTileRole::Wide,
                                                   Kernels::GemmTileRole::Compact),
                                 [](const ::testing::TestParamInfo<Kernels::GemmTileRole>& info)
                                 { return std::string(Kernels::to_string(info.param)); });

        TEST(Small_GemmTileCatalog, ANameOutsideTheCatalogNamesNoRole)
        {
            EXPECT_FALSE(Kernels::gemm_tile_role_from("Tile64x64x16").has_value());
            EXPECT_FALSE(Kernels::gemm_tile_role_from("").has_value());
            EXPECT_FALSE(Kernels::gemm_tile_role_from("balanced").has_value());
        }

        /** @brief Reads the traits of the catalog at one position of the role order. */
        class Small_GemmTileCatalogOrder : public ::testing::TestWithParam<std::size_t>
        {

        };

        TEST_P(Small_GemmTileCatalogOrder, TheTraitsFollowTheRoleOrder)
        {
            const auto traits = Kernels::gemm_tile_traits<float>();

            EXPECT_EQ(traits[GetParam()].role, Kernels::K_GEMM_TILE_ROLES[GetParam()]);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GemmTileCatalogOrder,
                                 ::testing::Values(std::size_t{0}, std::size_t{1}, std::size_t{2}),
                                 [](const ::testing::TestParamInfo<std::size_t>& info)
                                 { return "Position" + std::to_string(info.param); });

        TEST(Small_GemmTileCatalog, TheTraitsHoldTheBlockOfWide)
        {
            const auto traits = Kernels::gemm_tile_traits<float>();

            EXPECT_EQ(traits[1].block_m, 128);
        }

        TEST(Small_GemmTileCatalog, TheTraitsHoldTheWorkGroupOfCompact)
        {
            const auto traits = Kernels::gemm_tile_traits<float>();

            EXPECT_EQ(traits[2].work_group_items, std::size_t{64});
        }
    } // namespace Test
} // namespace SushiBLAS
