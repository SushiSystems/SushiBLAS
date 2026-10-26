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
 * @file test_gemm_tile_catalog.cpp
 * @brief Tests the sizes and the stable names of the tiled GEMM's catalog shapes.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
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

        TEST(Small_GemmTileCatalog, EveryRoleNameReadsBackToItsRole)
        {
            for (const Kernels::GemmTileRole role : Kernels::K_GEMM_TILE_ROLES)
                EXPECT_EQ(Kernels::gemm_tile_role_from(Kernels::to_string(role)), role);
        }

        TEST(Small_GemmTileCatalog, ANameOutsideTheCatalogNamesNoRole)
        {
            EXPECT_FALSE(Kernels::gemm_tile_role_from("Tile64x64x16").has_value());
            EXPECT_FALSE(Kernels::gemm_tile_role_from("").has_value());
            EXPECT_FALSE(Kernels::gemm_tile_role_from("balanced").has_value());
        }

        TEST(Small_GemmTileCatalog, TraitsFollowTheRoleOrder)
        {
            const auto traits = Kernels::gemm_tile_traits<float>();

            for (std::size_t i = 0; i < traits.size(); ++i)
                EXPECT_EQ(traits[i].role, Kernels::K_GEMM_TILE_ROLES[i]);
            EXPECT_EQ(traits[1].block_m, 128);
            EXPECT_EQ(traits[2].work_group_items, std::size_t{64});
        }
    } // namespace Test
} // namespace SushiBLAS
