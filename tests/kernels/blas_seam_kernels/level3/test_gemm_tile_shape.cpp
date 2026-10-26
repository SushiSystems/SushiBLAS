/**************************************************************************/
/* test_gemm_tile_shape.cpp                                               */
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
 * @file test_gemm_tile_shape.cpp
 * @brief Tests the sizes GemmTileShape derives from its four parameters.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_shape.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmTileShape, TheRegisterBlockIsTheLargestSquareThatFitsTheBudget)
        {
            EXPECT_EQ(Kernels::register_block_dim<float>(256, 8), 8);
            EXPECT_EQ(Kernels::register_block_dim<double>(256, 8), 4);
            EXPECT_EQ(Kernels::register_block_dim<std::complex<double>>(256, 8), 4);
            EXPECT_EQ(Kernels::register_block_dim<float>(64, 8), 4);
            EXPECT_EQ(Kernels::register_block_dim<float>(1024, 8), 8);
            EXPECT_EQ(Kernels::register_block_dim<double>(4, 8), 1);
        }

        TEST(Small_GemmTileShape, HALFTakesTheRegisterBlockOfItsFloat32Accumulator)
        {
            using Shape = Kernels::GemmTileShape<sycl::half, 8, 8, 256, 16>;

            EXPECT_EQ(Shape::K_ITEM_ROWS, 8);
            EXPECT_EQ(Shape::local_mem_bytes(), std::size_t{(64 * 17 + 16 * 66) * 2});
        }

        TEST(Small_GemmTileShape, BlocksAndPaddingFollowTheWorkGroupAndTheDepth)
        {
            using Shape = Kernels::GemmTileShape<double, 4, 16, 128, 8>;

            EXPECT_EQ(Shape::K_WG_ITEMS, 64);
            EXPECT_EQ(Shape::K_ITEM_ROWS, 4);
            EXPECT_EQ(Shape::K_BLOCK_M, 16);
            EXPECT_EQ(Shape::K_BLOCK_N, 64);
            EXPECT_EQ(Shape::K_TILE_A_LD, 9);
            EXPECT_EQ(Shape::K_TILE_B_LD, 66);
            EXPECT_EQ(Shape::local_mem_bytes(), std::size_t{(16 * 9 + 8 * 66) * 8});
        }
    } // namespace Test
} // namespace SushiBLAS
