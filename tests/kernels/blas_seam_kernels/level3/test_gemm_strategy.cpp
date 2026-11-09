/**************************************************************************/
/* test_gemm_strategy.cpp                                                 */
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
 * @file test_gemm_strategy.cpp
 * @brief Tests the request that picks the SYCL GEMM kernel.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_strategy.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmStrategy, ADefaultRequestLeavesTheChoiceToTheProfile)
        {
            EXPECT_EQ(Kernels::GemmStrategy{}.kind, Kernels::GemmStrategyKind::Automatic);
            EXPECT_EQ(Kernels::GemmStrategy::automatic().kind, Kernels::GemmStrategyKind::Automatic);
        }

        TEST(Small_GemmStrategy, TheNoLocalRequestNamesTheNoLocalKernel)
        {
            EXPECT_EQ(Kernels::GemmStrategy::no_local().kind, Kernels::GemmStrategyKind::NoLocal);
        }

        TEST(Small_GemmStrategy, TheTiledRequestCarriesTheRoleItWasGiven)
        {
            for (const Kernels::GemmTileRole role : Kernels::K_GEMM_TILE_ROLES)
            {
                const Kernels::GemmStrategy strategy = Kernels::GemmStrategy::local_tiled(role);

                EXPECT_EQ(strategy.kind, Kernels::GemmStrategyKind::LocalTiled);
                EXPECT_EQ(strategy.role, role);
            }
        }

        TEST(Small_GemmStrategy, TheTiledRequestDefaultsToTheBalancedShape)
        {
            EXPECT_EQ(Kernels::GemmStrategy::local_tiled().role, Kernels::GemmTileRole::Balanced);
        }
    } // namespace Test
} // namespace SushiBLAS
