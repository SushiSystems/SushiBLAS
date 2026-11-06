/**************************************************************************/
/* test_gmres_workspace.cpp                                               */
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
 * @file test_gmres_workspace.cpp
 * @brief Tests that GmresLayout places every array of the workspace after the one before it.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GmresLayout, PlacesTheArraysBackToBack)
        {
            const Kernels::GmresLayout layout(10, 4, 2);

            EXPECT_EQ(layout.basis, 0u);
            EXPECT_EQ(layout.hessenberg, 5u * 10u);
            EXPECT_EQ(layout.cosines, layout.hessenberg + 5u * 4u);
            EXPECT_EQ(layout.sines, layout.cosines + 4u);
            EXPECT_EQ(layout.projection, layout.sines + 4u);
            EXPECT_EQ(layout.coefficients, layout.projection + 5u);
            EXPECT_EQ(layout.residual, layout.coefficients + 4u);
            EXPECT_EQ(layout.work, layout.residual + 10u);
            EXPECT_EQ(layout.preconditioned, layout.work + 10u);
            EXPECT_EQ(layout.factors, layout.preconditioned + 10u);
            EXPECT_EQ(layout.partials, layout.factors + 10u * 2u);
            EXPECT_EQ(layout.scalars, layout.partials + Kernels::dot_chunks(10u));
        }

        TEST(Small_GmresLayout, CountsThePivotsAfterTheScalars)
        {
            const Kernels::GmresLayout layout(6, 2, 3);

            EXPECT_EQ(layout.bytes(sizeof(double)),
                      layout.scalars * sizeof(double) + 6u * sizeof(std::uint32_t));
        }

        TEST(Small_GmresSettings, DefaultsMatchTheEngineContract)
        {
            const Kernels::GmresSettings settings;

            EXPECT_EQ(settings.restart, 30u);
            EXPECT_EQ(settings.max_iterations, 300u);
            EXPECT_EQ(settings.relative_tolerance, 1e-6);
            EXPECT_EQ(settings.block_size, 1u);
        }
    } // namespace Test
} // namespace SushiBLAS
