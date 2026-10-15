/**************************************************************************/
/* test_packed_product_scratch.cpp                                        */
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
 * @file test_packed_product_scratch.cpp
 * @brief Tests that the packed product leases the scratch its plan names, and none without a plan.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas/packed_product_scratch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The extent of every axis of the product; large enough for a plan to be chosen.
        constexpr int64_t K_EXTENT = 512;

        /** @brief Builds output matrices on the fixture's engine. */
        class Medium_PackedProductScratch : public SushiBLASTest
        {

        };

        TEST_F(Medium_PackedProductScratch, TheBytesAreThoseOfThePlanOrNone)
        {
            const Device::Profile& profile =
                Device::profile_for(engine->create_tensor({1}).device());
            const Kernels::PackedProductPlan plan =
                Kernels::plan_packed_product<float>(profile, K_EXTENT, K_EXTENT, K_EXTENT, 1);

            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Float32, profile,
                                                             K_EXTENT, K_EXTENT, K_EXTENT, 1),
                      plan.selected ? plan.scratch_bytes : std::size_t{0});
            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Int32, profile,
                                                             K_EXTENT, K_EXTENT, K_EXTENT, 1),
                      std::size_t{0});
        }

        TEST_F(Medium_PackedProductScratch, TheLeaseHoldsThePlannedBytesAndIsDeclared)
        {
            const Tensor c = engine->create_tensor({K_EXTENT, K_EXTENT});
            const std::size_t bytes = Internal::packed_product_scratch_bytes(
                c.dtype, Device::profile_for(c.device()), K_EXTENT, K_EXTENT, K_EXTENT, 1);
            GraphProbe probe;
            OperandSet operands;

            const ScratchBuffer scratch = Internal::allocate_packed_product_scratch(
                probe.recorder(), c, K_EXTENT, K_EXTENT, K_EXTENT, 1, operands);

            EXPECT_EQ(scratch.bytes(), bytes);
            const std::size_t declared = bytes == 0 ? 0 : 1;
            EXPECT_EQ(operands.write_regions().size(), declared);
        }

        TEST_F(Medium_PackedProductScratch, TheReferencePathLeasesNothing)
        {
            const Tensor c = engine->create_tensor({K_EXTENT, K_EXTENT});
            GraphProbe probe(ExecutionMode::ForceReference);
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_packed_product_scratch(probe.recorder(), c, K_EXTENT,
                                                                  K_EXTENT, K_EXTENT, 1, operands)
                            .is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
