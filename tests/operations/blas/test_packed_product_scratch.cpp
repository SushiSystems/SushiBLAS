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
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/graph_probe/host_graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The extent of every axis of the product; large enough for a plan to be chosen.
        constexpr int64_t K_EXTENT = 512;

        /// @brief The extent of every axis of a product that stays below every packing threshold.
        constexpr int64_t K_SMALL_EXTENT = 8;

        /** @brief Leases packed-product scratch on host tensors for a profile the test builds. */
        class Small_PackedProductScratch : public HostEngineTest
        {
            protected:
                /** @brief Returns a CPU profile whose blocking selects a product of K_EXTENT. */
                [[nodiscard]] static Device::Profile selecting_profile()
                {
                    Device::Profile profile;
                    profile.is_cpu = true;
                    profile.compute_units = 4;
                    profile.blocking.min_operand_bytes = 0;
                    profile.blocking.min_register_blocks = 1;
                    return profile;
                }

                /** @brief Returns a profile that never selects the packed kernel: not a CPU. */
                [[nodiscard]] static Device::Profile unselecting_profile()
                {
                    Device::Profile profile = selecting_profile();
                    profile.is_cpu = false;
                    return profile;
                }

                /** @brief Returns the plan the selecting profile makes for the product. */
                [[nodiscard]] static Kernels::PackedProductPlan selected_plan()
                {
                    return Kernels::plan_packed_product<float>(selecting_profile(), K_EXTENT,
                                                               K_EXTENT, K_EXTENT, 1);
                }
        };

        TEST_F(Small_PackedProductScratch, TheBytesAreThoseOfTheSelectedPlan)
        {
            const Kernels::PackedProductPlan plan = selected_plan();
            ASSERT_TRUE(plan.selected);

            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Float32,
                                                             selecting_profile(), K_EXTENT,
                                                             K_EXTENT, K_EXTENT, 1),
                      plan.scratch_bytes);
        }

        TEST_F(Small_PackedProductScratch, TheBytesAreNoneWhenTheProfileSelectsNoPlan)
        {
            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Float32,
                                                             unselecting_profile(), K_EXTENT,
                                                             K_EXTENT, K_EXTENT, 1),
                      std::size_t{0});
        }

        TEST_F(Small_PackedProductScratch, AnIntegerDtypeNeedsNoBytes)
        {
            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Int32,
                                                             selecting_profile(), K_EXTENT,
                                                             K_EXTENT, K_EXTENT, 1),
                      std::size_t{0});
        }

        TEST_F(Small_PackedProductScratch, TheLeaseHoldsThePlannedBytesAndIsDeclared)
        {
            const Tensor c = engine->create_tensor({K_EXTENT, K_EXTENT});
            HostGraphProbe probe;
            probe.recorder().set_mode(ExecutionMode::Auto);
            OperandSet operands;

            const ScratchBuffer scratch = Internal::allocate_packed_product_scratch(
                probe.recorder(), c, selecting_profile(), K_EXTENT, K_EXTENT, K_EXTENT, 1,
                operands);

            EXPECT_EQ(scratch.bytes(), selected_plan().scratch_bytes);
            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
        }

        TEST_F(Small_PackedProductScratch, NothingIsLeasedWhenTheProfileSelectsNoPlan)
        {
            const Tensor c = engine->create_tensor({K_EXTENT, K_EXTENT});
            HostGraphProbe probe;
            probe.recorder().set_mode(ExecutionMode::Auto);
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_packed_product_scratch(
                            probe.recorder(), c, unselecting_profile(), K_EXTENT, K_EXTENT,
                            K_EXTENT, 1, operands)
                            .is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Small_PackedProductScratch, TheReferencePathLeasesNothing)
        {
            const Tensor c = engine->create_tensor({K_EXTENT, K_EXTENT});
            HostGraphProbe probe;
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_packed_product_scratch(
                            probe.recorder(), c, selecting_profile(), K_EXTENT, K_EXTENT,
                            K_EXTENT, 1, operands)
                            .is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Small_PackedProductScratch, AProductBelowThePackingThresholdsKeepsTheUnpackedKernel)
        {
            Device::Profile profile;
            profile.is_cpu = true;
            profile.compute_units = 4;

            EXPECT_FALSE(Kernels::plan_packed_product<float>(profile, K_SMALL_EXTENT,
                                                             K_SMALL_EXTENT, K_SMALL_EXTENT, 1)
                             .selected);
            EXPECT_EQ(Internal::packed_product_scratch_bytes(Core::DataType::Float32, profile,
                                                             K_SMALL_EXTENT, K_SMALL_EXTENT,
                                                             K_SMALL_EXTENT, 1),
                      std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
