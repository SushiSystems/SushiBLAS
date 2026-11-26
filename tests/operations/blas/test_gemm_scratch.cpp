/**************************************************************************/
/* test_gemm_scratch.cpp                                                  */
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
 * @file test_gemm_scratch.cpp
 * @brief Tests the scratch a GEMM leases: packed panels or the partial sums of depth slices.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_depth_slice_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas/gemm_scratch.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/graph_probe/host_graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rows of C of a weight-gradient product with too few tiles to fill a GPU.
            constexpr int64_t K_ROWS = 256;

            /// @brief Columns of C of that product.
            constexpr int64_t K_COLS = 1152;

            /// @brief The summed extent of that product.
            constexpr int64_t K_DEPTH = 11552;

            /** @brief Returns a GPU profile of 80 units with local memory for the Balanced tile. */
            [[nodiscard]] Device::Profile slicing_profile()
            {
                Device::Profile p;
                p.is_cpu = false;
                p.has_dedicated_local_memory = true;
                p.compute_units = 80;
                p.local_mem_bytes = 48 * 1024;
                p.local_mem_per_compute_unit = 100 * 1024;
                p.max_work_group_size = 1024;
                p.sub_group_size = 32;
                return p;
            }

            /** @brief Returns a CPU profile whose blocking selects the packed product. */
            [[nodiscard]] Device::Profile packing_profile()
            {
                Device::Profile p;
                p.is_cpu = true;
                p.compute_units = 4;
                p.blocking.min_operand_bytes = 0;
                p.blocking.min_register_blocks = 1;
                return p;
            }
        } // namespace

        /** @brief Sizes and leases GEMM scratch on host tensors for profiles the test builds. */
        class Small_GemmScratch : public HostEngineTest
        {

        };

        TEST_F(Small_GemmScratch, AGpuThatSlicesTakesTheBytesOfTheSlicePlan)
        {
            const Kernels::GemmDepthSlicePlan plan = Kernels::plan_gemm_depth_slices<float>(
                slicing_profile(), K_ROWS, K_COLS, K_DEPTH, 1);
            ASSERT_GE(plan.slices, 2);

            EXPECT_EQ(Internal::gemm_scratch_bytes(Core::DataType::Float32, slicing_profile(),
                                                   K_ROWS, K_COLS, K_DEPTH, 1),
                      plan.scratch_bytes);
        }

        TEST_F(Small_GemmScratch, ACpuThatPacksTakesTheBytesOfThePackedPlan)
        {
            const Kernels::PackedProductPlan plan = Kernels::plan_packed_product<float>(
                packing_profile(), K_ROWS, K_COLS, K_DEPTH, 1);
            ASSERT_TRUE(plan.selected);

            EXPECT_EQ(Internal::gemm_scratch_bytes(Core::DataType::Float32, packing_profile(),
                                                   K_ROWS, K_COLS, K_DEPTH, 1),
                      plan.scratch_bytes);
        }

        TEST_F(Small_GemmScratch, HALFOnASlicingGpuNeedsNoBytes)
        {
            EXPECT_EQ(Internal::gemm_scratch_bytes(Core::DataType::Half, slicing_profile(),
                                                   K_ROWS, K_COLS, K_DEPTH, 1),
                      std::size_t{0});
        }

        TEST_F(Small_GemmScratch, TheReferencePathLeasesNothing)
        {
            const Tensor c = engine->create_tensor({K_ROWS, K_COLS});
            HostGraphProbe probe;
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_gemm_scratch(probe.recorder(), c, slicing_profile(),
                                                        K_ROWS, K_COLS, K_DEPTH, 1, operands)
                            .is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
