/**************************************************************************/
/* test_gemm_depth_slice_plan.cpp                                         */
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
 * @file test_gemm_depth_slice_plan.cpp
 * @brief Tests the plan that cuts the K extent of an under-filled tiled GEMM into slices.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_depth_slice_plan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the profile of an RTX 3080 Ti: 80 units, 48 KiB of local memory. */
            [[nodiscard]] Device::Profile sm_86()
            {
                Device::Profile p;
                p.is_cpu = false;
                p.has_dedicated_local_memory = true;
                p.architecture = Device::Architecture::NvidiaSm86;
                p.compute_units = 80;
                p.local_mem_bytes = 48 * 1024;
                p.local_mem_per_compute_unit = 100 * 1024;
                p.max_work_group_size = 1024;
                p.sub_group_size = 32;
                return p;
            }

            /** @brief Returns the profile of a CPU device, which has no dedicated local memory. */
            [[nodiscard]] Device::Profile cpu()
            {
                Device::Profile p;
                p.is_cpu = true;
                p.local_mem_bytes = 256 * 1024;
                p.max_work_group_size = 8192;
                p.compute_units = 12;
                return p;
            }

            /** @brief Returns a FLOAT32 problem on 64 x 64 tiles of depth 16, 80 units of 8 groups. */
            [[nodiscard]] Kernels::GemmDepthSliceProblem float_problem(int64_t m, int64_t n,
                                                                       int64_t k, int64_t batch)
            {
                return Kernels::GemmDepthSliceProblem{m, n, k, batch, 64, 64, 16, 80, 8, 4};
            }
        } // namespace

        /** @brief Holds one problem and the plan the slicing rule gives it by hand. */
        struct SliceRow
        {
            int64_t m;
            int64_t n;
            int64_t k;
            int64_t batch;
            int64_t slices;
            int64_t depth;
            std::size_t scratch_bytes;
        };

        class Small_GemmDepthSlicePlanRow : public ::testing::TestWithParam<SliceRow>
        {
        };

        TEST_P(Small_GemmDepthSlicePlanRow, SlicesFillFourGroupsPerUnitWithWholeKTiles)
        {
            const SliceRow row = GetParam();

            const Kernels::GemmDepthSlicePlan plan =
                Kernels::plan_depth_slices(float_problem(row.m, row.n, row.k, row.batch));

            EXPECT_EQ(plan.slices, row.slices);
            EXPECT_EQ(plan.depth, row.depth);
            EXPECT_EQ(plan.scratch_bytes, row.scratch_bytes);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GemmDepthSlicePlanRow,
            ::testing::Values(SliceRow{256, 1152, 11552, 1, 5, 2320, 5 * 256 * 1152 * 4},
                              SliceRow{64, 64, 184832, 1, 313, 592, 313 * 64 * 64 * 4},
                              SliceRow{64, 64, 512, 1, 2, 256, 2 * 64 * 64 * 4},
                              SliceRow{64, 64, 512, 3, 2, 256, 2 * 3 * 64 * 64 * 4},
                              SliceRow{1024, 4608, 722, 1, 0, 0, 0},
                              SliceRow{64, 64, 184832, 320, 0, 0, 0},
                              SliceRow{64, 64, 300, 1, 0, 0, 0}));

        TEST(Small_GemmDepthSlicePlan, AUnitThatHoldsOneGroupAimsForOneGroupPerUnit)
        {
            Kernels::GemmDepthSliceProblem problem = float_problem(64, 64, 184832, 1);
            problem.groups_per_unit = 1;

            EXPECT_EQ(Kernels::plan_depth_slices(problem).slices, 80);
        }

        TEST(Small_GemmDepthSlicePlan, AWavefrontOf64AndMoreUnitsChangeOnlyTheGroupCount)
        {
            Device::Profile gfx90a = sm_86();
            gfx90a.architecture = Device::Architecture::AmdGfx90a;
            gfx90a.compute_units = 110;
            gfx90a.sub_group_size = 64;
            gfx90a.local_mem_bytes = 64 * 1024;
            gfx90a.local_mem_per_compute_unit = 64 * 1024;

            EXPECT_EQ(Kernels::plan_gemm_depth_slices<float>(gfx90a, 64, 64, 184832, 1).slices,
                      428);
        }

        TEST(Small_GemmDepthSlicePlan, Float32OnAGpuWithLocalMemoryTakesTheBalancedTile)
        {
            const Kernels::GemmDepthSlicePlan plan =
                Kernels::plan_gemm_depth_slices<float>(sm_86(), 256, 1152, 11552, 1);

            EXPECT_EQ(plan.slices, 5);
            EXPECT_EQ(plan.depth, 2320);
        }

        TEST(Small_GemmDepthSlicePlan, ADeviceWithoutDedicatedLocalMemoryIsNotSliced)
        {
            EXPECT_EQ(Kernels::plan_gemm_depth_slices<float>(cpu(), 256, 1152, 11552, 1).slices,
                      0);
        }

        TEST(Small_GemmDepthSlicePlan, HALFIsNotSlicedBecauseItsPartialsWouldRoundToHALF)
        {
            EXPECT_EQ(
                Kernels::plan_gemm_depth_slices<sycl::half>(sm_86(), 256, 1152, 11552, 1).slices,
                0);
        }

        TEST(Small_GemmDepthSlicePlan, ScratchIsReadyOnlyWhenItHoldsEverySlice)
        {
            const Kernels::GemmDepthSlicePlan plan{2, 256, 1024};
            int storage = 0;

            EXPECT_TRUE(Kernels::depth_slices_are_ready(plan, &storage, 1024));
            EXPECT_FALSE(Kernels::depth_slices_are_ready(plan, &storage, 1023));
            EXPECT_FALSE(Kernels::depth_slices_are_ready(plan, nullptr, 1024));
            EXPECT_FALSE(Kernels::depth_slices_are_ready(Kernels::GemmDepthSlicePlan{}, &storage,
                                                         1024));
        }
    } // namespace Test
} // namespace SushiBLAS
