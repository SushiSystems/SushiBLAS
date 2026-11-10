/**************************************************************************/
/* test_triangular_plan.cpp                                               */
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
 * @file test_triangular_plan.cpp
 * @brief Tests the block plan of a triangular solve on a CPU and a GPU profile.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the compute units of the hand-built CPU profile.
            constexpr std::size_t K_CPU_COMPUTE_UNITS = 16;

            /// @brief Holds the compute units of the hand-built GPU profile.
            constexpr std::size_t K_GPU_COMPUTE_UNITS = 80;

            /// @brief Holds the work-group size the hand-built GPU profile launches with.
            constexpr std::size_t K_GPU_WORK_GROUP = 256;

            /** @brief Holds the block one planned solve is expected to take. */
            struct PlanRow
            {
                int64_t order;   /**< Steps of one substitution. */
                int64_t streams; /**< Independent substitutions. */
                int64_t block;   /**< The expected steps of one diagonal block. */
            };

            /** @brief Holds the block a one-vector solve is expected to take. */
            struct VectorRow
            {
                int64_t order; /**< Steps of the substitution. */
                int64_t reach; /**< Rows one block updates. */
                int64_t block; /**< The expected steps of one diagonal block. */
            };

            /** @brief Holds a requested block, a system order and the value the plan returns. */
            struct ClampRow
            {
                int64_t block; /**< The requested steps of one block. */
                int64_t order; /**< Steps of the system. */
                int64_t value; /**< The expected width or count. */
            };

            /** @brief Returns a CPU profile of compute_units hardware threads. */
            [[nodiscard]] Device::Profile cpu_profile(std::size_t compute_units) noexcept
            {
                Device::Profile profile;
                profile.is_cpu = true;
                profile.compute_units = compute_units;
                return profile;
            }

            /** @brief Returns a GPU profile whose work-group size is K_GPU_WORK_GROUP. */
            [[nodiscard]] Device::Profile gpu_profile() noexcept
            {
                Device::Profile profile;
                profile.is_cpu = false;
                profile.compute_units = K_GPU_COMPUTE_UNITS;
                profile.max_work_group_size = 1024;
                profile.sub_group_size = 32;
                return profile;
            }

            /// @brief Names a row after its order and its streams.
            std::string plan_row_name(const ::testing::TestParamInfo<PlanRow>& info)
            {
                return "Order" + std::to_string(info.param.order) + "Streams" +
                       std::to_string(info.param.streams);
            }

            /// @brief Names a row after its order and its reach.
            std::string vector_row_name(const ::testing::TestParamInfo<VectorRow>& info)
            {
                return "Order" + std::to_string(info.param.order) + "Reach" +
                       std::to_string(info.param.reach);
            }

            /// @brief Names a row after its requested block and its order.
            std::string clamp_row_name(const ::testing::TestParamInfo<ClampRow>& info)
            {
                const int64_t block = info.param.block;
                const std::string sign = block < 0 ? "Minus" : "";
                return "Block" + sign + std::to_string(block < 0 ? -block : block) + "Order" +
                       std::to_string(info.param.order);
            }

            /// @brief Plans a row on a profile and expects the block of the row and its width.
            void expect_planned_block(const Device::Profile& profile, const PlanRow& row)
            {
                const Kernels::SYCL::TriangularSolvePlan plan =
                    Kernels::SYCL::plan_triangular_solve(profile, row.order, row.streams);

                EXPECT_EQ(row.block, plan.block);
                EXPECT_EQ(row.block, plan.block_width(row.order));
            }
        } // namespace

        /// @brief Plans the solves of a CPU of 16 threads, one order and stream count at a time.
        class Small_TriangularPlanCpu : public ::testing::TestWithParam<PlanRow> {};

        /// @brief Tests that a CPU blocks exactly when the order exceeds its break-even, 546 at one stream (546 * 15 = 8190 <= 8192 < 547 * 15).
        TEST_P(Small_TriangularPlanCpu, ACPUBlocksOnlyWhereItPays)
        {
            expect_planned_block(cpu_profile(K_CPU_COMPUTE_UNITS), GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(OrdersAndStreams, Small_TriangularPlanCpu,
                                 ::testing::Values(PlanRow{512, 1, 512}, PlanRow{546, 1, 546},
                                                   PlanRow{1024, 8, 1024}, PlanRow{4096, 16, 4096},
                                                   PlanRow{547, 1, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                                   PlanRow{4096, 1, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                                   PlanRow{1025, 8, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                                   PlanRow{4096, 8, Kernels::SYCL::K_CPU_SOLVE_BLOCK}),
                                 plan_row_name);

        /// @brief Plans the solves of a CPU of one compute unit, one order at a time.
        class Small_TriangularPlanOneUnit : public ::testing::TestWithParam<PlanRow> {};

        /// @brief Tests that a device of one compute unit plans every order whole.
        TEST_P(Small_TriangularPlanOneUnit, OneComputeUnitNeverBlocks)
        {
            expect_planned_block(cpu_profile(1), GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(Orders, Small_TriangularPlanOneUnit,
                                 ::testing::Values(PlanRow{1, 1, 1}, PlanRow{1000, 1, 1000},
                                                   PlanRow{1000000, 1, 1000000}),
                                 plan_row_name);

        /// @brief Plans the solves of a GPU, one order and stream count at a time.
        class Small_TriangularPlanGpu : public ::testing::TestWithParam<PlanRow> {};

        /// @brief Tests that a GPU blocks exactly when the order exceeds its break-even for the streams.
        TEST_P(Small_TriangularPlanGpu, AGPUBlocksAtItsOwnWidth)
        {
            expect_planned_block(gpu_profile(), GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(OrdersAndStreams, Small_TriangularPlanGpu,
                                 ::testing::Values(PlanRow{257, 1, Kernels::SYCL::K_GPU_SOLVE_BLOCK},
                                                   PlanRow{4096, 1000, Kernels::SYCL::K_GPU_SOLVE_BLOCK},
                                                   PlanRow{256, 1, 256}, PlanRow{4096, 200000, 4096}),
                                 plan_row_name);

        /// @brief Plans one-vector solves on a GPU, one order and reach at a time.
        class Small_TriangularPlanVectorBlocked : public ::testing::TestWithParam<VectorRow> {};

        /// @brief Tests that a one-vector solve blocks at the device block when the reach is wider than it.
        TEST_P(Small_TriangularPlanVectorBlocked, AVectorSolveBlocksWhenTheReachIsWiderThanTheBlock)
        {
            EXPECT_EQ(GetParam().block,
                      Kernels::SYCL::plan_vector_solve(gpu_profile(), GetParam().order,
                                                       GetParam().reach).block);
        }

        INSTANTIATE_TEST_SUITE_P(Reaches, Small_TriangularPlanVectorBlocked,
                                 ::testing::Values(VectorRow{8192, 8192, Kernels::SYCL::K_GPU_SOLVE_BLOCK},
                                                   VectorRow{8192, 257, Kernels::SYCL::K_GPU_SOLVE_BLOCK}),
                                 vector_row_name);

        /// @brief Plans one-vector solves on a GPU, one order and reach at a time.
        class Small_TriangularPlanVectorWhole : public ::testing::TestWithParam<VectorRow> {};

        /// @brief Tests that a reach of zero, or a block at or over the reach, plans the whole order.
        TEST_P(Small_TriangularPlanVectorWhole, AVectorSolveStaysWholeWhenTheReachFitsTheBlock)
        {
            EXPECT_EQ(GetParam().block,
                      Kernels::SYCL::plan_vector_solve(gpu_profile(), GetParam().order,
                                                       GetParam().reach).block);
        }

        INSTANTIATE_TEST_SUITE_P(Reaches, Small_TriangularPlanVectorWhole,
                                 ::testing::Values(VectorRow{8192, 256, 8192}, VectorRow{8192, 0, 8192},
                                                   VectorRow{256, 256, 256}, VectorRow{0, 0, 1}),
                                 vector_row_name);

        /// @brief Tests that the resident work-items of a CPU are its threads.
        TEST(Small_TriangularPlanResident, ACPUHasOneWorkItemPerThread)
        {
            EXPECT_EQ(K_CPU_COMPUTE_UNITS,
                      Kernels::SYCL::resident_work_items(cpu_profile(K_CPU_COMPUTE_UNITS)));
        }

        /// @brief Tests that the resident work-items of a GPU are its compute units times the waves times the work-group.
        TEST(Small_TriangularPlanResident, AGPUHasTheGridOfItsWaves)
        {
            EXPECT_EQ(K_GPU_COMPUTE_UNITS * Device::K_WAVES_PER_COMPUTE_UNIT * K_GPU_WORK_GROUP,
                      Kernels::SYCL::resident_work_items(gpu_profile()));
        }

        /// @brief Clamps a requested block to a system, one block and order at a time.
        class Small_TriangularPlanClampWidth : public ::testing::TestWithParam<ClampRow> {};

        /// @brief Tests that a block is at least one step and at most the whole system.
        TEST_P(Small_TriangularPlanClampWidth, ABlockIsClampedToTheSystem)
        {
            EXPECT_EQ(GetParam().value,
                      Kernels::SYCL::TriangularSolvePlan{GetParam().block}.block_width(GetParam().order));
        }

        INSTANTIATE_TEST_SUITE_P(Blocks, Small_TriangularPlanClampWidth,
                                 ::testing::Values(ClampRow{64, 200, 64}, ClampRow{1000, 200, 200},
                                                   ClampRow{0, 200, 1}, ClampRow{-5, 200, 1},
                                                   ClampRow{64, 0, 1}),
                                 clamp_row_name);

        /// @brief Counts the blocks of a system, one block and order at a time.
        class Small_TriangularPlanClampCount : public ::testing::TestWithParam<ClampRow> {};

        /// @brief Tests that the block count covers the order and is at least one.
        TEST_P(Small_TriangularPlanClampCount, TheBlocksCoverTheSystem)
        {
            EXPECT_EQ(GetParam().value,
                      Kernels::SYCL::TriangularSolvePlan{GetParam().block}.block_count(GetParam().order));
        }

        INSTANTIATE_TEST_SUITE_P(Blocks, Small_TriangularPlanClampCount,
                                 ::testing::Values(ClampRow{64, 200, 4}, ClampRow{1000, 200, 1},
                                                   ClampRow{0, 200, 200}, ClampRow{-5, 200, 200},
                                                   ClampRow{64, 0, 1}),
                                 clamp_row_name);
    } // namespace Test
} // namespace SushiBLAS
