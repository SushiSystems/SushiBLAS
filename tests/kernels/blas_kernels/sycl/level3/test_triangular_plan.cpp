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
 * @file test_triangular_plan.cpp
 * @brief Tests the block plan of a triangular solve on a CPU and a GPU profile.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
            struct PlanExpectation
            {
                int64_t order;   /**< Steps of one substitution. */
                int64_t streams; /**< Independent substitutions. */
                int64_t block;   /**< The expected steps of one diagonal block. */
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

            /** @brief Expects that a profile plans each case with its expected block. */
            void expect_planned_blocks(const Device::Profile& profile,
                                       std::initializer_list<PlanExpectation> expectations)
            {
                for (const PlanExpectation& expected : expectations)
                {
                    SCOPED_TRACE(::testing::Message() << "order " << expected.order
                                                      << ", streams " << expected.streams);

                    const Kernels::SYCL::TriangularSolvePlan plan =
                        Kernels::SYCL::plan_triangular_solve(profile, expected.order,
                                                             expected.streams);

                    EXPECT_EQ(expected.block, plan.block);
                    EXPECT_EQ(expected.block, plan.block_width(expected.order));
                }
            }

        } // namespace

        /**
         * @brief Tests that a CPU blocks a solve only above its break-even order and below a
         *        full set of streams.
         */
        TEST(Medium_TriangularPlan, ACPUBlocksOnlyWhereItPays)
        {
            const Device::Profile profile = cpu_profile(K_CPU_COMPUTE_UNITS);

            expect_planned_blocks(profile, {{512, 1, 512},
                                            {546, 1, 546},
                                            {1024, 8, 1024},
                                            {4096, 16, 4096}});

            expect_planned_blocks(profile,
                                  {{547, 1, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                   {4096, 1, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                   {1025, 8, Kernels::SYCL::K_CPU_SOLVE_BLOCK},
                                   {4096, 8, Kernels::SYCL::K_CPU_SOLVE_BLOCK}});

            EXPECT_EQ(int64_t{256}, Kernels::SYCL::K_CPU_SOLVE_BLOCK);
        }

        /**
         * @brief Tests that a device of one compute unit solves every order in one block.
         */
        TEST(Medium_TriangularPlan, OneComputeUnitNeverBlocks)
        {
            expect_planned_blocks(cpu_profile(1), {{1, 1, 1},
                                                   {1000, 1, 1000},
                                                   {1000000, 1, 1000000}});
        }

        /**
         * @brief Tests that a GPU blocks at its own width, between its two thresholds.
         */
        TEST(Medium_TriangularPlan, AGPUBlocksAtItsOwnWidth)
        {
            const Device::Profile profile = gpu_profile();

            expect_planned_blocks(profile,
                                  {{257, 1, Kernels::SYCL::K_GPU_SOLVE_BLOCK},
                                   {4096, 1000, Kernels::SYCL::K_GPU_SOLVE_BLOCK}});

            expect_planned_blocks(profile, {{256, 1, 256}, {4096, 200000, 4096}});

            EXPECT_EQ(int64_t{128}, Kernels::SYCL::K_GPU_SOLVE_BLOCK);
        }

        /**
         * @brief Tests that a one-vector solve blocks only when one block updates enough rows.
         */
        TEST(Medium_TriangularPlan, AVectorSolveBlocksByItsReach)
        {
            using Kernels::SYCL::K_GPU_SOLVE_BLOCK;
            using Kernels::SYCL::plan_vector_solve;
            const Device::Profile profile = gpu_profile();

            EXPECT_EQ(K_GPU_SOLVE_BLOCK, plan_vector_solve(profile, 8192, 8192).block);
            EXPECT_EQ(K_GPU_SOLVE_BLOCK, plan_vector_solve(profile, 8192, 257).block);

            EXPECT_EQ(int64_t{8192}, plan_vector_solve(profile, 8192, 256).block);
            EXPECT_EQ(int64_t{8192}, plan_vector_solve(profile, 8192, 0).block);
            EXPECT_EQ(int64_t{256}, plan_vector_solve(profile, 256, 256).block);
            EXPECT_EQ(int64_t{1}, plan_vector_solve(profile, 0, 0).block);
        }

        /**
         * @brief Tests that the resident work-items are the threads of a CPU and the grid of a GPU.
         */
        TEST(Medium_TriangularPlan, ResidentWorkItemsFollowTheProfile)
        {
            EXPECT_EQ(K_CPU_COMPUTE_UNITS,
                      Kernels::SYCL::resident_work_items(cpu_profile(K_CPU_COMPUTE_UNITS)));
            EXPECT_EQ(K_GPU_COMPUTE_UNITS * Device::K_WAVES_PER_COMPUTE_UNIT * K_GPU_WORK_GROUP,
                      Kernels::SYCL::resident_work_items(gpu_profile()));
        }

        /**
         * @brief Tests that a block is at least one step and at most the whole system.
         */
        TEST(Medium_TriangularPlan, ABlockIsClampedToTheSystem)
        {
            using Kernels::SYCL::TriangularSolvePlan;

            EXPECT_EQ(64, TriangularSolvePlan{64}.block_width(200));
            EXPECT_EQ(4, TriangularSolvePlan{64}.block_count(200));

            EXPECT_EQ(200, TriangularSolvePlan{1000}.block_width(200));
            EXPECT_EQ(1, TriangularSolvePlan{1000}.block_count(200));

            EXPECT_EQ(1, TriangularSolvePlan{0}.block_width(200));
            EXPECT_EQ(200, TriangularSolvePlan{0}.block_count(200));
            EXPECT_EQ(1, TriangularSolvePlan{-5}.block_width(200));
            EXPECT_EQ(200, TriangularSolvePlan{-5}.block_count(200));

            EXPECT_EQ(1, TriangularSolvePlan{64}.block_width(0));
            EXPECT_EQ(1, TriangularSolvePlan{64}.block_count(0));
        }
    } // namespace Test
} // namespace SushiBLAS
