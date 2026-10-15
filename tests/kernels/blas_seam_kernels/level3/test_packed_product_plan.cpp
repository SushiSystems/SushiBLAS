/**************************************************************************/
/* test_packed_product_plan.cpp                                           */
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
 * @file test_packed_product_plan.cpp
 * @brief Tests the register block, the default blocking and the plan of a packed product.
 * @author Mustafa Garip
 */

#include <limits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the cache size the documented plans are computed for, in bytes.
            constexpr std::size_t K_DOCUMENTED_CACHE_BYTES = 1048576;

            /// @brief Holds the compute units the documented plans are computed for.
            constexpr std::size_t K_DOCUMENTED_COMPUTE_UNITS = 12;

            /// @brief Holds the largest value a budget or a count can carry.
            constexpr std::size_t K_SATURATED = std::numeric_limits<std::size_t>::max();

            /** @brief Holds one row of the plan table of DESIGN_GEMM_CPU, section 3.2. */
            struct DocumentedPlan
            {
                int64_t extent;            /**< The value of m, n and k. */
                int64_t block_depth;       /**< The expected terms of one packed panel. */
                int64_t block_rows;        /**< The expected rows of one macro-block. */
                int64_t block_cols;        /**< The expected columns of one macro-block. */
                int64_t row_blocks;        /**< The expected macro-blocks down the rows. */
                int64_t col_blocks;        /**< The expected macro-blocks across the columns. */
                int64_t workers;           /**< The expected work-items. */
                std::size_t slot_bytes;    /**< The expected bytes of one slot. */
                std::size_t scratch_bytes; /**< The expected bytes of the launch. */
            };

            /** @brief Returns the blocking of one register block and one term per block. */
            [[nodiscard]] Device::CacheBlocking smallest_blocking() noexcept
            {
                Device::CacheBlocking blocking;
                blocking.inner_bytes = 1;
                blocking.outer_bytes = 1;
                blocking.stream_bytes = 1;
                blocking.min_operand_bytes = 0;
                blocking.min_register_blocks = 0;
                return blocking;
            }

            /** @brief Returns the blocking whose every block is larger than any matrix. */
            [[nodiscard]] Device::CacheBlocking widest_blocking() noexcept
            {
                Device::CacheBlocking blocking = smallest_blocking();
                blocking.inner_bytes = K_SATURATED;
                blocking.outer_bytes = K_SATURATED;
                blocking.stream_bytes = K_SATURATED;
                return blocking;
            }

            /**
             * @brief Returns the blocking of depth 7, two register rows and three columns.
             * @tparam T The element type the product runs in.
             */
            template<typename T>
            [[nodiscard]] Device::CacheBlocking ragged_blocking() noexcept
            {
                using W = Kernels::Accumulator<T>;
                using Block = Kernels::PackedRegisterBlock<W>;
                constexpr std::size_t depth = 7;

                Device::CacheBlocking blocking = smallest_blocking();
                blocking.inner_bytes = depth * (Block::K_ROWS + Block::K_COLS) * sizeof(W);
                blocking.outer_bytes = depth * 2 * Block::K_ROWS * sizeof(W);
                blocking.stream_bytes = depth * 3 * Block::K_COLS * sizeof(W);
                return blocking;
            }

            /** @brief Returns the blocking of the documented cache with both thresholds at zero. */
            [[nodiscard]] Device::CacheBlocking documented_blocking_without_thresholds() noexcept
            {
                Device::CacheBlocking blocking =
                    Device::cache_blocking_for(K_DOCUMENTED_CACHE_BYTES);
                blocking.min_operand_bytes = 0;
                blocking.min_register_blocks = 0;
                return blocking;
            }

            /** @brief Expects that two blockings agree in every field. */
            void expect_same_blocking(const Device::CacheBlocking& expected,
                                      const Device::CacheBlocking& actual)
            {
                EXPECT_EQ(expected.inner_bytes, actual.inner_bytes);
                EXPECT_EQ(expected.outer_bytes, actual.outer_bytes);
                EXPECT_EQ(expected.stream_bytes, actual.stream_bytes);
                EXPECT_EQ(expected.min_operand_bytes, actual.min_operand_bytes);
                EXPECT_EQ(expected.min_register_blocks, actual.min_register_blocks);
                EXPECT_EQ(expected.workers_per_compute_unit, actual.workers_per_compute_unit);
            }

            /** @brief Expects that two plans agree in every field. */
            void expect_same_plan(const Kernels::PackedProductPlan& expected,
                                  const Kernels::PackedProductPlan& actual)
            {
                EXPECT_EQ(expected.selected, actual.selected);
                EXPECT_EQ(expected.block_rows, actual.block_rows);
                EXPECT_EQ(expected.block_cols, actual.block_cols);
                EXPECT_EQ(expected.block_depth, actual.block_depth);
                EXPECT_EQ(expected.row_blocks, actual.row_blocks);
                EXPECT_EQ(expected.col_blocks, actual.col_blocks);
                EXPECT_EQ(expected.blocks, actual.blocks);
                EXPECT_EQ(expected.workers, actual.workers);
                EXPECT_EQ(expected.a_panel_bytes, actual.a_panel_bytes);
                EXPECT_EQ(expected.b_panel_bytes, actual.b_panel_bytes);
                EXPECT_EQ(expected.slot_bytes, actual.slot_bytes);
                EXPECT_EQ(expected.scratch_bytes, actual.scratch_bytes);
            }

            /**
             * @brief Expects that the documented blocking plans a cubic product as its row says.
             * @tparam T The element type of the product.
             */
            template<typename T>
            void expect_documented_plan(const DocumentedPlan& row)
            {
                SCOPED_TRACE(::testing::Message() << "extent " << row.extent);

                const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<T>(
                    Device::cache_blocking_for(K_DOCUMENTED_CACHE_BYTES),
                    K_DOCUMENTED_COMPUTE_UNITS, row.extent, row.extent, row.extent, 1);

                EXPECT_EQ(row.block_depth, plan.block_depth);
                EXPECT_EQ(row.block_rows, plan.block_rows);
                EXPECT_EQ(row.block_cols, plan.block_cols);
                EXPECT_EQ(row.row_blocks, plan.row_blocks);
                EXPECT_EQ(row.col_blocks, plan.col_blocks);
                EXPECT_EQ(row.row_blocks * row.col_blocks, plan.blocks);
                EXPECT_EQ(row.workers, plan.workers);
                EXPECT_EQ(row.slot_bytes, plan.slot_bytes);
                EXPECT_EQ(row.scratch_bytes, plan.scratch_bytes);
            }

            /**
             * @brief Expects that one plan tiles an m x n matrix once and lays its slot out whole.
             * @tparam T The element type of the product.
             */
            template<typename T>
            void expect_plan_covers(const Device::CacheBlocking& blocking,
                                    std::size_t compute_units, int64_t m, int64_t n, int64_t k,
                                    int64_t batch_size)
            {
                using W = Kernels::Accumulator<T>;
                using Block = Kernels::PackedRegisterBlock<W>;

                SCOPED_TRACE(::testing::Message()
                             << "m " << m << ", n " << n << ", k " << k << ", batch " << batch_size
                             << ", compute units " << compute_units);

                const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<T>(
                    blocking, compute_units, m, n, k, batch_size);

                EXPECT_EQ(0, plan.block_rows % Block::K_ROWS);
                EXPECT_EQ(0, plan.block_cols % Block::K_COLS);
                EXPECT_GE(plan.block_rows * plan.row_blocks, m);
                EXPECT_GT(m, plan.block_rows * (plan.row_blocks - 1));
                EXPECT_GE(plan.block_cols * plan.col_blocks, n);
                EXPECT_GT(n, plan.block_cols * (plan.col_blocks - 1));
                EXPECT_GE(plan.block_depth, 1);
                EXPECT_EQ(batch_size * plan.row_blocks * plan.col_blocks, plan.blocks);
                EXPECT_GE(plan.workers, 1);
                EXPECT_LE(plan.workers, plan.blocks);

                const std::size_t tile_bytes = static_cast<std::size_t>(plan.block_rows) *
                                               static_cast<std::size_t>(plan.block_cols) *
                                               sizeof(W);
                EXPECT_EQ(std::size_t{0}, plan.slot_bytes % Kernels::K_PACKED_ALIGNMENT);
                EXPECT_EQ(plan.a_panel_bytes + plan.b_panel_bytes + tile_bytes, plan.slot_bytes);
                EXPECT_EQ(static_cast<std::size_t>(plan.workers) * plan.slot_bytes +
                              Kernels::K_PACKED_ALIGNMENT,
                          plan.scratch_bytes);
            }

            /**
             * @brief Expects that every plan of the sweep of one element type covers its matrix.
             * @tparam T The element type of the product.
             */
            template<typename T>
            void expect_every_plan_covers()
            {
                const Device::CacheBlocking blockings[] = {
                    smallest_blocking(), widest_blocking(), ragged_blocking<T>(),
                    documented_blocking_without_thresholds()};
                const int64_t extents[] = {1, 5, 6, 7, 16, 17, 100, 1000};
                const int64_t depths[] = {0, 1, 7, 300};
                const int64_t batches[] = {1, 3};
                const std::size_t unit_counts[] = {1, 5, 12};

                for (const Device::CacheBlocking& blocking : blockings)
                    for (const std::size_t units : unit_counts)
                        for (const int64_t m : extents)
                            for (const int64_t n : extents)
                                for (const int64_t k : depths)
                                    for (const int64_t batch : batches)
                                        expect_plan_covers<T>(blocking, units, m, n, k, batch);
            }

            /** @brief Returns a CPU profile a 24 x 64 x 10 FLOAT32 product exactly selects. */
            [[nodiscard]] Device::Profile selecting_profile() noexcept
            {
                Device::Profile profile;
                profile.is_cpu = true;
                profile.blocking.min_operand_bytes = (24 * 10 + 10 * 64) * sizeof(float);
                return profile;
            }
        }

        /**
         * @brief Tests that the register block of each accumulation type has the documented shape.
         */
        TEST(Unit_PackedProductPlan, TheRegisterBlockFollowsTheAccumulationType)
        {
            using Float = Kernels::PackedRegisterBlock<float>;
            using Double = Kernels::PackedRegisterBlock<double>;
            using ComplexFloat = Kernels::PackedRegisterBlock<std::complex<float>>;
            using ComplexDouble = Kernels::PackedRegisterBlock<std::complex<double>>;

            static_assert(Float::K_ROWS == 6 && Float::K_COLS == 16 && Float::K_PARTS == 1,
                          "A float register block is 6 by 16 in one plane.");
            static_assert(Double::K_ROWS == 6 && Double::K_COLS == 8 && Double::K_PARTS == 1,
                          "A double register block is 6 by 8 in one plane.");
            static_assert(ComplexFloat::K_ROWS == 4 && ComplexFloat::K_COLS == 8 &&
                              ComplexFloat::K_PARTS == 2,
                          "A complex float register block is 4 by 8 in two planes.");
            static_assert(ComplexDouble::K_ROWS == 4 && ComplexDouble::K_COLS == 4 &&
                              ComplexDouble::K_PARTS == 2,
                          "A complex double register block is 4 by 4 in two planes.");

            static_assert(Float::K_TILE_REALS * sizeof(Float::Real) %
                              Kernels::K_PACKED_ALIGNMENT == 0,
                          "A float micro-tile is a whole number of alignments.");
            static_assert(Double::K_TILE_REALS * sizeof(Double::Real) %
                              Kernels::K_PACKED_ALIGNMENT == 0,
                          "A double micro-tile is a whole number of alignments.");
            static_assert(ComplexFloat::K_TILE_REALS * sizeof(ComplexFloat::Real) %
                              Kernels::K_PACKED_ALIGNMENT == 0,
                          "A complex float micro-tile is a whole number of alignments.");
            static_assert(ComplexDouble::K_TILE_REALS * sizeof(ComplexDouble::Real) %
                              Kernels::K_PACKED_ALIGNMENT == 0,
                          "A complex double micro-tile is a whole number of alignments.");

            using Half = Kernels::PackedRegisterBlock<Kernels::Accumulator<sycl::half>>;
            static_assert(
                std::is_same_v<Half, Float>,
                "A HALF product accumulates in the register block of float.");

            SUCCEED();
        }

        /**
         * @brief Tests that the starting blocking is derived from the cache a device reports.
         */
        TEST(Unit_PackedProductPlan, TheDefaultBlockingComesFromTheReportedCache)
        {
            const Device::CacheBlocking blocking =
                Device::cache_blocking_for(K_DOCUMENTED_CACHE_BYTES);

            EXPECT_EQ(std::size_t{28672}, blocking.inner_bytes);
            EXPECT_EQ(std::size_t{262144}, blocking.outer_bytes);
            EXPECT_EQ(std::size_t{524288}, blocking.stream_bytes);
            EXPECT_EQ(std::size_t{1048576}, blocking.min_operand_bytes);
            EXPECT_EQ(std::size_t{4}, blocking.min_register_blocks);
            EXPECT_EQ(std::size_t{1}, blocking.workers_per_compute_unit);

            expect_same_blocking(Device::cache_blocking_for(Device::K_ASSUMED_CACHE_BYTES),
                                 Device::CacheBlocking{});
        }

        /**
         * @brief Tests that the plans of the design's table come out of plan_packed_blocks.
         */
        TEST(Unit_PackedProductPlan, TheDocumentedPlansHold)
        {
            expect_documented_plan<double>({2048, 256, 126, 256, 17, 8, 12, 1040384, 12484672});
            expect_documented_plan<double>({1024, 256, 126, 256, 9, 4, 12, 1040384, 12484672});
            expect_documented_plan<double>({512, 256, 126, 128, 5, 4, 12, 649216, 7790656});
            expect_documented_plan<float>({2048, 293, 222, 432, 10, 5, 12, 1150144, 13801792});
            expect_documented_plan<float>({1024, 256, 252, 256, 5, 4, 12, 778240, 9338944});
            expect_documented_plan<float>({512, 256, 252, 128, 3, 4, 12, 518144, 6217792});
            expect_documented_plan<std::complex<double>>(
                {1024, 205, 76, 156, 14, 7, 12, 950656, 11407936});

            const Device::CacheBlocking blocking =
                Device::cache_blocking_for(K_DOCUMENTED_CACHE_BYTES);
            for (const int64_t extent : {int64_t{2048}, int64_t{1024}, int64_t{512}})
            {
                SCOPED_TRACE(::testing::Message() << "HALF against FLOAT32, extent " << extent);
                expect_same_plan(
                    Kernels::plan_packed_blocks<float>(blocking, K_DOCUMENTED_COMPUTE_UNITS,
                                                       extent, extent, extent, 1),
                    Kernels::plan_packed_blocks<sycl::half>(blocking, K_DOCUMENTED_COMPUTE_UNITS,
                                                            extent, extent, extent, 1));
            }
        }

        /**
         * @brief Tests that the macro-blocks of a plan tile the matrix and its slot adds up.
         */
        TEST(Unit_PackedProductPlan, EveryPlanCoversTheMatrixOnce)
        {
            expect_every_plan_covers<float>();
            expect_every_plan_covers<std::complex<double>>();
        }

        /**
         * @brief Tests that a product without rows, columns or matrices plans no block.
         */
        TEST(Unit_PackedProductPlan, AnEmptyProductPlansNothing)
        {
            const Device::Profile profile = selecting_profile();
            const int64_t shapes[][3] = {{0, 64, 1}, {24, 0, 1}, {24, 64, 0}};

            for (const auto& shape : shapes)
            {
                SCOPED_TRACE(::testing::Message() << "m " << shape[0] << ", n " << shape[1]
                                                  << ", batch " << shape[2]);

                const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<float>(
                    profile.blocking, profile.compute_units, shape[0], shape[1], 10, shape[2]);

                EXPECT_EQ(0, plan.blocks);
                EXPECT_EQ(0, plan.workers);
                EXPECT_EQ(std::size_t{0}, plan.scratch_bytes);
                EXPECT_FALSE(Kernels::plan_packed_product<float>(profile, shape[0], shape[1], 10,
                                                                 shape[2]).selected);
            }
        }

        /**
         * @brief Tests that budgets above the whole matrix leave it in one macro-block.
         */
        TEST(Unit_PackedProductPlan, ABudgetAboveEveryMatrixGivesOneBlock)
        {
            const Kernels::PackedProductPlan plan =
                Kernels::plan_packed_blocks<float>(widest_blocking(), 1, 37, 45, 23, 1);

            EXPECT_EQ(1, plan.blocks);
            EXPECT_EQ(1, plan.workers);
            EXPECT_EQ(42, plan.block_rows);
            EXPECT_EQ(48, plan.block_cols);
            EXPECT_EQ(23, plan.block_depth);
        }

        /**
         * @brief Tests that no worker count splits a macro-block below one register block.
         */
        TEST(Unit_PackedProductPlan, TheSplitStopsAtTheRegisterBlock)
        {
            const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<float>(
                Device::CacheBlocking{}, 1000, 6, 16, 8, 1);

            EXPECT_EQ(6, plan.block_rows);
            EXPECT_EQ(16, plan.block_cols);
            EXPECT_EQ(1, plan.blocks);
            EXPECT_EQ(1, plan.workers);
        }

        /**
         * @brief Tests each of the three conditions a profile sets on taking the packed kernel.
         */
        TEST(Unit_PackedProductPlan, SelectionNeedsACPUASpanAndEnoughBytes)
        {
            using Block = Kernels::PackedRegisterBlock<float>;
            constexpr int64_t m = 4 * Block::K_ROWS;
            constexpr int64_t n = 4 * Block::K_COLS;
            constexpr int64_t k = 10;

            const Device::Profile at_threshold = selecting_profile();
            EXPECT_TRUE(Kernels::selects_packed_product<float>(at_threshold, m, n, k));
            EXPECT_TRUE(Kernels::plan_packed_product<float>(at_threshold, m, n, k, 1).selected);

            Device::Profile gpu = at_threshold;
            gpu.is_cpu = false;
            EXPECT_FALSE(Kernels::selects_packed_product<float>(gpu, m, n, k));
            EXPECT_FALSE(Kernels::plan_packed_product<float>(gpu, m, n, k, 1).selected);

            // min_operand_bytes is zero here, so only the span can refuse the product.
            Device::Profile narrow = at_threshold;
            narrow.blocking.min_operand_bytes = 0;
            EXPECT_FALSE(Kernels::selects_packed_product<float>(narrow, m - 1, n, k));
            EXPECT_TRUE(Kernels::selects_packed_product<float>(narrow, m, n, k));

            Device::Profile one_element_short = at_threshold;
            one_element_short.blocking.min_operand_bytes += sizeof(float);
            EXPECT_FALSE(Kernels::selects_packed_product<float>(one_element_short, m, n, k));
        }

        /**
         * @brief Tests that saturated budgets and counts still give a plan of positive sizes.
         */
        TEST(Unit_PackedProductPlan, ASaturatedBudgetDoesNotOverflow)
        {
            Device::CacheBlocking blocking;
            blocking.inner_bytes = K_SATURATED;
            blocking.outer_bytes = K_SATURATED;
            blocking.stream_bytes = K_SATURATED;
            blocking.min_operand_bytes = K_SATURATED;
            blocking.min_register_blocks = K_SATURATED;
            blocking.workers_per_compute_unit = K_SATURATED;

            const Kernels::PackedProductPlan plan =
                Kernels::plan_packed_blocks<float>(blocking, K_SATURATED, 100, 100, 100, 1);

            EXPECT_GT(plan.block_rows, 0);
            EXPECT_GT(plan.block_cols, 0);
            EXPECT_GT(plan.block_depth, 0);
            EXPECT_GT(plan.row_blocks, 0);
            EXPECT_GT(plan.col_blocks, 0);
            EXPECT_GT(plan.blocks, 0);
            EXPECT_GT(plan.workers, 0);
            EXPECT_GT(plan.a_panel_bytes, std::size_t{0});
            EXPECT_GT(plan.b_panel_bytes, std::size_t{0});
            EXPECT_GT(plan.slot_bytes, std::size_t{0});
            EXPECT_GT(plan.scratch_bytes, std::size_t{0});
            EXPECT_LE(plan.workers, plan.blocks);
        }

        /**
         * @brief Tests that a launch is ready only for a selected plan whose scratch holds it.
         */
        TEST(Unit_PackedProductPlan, APlanIsReadyOnlyWithItsScratch)
        {
            const Device::Profile profile = selecting_profile();
            const Kernels::PackedProductPlan plan =
                Kernels::plan_packed_product<float>(profile, 24, 64, 10, 1);
            ASSERT_TRUE(plan.selected);

            const char storage = 0;
            EXPECT_TRUE(Kernels::packed_product_is_ready(plan, &storage, plan.scratch_bytes));
            EXPECT_FALSE(Kernels::packed_product_is_ready(plan, nullptr, plan.scratch_bytes));
            EXPECT_FALSE(Kernels::packed_product_is_ready(plan, &storage, plan.scratch_bytes - 1));

            Kernels::PackedProductPlan unselected = plan;
            unselected.selected = false;
            EXPECT_FALSE(
                Kernels::packed_product_is_ready(unselected, &storage, plan.scratch_bytes));
        }
    } // namespace Test
} // namespace SushiBLAS
