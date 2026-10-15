/**************************************************************************/
/* packed_product_plan.hpp                                                */
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
 * @file packed_product_plan.hpp
 * @brief Declares the register block and the block geometry of a packed matrix product.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /** @brief Holds the alignment, in bytes, of every panel and tile of a packed product. */
        inline constexpr std::size_t K_PACKED_ALIGNMENT = 64;

        /**
         * @brief Holds the rows and columns of C one micro-kernel call accumulates in W.
         * @tparam W An accumulation type: float, double or the std::complex of either.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename W>
        struct PackedRegisterBlock
        {
            /** @brief The type a packed panel and a tile store: the real component of W. */
            using Real = RealType<W>;

            /** @brief Reals stored per element: 2 for a complex W, real parts first. */
            static constexpr int K_PARTS = Core::is_complex_v<W> ? 2 : 1;

            /** @brief Rows of C per micro-kernel call. */
            static constexpr int K_ROWS = Core::is_complex_v<W> ? 4 : 6;

            /** @brief Columns of C per micro-kernel call: K_PACKED_ALIGNMENT bytes of W. */
            static constexpr int K_COLS = static_cast<int>(K_PACKED_ALIGNMENT / sizeof(W));

            /** @brief Reals one micro-tile of the accumulation tile holds. */
            static constexpr int K_TILE_REALS = K_PARTS * K_ROWS * K_COLS;
        };

        /**
         * @brief Holds the block sizes, the work split and the scratch layout of one product.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct PackedProductPlan
        {
            bool selected = false;         /**< Whether the profile chooses the packed kernel. */
            int64_t block_rows = 0;        /**< Rows of one macro-block; a multiple of K_ROWS. */
            int64_t block_cols = 0;        /**< Columns of one macro-block; a multiple of K_COLS. */
            int64_t block_depth = 0;       /**< Terms of the sum one packed panel holds. */
            int64_t row_blocks = 0;        /**< Macro-blocks down the rows of one matrix. */
            int64_t col_blocks = 0;        /**< Macro-blocks across the columns of one matrix. */
            int64_t blocks = 0;            /**< Macro-blocks of the whole batch. */
            int64_t workers = 0;           /**< Work-items launched; each owns one scratch slot. */
            std::size_t a_panel_bytes = 0; /**< Bytes of the A panel, which opens a slot. */
            std::size_t b_panel_bytes = 0; /**< Bytes of the B panel, which follows the A panel. */
            std::size_t slot_bytes = 0;    /**< Bytes of one slot: both panels and the tile. */
            std::size_t scratch_bytes = 0; /**< Bytes the launch needs: every slot, and slack. */
        };

        namespace Detail
        {
            /** @brief Holds the largest byte budget a plan reads; its arithmetic is int64_t. */
            inline constexpr std::size_t K_PACKED_BUDGET_CEILING = std::size_t{1} << 40;

            /** @brief Holds the largest count of compute units, or of workers on one, read. */
            inline constexpr std::size_t K_PACKED_WORKER_CEILING = std::size_t{1} << 16;

            /** @brief Returns a byte budget as a signed count, capped at its ceiling. */
            [[nodiscard]] inline constexpr int64_t packed_budget(std::size_t bytes) noexcept
            {
                return static_cast<int64_t>(std::min(bytes, K_PACKED_BUDGET_CEILING));
            }

            /** @brief Returns a worker count as a signed count, capped at its ceiling. */
            [[nodiscard]] inline constexpr int64_t packed_workers(std::size_t count) noexcept
            {
                return static_cast<int64_t>(std::min(count, K_PACKED_WORKER_CEILING));
            }

            /**
             * @brief Returns the smallest multiple of step that is at least value.
             * @pre value is not negative and step is positive.
             */
            [[nodiscard]] inline constexpr int64_t packed_round_up(int64_t value,
                                                                   int64_t step) noexcept
            {
                return (value + step - 1) / step * step;
            }

            /** @brief Returns how many blocks of size block cover extent. */
            [[nodiscard]] inline constexpr int64_t packed_block_count(int64_t extent,
                                                                      int64_t block) noexcept
            {
                return (extent + block - 1) / block;
            }
        } // namespace Detail

        /**
         * @brief Sizes the blocks, the work split and the scratch of an m x n x k product.
         * @tparam T The element type of the operands; the blocks are sized in Accumulator<T>.
         * @return A plan with blocks == 0 when m, n or batch_size is not positive; selected
         *         stays false.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T>
        [[nodiscard]] inline PackedProductPlan
        plan_packed_blocks(const Device::CacheBlocking& blocking, std::size_t compute_units,
                           int64_t m, int64_t n, int64_t k, int64_t batch_size) noexcept
        {
            using W = Accumulator<T>;
            using Block = PackedRegisterBlock<W>;
            constexpr int64_t reg_rows = Block::K_ROWS;
            constexpr int64_t reg_cols = Block::K_COLS;
            constexpr int64_t element_bytes = static_cast<int64_t>(sizeof(W));

            PackedProductPlan plan;
            if (m <= 0 || n <= 0 || batch_size <= 0)
                return plan;

            // The depth is balanced over its blocks, so the last panel is never a sliver.
            const int64_t inner = Detail::packed_budget(blocking.inner_bytes);
            const int64_t depth_cap =
                std::max<int64_t>(1, inner / ((reg_rows + reg_cols) * element_bytes));
            const int64_t depth_total = std::max<int64_t>(1, k);
            const int64_t depth_blocks = Detail::packed_block_count(depth_total, depth_cap);
            plan.block_depth = Detail::packed_block_count(depth_total, depth_blocks);

            const int64_t depth_bytes = plan.block_depth * element_bytes;
            const int64_t outer = Detail::packed_budget(blocking.outer_bytes);
            const int64_t stream = Detail::packed_budget(blocking.stream_bytes);
            const int64_t rows_cap = std::max(reg_rows, outer / depth_bytes / reg_rows * reg_rows);
            const int64_t cols_cap = std::max(reg_cols, stream / depth_bytes / reg_cols * reg_cols);
            plan.block_rows = std::min(rows_cap, Detail::packed_round_up(m, reg_rows));
            plan.block_cols = std::min(cols_cap, Detail::packed_round_up(n, reg_cols));

            const int64_t wanted = std::max<int64_t>(
                1, Detail::packed_workers(compute_units) *
                       Detail::packed_workers(blocking.workers_per_compute_unit));

            // Each pass halves one block extent that is above its register extent, so it ends.
            for (;;)
            {
                plan.row_blocks = Detail::packed_block_count(m, plan.block_rows);
                plan.col_blocks = Detail::packed_block_count(n, plan.block_cols);
                plan.blocks = batch_size * plan.row_blocks * plan.col_blocks;

                const bool tile_fits = plan.block_rows * plan.block_cols * element_bytes <= stream;
                if (plan.blocks >= wanted && tile_fits)
                    break;

                const bool rows_split = plan.block_rows > reg_rows;
                const bool cols_split = plan.block_cols > reg_cols;
                if (cols_split && (plan.block_cols >= plan.block_rows || !rows_split))
                    plan.block_cols = Detail::packed_round_up(plan.block_cols / 2, reg_cols);
                else if (rows_split)
                    plan.block_rows = Detail::packed_round_up(plan.block_rows / 2, reg_rows);
                else
                    break;
            }
            plan.workers = std::min(plan.blocks, wanted);

            const std::size_t rows = static_cast<std::size_t>(plan.block_rows);
            const std::size_t cols = static_cast<std::size_t>(plan.block_cols);
            const std::size_t depth = static_cast<std::size_t>(plan.block_depth);
            const std::size_t align = K_PACKED_ALIGNMENT;
            plan.a_panel_bytes = (rows * depth * sizeof(W) + align - 1) / align * align;
            plan.b_panel_bytes = (cols * depth * sizeof(W) + align - 1) / align * align;

            // rows * cols is a whole number of micro-tiles, each a multiple of the alignment.
            plan.slot_bytes = plan.a_panel_bytes + plan.b_panel_bytes + rows * cols * sizeof(W);
            plan.scratch_bytes = static_cast<std::size_t>(plan.workers) * plan.slot_bytes + align;
            return plan;
        }

        /**
         * @brief Returns whether a profile chooses the packed kernel for one m x n x k product.
         * @tparam T The element type of the operands.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T>
        [[nodiscard]] inline bool selects_packed_product(const Device::Profile& profile,
                                                         int64_t m, int64_t n, int64_t k) noexcept
        {
            using Block = PackedRegisterBlock<Accumulator<T>>;

            if (!profile.is_cpu || m <= 0 || n <= 0 || k < 0)
                return false;

            const int64_t spans = Detail::packed_budget(profile.blocking.min_register_blocks);
            const int64_t operand_bytes = (m * k + k * n) * static_cast<int64_t>(sizeof(T));
            return m >= spans * Block::K_ROWS && n >= spans * Block::K_COLS &&
                   operand_bytes >= Detail::packed_budget(profile.blocking.min_operand_bytes);
        }

        /**
         * @brief Plans one product for the device a profile describes.
         * @tparam T The element type of the operands.
         * @return The plan of plan_packed_blocks, with selected set by selects_packed_product.
         */
        template<typename T>
        [[nodiscard]] inline PackedProductPlan
        plan_packed_product(const Device::Profile& profile, int64_t m, int64_t n, int64_t k,
                            int64_t batch_size) noexcept
        {
            PackedProductPlan plan = plan_packed_blocks<T>(profile.blocking, profile.compute_units,
                                                           m, n, k, batch_size);
            plan.selected = plan.blocks > 0 && selects_packed_product<T>(profile, m, n, k);
            return plan;
        }

        /**
         * @brief Returns whether a launch can run a plan: the profile selected it and the
         *        scratch holds its bytes.
         * @param scratch May be null, which is what a caller without scratch passes.
         */
        [[nodiscard]] inline bool packed_product_is_ready(const PackedProductPlan& plan,
                                                          const void* scratch,
                                                          std::size_t scratch_bytes) noexcept
        {
            return plan.selected && scratch != nullptr && scratch_bytes >= plan.scratch_bytes;
        }
    } // namespace Kernels
} // namespace SushiBLAS
