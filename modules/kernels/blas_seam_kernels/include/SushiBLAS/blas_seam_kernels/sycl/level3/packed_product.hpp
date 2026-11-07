/**************************************************************************/
/* packed_product.hpp                                                     */
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
 * @file packed_product.hpp
 * @brief Declares the packed matrix product kernel, which blocks op(A) * op(B) for the caches.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_seam_kernels/blas_seam_kernels_log_category.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_product_index.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_panel.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_micro_kernel.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Computes the macro-blocks one work-item owns and hands each sum to a
                 *        store.
                 * @tparam AccessA Brick with `T at(const T*, int64_t, int64_t) const`; AccessB too.
                 * @tparam Store Device-copyable; `bool wants(row0, rows, col0, cols) const` and
                 *         `void operator()(T* c, row, col, Accumulator<T> sum) const`.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Store>
                class PackedProductKernel final
                {
                    public:
                        /** @brief The type a sum is accumulated in. */
                        using W = Accumulator<T>;

                        /** @brief The type the panels and the tile store. */
                        using R = RealType<W>;

                        /** @brief The register block of W. */
                        using Block = PackedRegisterBlock<W>;

                        /**
                         * @brief Binds a plan, the operands and the first scratch slot.
                         * @param slots Holds plan.workers slots of plan.slot_bytes, aligned to
                         *        K_PACKED_ALIGNMENT.
                         */
                        PackedProductKernel(const PackedProductPlan& plan,
                                            int64_t m, int64_t n, int64_t k,
                                            const T* a, AccessA a_access, int64_t str_a,
                                            const T* b, AccessB b_access, int64_t str_b,
                                            T* c, int64_t str_c, R* slots, Store store) noexcept
                            : plan_(plan), m_(m), n_(n), k_(k),
                              a_(a), a_access_(a_access), str_a_(str_a),
                              b_(b), b_access_(b_access), str_b_(str_b),
                              c_(c), str_c_(str_c), slots_(slots), store_(store)
                        {

                        }

                        /** @brief Runs every macro-block of this work-item, in ascending order. */
                        void operator()(sycl::nd_item<1> item) const
                        {
                            const int64_t worker = static_cast<int64_t>(item.get_global_id(0));
                            const int64_t slot_reals =
                                static_cast<int64_t>(plan_.slot_bytes / sizeof(R));

                            R* slot = slots_ + worker * slot_reals;
                            for (int64_t block = worker; block < plan_.blocks;
                                 block += plan_.workers)
                                run_block(block, slot);
                        }

                    private:
                        /** @brief Computes one macro-block of C in a scratch slot and stores it. */
                        void run_block(int64_t block, R* slot) const
                        {
                            const Indexing::PackedBlock at = Indexing::packed_block(
                                block, plan_.row_blocks, plan_.col_blocks, plan_.block_rows,
                                plan_.block_cols);
                            const int64_t batch_id = at.batch;
                            const int64_t row0 = at.row0;
                            const int64_t col0 = at.col0;
                            const int64_t rows = Indexing::live_count(m_, row0, plan_.block_rows);
                            const int64_t cols = Indexing::live_count(n_, col0, plan_.block_cols);

                            if (!store_.wants(row0, rows, col0, cols))
                                return;

                            R* a_panel = slot;
                            R* b_panel = slot + plan_.a_panel_bytes / sizeof(R);
                            R* tile = b_panel + plan_.b_panel_bytes / sizeof(R);

                            const int64_t row_groups = Indexing::ceil_div(rows, Block::K_ROWS);
                            const int64_t col_groups = Indexing::ceil_div(cols, Block::K_COLS);
                            const int64_t live_reals =
                                row_groups * col_groups * Block::K_TILE_REALS;
                            for (int64_t i = 0; i < live_reals; ++i)
                                tile[i] = R(0);

                            const T* A = a_ + batch_id * str_a_;
                            const T* B = b_ + batch_id * str_b_;

                            for (int64_t term0 = 0; term0 < k_; term0 += plan_.block_depth)
                            {
                                const int64_t depth =
                                    Indexing::live_count(k_, term0, plan_.block_depth);

                                pack_panel<T, Block::K_ROWS, PanelLines::Rows>(
                                    A, a_access_, row0, rows, term0, depth, a_panel);
                                pack_panel<T, Block::K_COLS, PanelLines::Columns>(
                                    B, b_access_, col0, cols, term0, depth, b_panel);

                                const int64_t a_micro = depth * Block::K_PARTS * Block::K_ROWS;
                                const int64_t b_micro = depth * Block::K_PARTS * Block::K_COLS;
                                for (int64_t h = 0; h < col_groups; ++h)
                                    for (int64_t g = 0; g < row_groups; ++g)
                                        packed_micro_kernel<W>(
                                            depth,
                                            a_panel + Indexing::micro_panel_offset(g, a_micro),
                                            b_panel + Indexing::micro_panel_offset(h, b_micro),
                                            tile + Indexing::micro_tile_index(g, h, row_groups) *
                                                       Block::K_TILE_REALS);
                            }

                            store_block(c_ + batch_id * str_c_, tile, row0, rows, col0, cols,
                                        row_groups, col_groups);
                        }

                        /** @brief Hands every sum of a finished tile that is in C to the store. */
                        void store_block(T* C, const R* tile, int64_t row0, int64_t rows,
                                         int64_t col0, int64_t cols,
                                         int64_t row_groups, int64_t col_groups) const
                        {
                            for (int64_t h = 0; h < col_groups; ++h)
                            {
                                const int64_t col_base = h * Block::K_COLS;
                                const int64_t live_cols =
                                    Indexing::live_count(cols, col_base, Block::K_COLS);

                                for (int64_t g = 0; g < row_groups; ++g)
                                {
                                    const int64_t row_base = g * Block::K_ROWS;
                                    const int64_t live_rows =
                                        Indexing::live_count(rows, row_base, Block::K_ROWS);
                                    const R* micro =
                                        tile + Indexing::micro_tile_index(g, h, row_groups) *
                                                   Block::K_TILE_REALS;

                                    for (int64_t r = 0; r < live_rows; ++r)
                                        for (int64_t s = 0; s < live_cols; ++s)
                                            store_(C, row0 + row_base + r, col0 + col_base + s,
                                                   sum_at(micro, Indexing::row_major_offset(r, Block::K_COLS, s)));
                                }
                            }
                        }

                        /** @brief Returns the sum at an offset of a micro-tile, parts joined. */
                        [[nodiscard]] static W sum_at(const R* micro, int64_t offset) noexcept
                        {
                            if constexpr (Core::is_complex_v<W>)
                                return W(micro[offset],
                                         micro[Block::K_ROWS * Block::K_COLS + offset]);
                            else
                                return micro[offset];
                        }

                        /// @brief The block sizes, the work split and the slot layout.
                        PackedProductPlan plan_;

                        /// @brief Rows of C.
                        int64_t m_;

                        /// @brief Columns of C.
                        int64_t n_;

                        /// @brief Terms of each sum.
                        int64_t k_;

                        /// @brief The first matrix of A.
                        const T* a_;

                        /// @brief How element (i, l) of the first factor is read.
                        AccessA a_access_;

                        /// @brief Elements between consecutive matrices of A.
                        int64_t str_a_;

                        /// @brief The first matrix of B.
                        const T* b_;

                        /// @brief How element (l, j) of the second factor is read.
                        AccessB b_access_;

                        /// @brief Elements between consecutive matrices of B.
                        int64_t str_b_;

                        /// @brief The first matrix of C.
                        T* c_;

                        /// @brief Elements between consecutive matrices of C.
                        int64_t str_c_;

                        /// @brief The first scratch slot.
                        R* slots_;

                        /// @brief Turns a finished sum into the stored element of C.
                        Store store_;
                };

                /**
                 * @brief Launches a packed product: one work-item per worker of the plan.
                 * @pre plan.blocks is above zero and scratch holds plan.scratch_bytes.
                 * @param store Receives every sum; see PackedProductKernel for its two members.
                 * @param c Must not overlap A, B or the scratch.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Store>
                [[nodiscard]] sycl::event packed_product(
                    sycl::queue& queue, const PackedProductPlan& plan,
                    int64_t m, int64_t n, int64_t k,
                    const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T* c, int64_t str_c, void* scratch,
                    const std::vector<sycl::event>& deps, Store store)
                {
                    using R = RealType<Accumulator<T>>;

                    SB_ASSERT(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, plan.blocks > 0 && plan.workers > 0,
                              "packed_product launched with a plan that holds no macro-block");
                    SB_ASSERT(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, scratch != nullptr, "packed_product launched without scratch");

                    // scratch_bytes holds one alignment unit more than the slots, for this step.
                    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(scratch);
                    const std::uintptr_t unit = K_PACKED_ALIGNMENT;
                    const std::uintptr_t aligned = (address + unit - 1) / unit * unit;
                    R* slots = reinterpret_cast<R*>(aligned);

                    const PackedProductKernel<T, AccessA, AccessB, Store> kernel(
                        plan, m, n, k, a, a_access, str_a, b, b_access, str_b, c, str_c, slots,
                        store);
                    const sycl::nd_range<1> launch(
                        sycl::range<1>(static_cast<std::size_t>(plan.workers)), sycl::range<1>(1));

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(launch, kernel);
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
