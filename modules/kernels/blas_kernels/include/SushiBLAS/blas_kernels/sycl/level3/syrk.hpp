/**************************************************************************/
/* syrk.hpp                                                               */
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
/*                                                                        */
/* Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited   */
/* Apache-2.0, see NOTICE.md                                              */
/**************************************************************************/

/**
 * @file syrk.hpp
 * @brief Defines the SYCL kernels of SYRK and HERK, their tile geometry and their packed store.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Holds SYRK's own tile geometry, independent of GEMM's.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct SyrkTileConfig
                {
                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int64_t K_ITEM_ROWS = 4;

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int64_t K_ITEM_COLS = 4;

                    /** @brief Work-items down the row axis of a work-group. */
                    static constexpr int64_t K_WG_ROWS = 8;

                    /** @brief Work-items across the column axis of a work-group. */
                    static constexpr int64_t K_WG_COLS = 8;

                    /** @brief Rows of C covered by one work-group. */
                    static constexpr int64_t K_BLOCK_M = K_ITEM_ROWS * K_WG_ROWS;

                    /** @brief Columns of C covered by one work-group. */
                    static constexpr int64_t K_BLOCK_N = K_ITEM_COLS * K_WG_COLS;

                    /** @brief K-extent staged into local memory per iteration. */
                    static constexpr int64_t K_TILE_K = 16;

                    /** @brief Work-items per work-group. */
                    static constexpr int64_t K_WG_ITEMS = K_WG_ROWS * K_WG_COLS;

                    /** @brief Returns the local memory one work-group needs, in bytes. */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
                    {
                        return static_cast<std::size_t>(K_BLOCK_M * K_TILE_K + K_TILE_K * K_BLOCK_N) *
                               sizeof(T);
                    }
                };

                /**
                 * @brief Stores one triangle of a packed SYRK or HERK: alpha * sum + beta * C.
                 * @tparam Conj Unconjugated for SYRK, Conjugated for HERK.
                 * @tparam Scalar The type of alpha and beta: T, or its real component for HERK.
                 * @see kernels/README.md
                 */
                template<typename T, typename Conj, typename Scalar>
                class SyrkStore final
                {
                    public:
                        /** @brief Binds the triangle, the scalars and the addressing of C. */
                        SyrkStore(bool upper, Scalar alpha, Scalar beta,
                                  OperandAccess c_access) noexcept
                            : upper_(upper), alpha_(alpha), beta_(beta), c_access_(c_access)
                        {

                        }

                        /** @brief Returns whether a macro-block holds part of the triangle. */
                        [[nodiscard]] bool wants(int64_t row0, int64_t rows,
                                                 int64_t col0, int64_t cols) const noexcept
                        {
                            return upper_ ? (col0 + cols - 1 >= row0) : (row0 + rows - 1 >= col0);
                        }

                        /**
                         * @brief Writes element (row, col) of one matrix if the triangle holds it.
                         * @param c The first element of the matrix.
                         */
                        void operator()(T* c, int64_t row, int64_t col, Accumulator<T> sum) const
                        {
                            using W = Accumulator<T>;
                            using WS = Accumulator<Scalar>;

                            if (upper_ ? (col < row) : (col > row))
                                return;

                            const int64_t c_idx = c_access_.index(row, col);

                            // A zero beta must not read C, whose prior value may be uninitialised.
                            const W updated = (beta_ == Scalar(0))
                                ? W(multiply(WS(alpha_), sum))
                                : W(multiply(WS(alpha_), sum) + multiply(WS(beta_), W(c[c_idx])));

                            c[c_idx] = T((row == col) ? Conj::diagonal(updated) : updated);
                        }

                    private:
                        /// @brief Whether the upper triangle is the one stored.
                        bool upper_;

                        /// @brief The scale of the product.
                        Scalar alpha_;

                        /// @brief The scale of the prior C; exactly zero skips reading it.
                        Scalar beta_;

                        /// @brief How element (row, col) of C is addressed.
                        OperandAccess c_access_;
                };
            } // namespace Detail

            /**
             * @brief Names the work-group tile the tiled SYRK stages in local memory, per element type.
             * @see kernels/README.md
             */
            struct SyrkTile
            {
                /** @brief Returns what the tile of element type T needs from a device. */
                template<typename T>
                [[nodiscard]] static constexpr Device::LocalTileDemand demand() noexcept
                {
                    using Configuration = Detail::SyrkTileConfig<T>;
                    return Device::LocalTileDemand{
                        Configuration::local_mem_bytes(),
                        static_cast<std::size_t>(Configuration::K_WG_ITEMS)};
                }
            };

            /**
             * @brief Computes C = alpha*op(A)*Conj(op(A))^T + beta*C on one triangle of C.
             * @tparam Conj Unconjugated for SYRK, Conjugated for HERK.
             * @tparam Scalar The type of alpha and beta: T, or its real component for HERK.
             * @param scratch Device memory for the packed panels; may be null.
             * @throws std::runtime_error When the device cannot host the tile's work-group.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            [[nodiscard]] sycl::event syrk(sycl::queue& queue, Core::Layout layout,
                             Core::Uplo uplo, Core::Transpose trans,
                             int64_t n, int64_t k,
                             Scalar alpha, const T* a, int64_t lda, int64_t str_a,
                             Scalar beta, T* c, int64_t ldc, int64_t str_c,
                             int64_t batch_size, void* scratch, std::size_t scratch_bytes,
                             const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;
                using Configuration = Detail::SyrkTileConfig<T>;
                constexpr int64_t item_rows = Configuration::K_ITEM_ROWS;
                constexpr int64_t item_cols = Configuration::K_ITEM_COLS;
                constexpr int64_t wg_rows = Configuration::K_WG_ROWS;
                constexpr int64_t wg_cols = Configuration::K_WG_COLS;
                constexpr int64_t block_m = Configuration::K_BLOCK_M;
                constexpr int64_t block_n = Configuration::K_BLOCK_N;
                constexpr int64_t tile_k = Configuration::K_TILE_K;

                const bool upper = (uplo == Core::Uplo::Upper);

                // The second factor is read through the transpose of op(A)'s access.
                const OperandAccess a_access = operand_access(layout, trans, lda);
                const OperandAccess b_access = a_access.transposed();
                const OperandAccess c_access =
                    operand_access(layout, Core::Transpose::NoTrans, ldc);

                const Device::Profile& profile = Device::profile_for(queue);
                const PackedProductPlan packed =
                    plan_packed_product<T>(profile, n, n, k, batch_size);

                if (packed_product_is_ready(packed, scratch, scratch_bytes))
                {
                    // The second factor is op(A) transposed, conjugated once more under Conjugated.
                    OperandAccess second = a_access.transposed();
                    second.conjugated = (a_access.conjugated != std::is_same_v<Conj, Conjugated>);

                    SB_LOG_INFO("SYCL SYRK [packed]: batch={} {}x{} k={} block={}x{}x{} workers={}",
                                batch_size, n, n, k, packed.block_rows, packed.block_cols,
                                packed.block_depth, packed.workers);

                    return Detail::packed_product<T>(
                        queue, packed, n, n, k, a, a_access, str_a, a, second, str_a,
                        c, str_c, scratch, deps,
                        Detail::SyrkStore<T, Conj, Scalar>(upper, alpha, beta, c_access));
                }

                const Device::LocalTileDemand demand = SyrkTile::demand<T>();
                SB_THROW_IF(!Device::hosts_local_tile(profile, demand),
                            "SYCL SYRK stages {} B of local memory in {}-item work-groups; the "
                            "device offers {} B and {}.",
                            demand.local_bytes, demand.work_group_items,
                            profile.local_mem_bytes, profile.max_work_group_size);

                const int64_t n_tiles_m = (n + block_m - 1) / block_m;
                const int64_t n_tiles_n = (n + block_n - 1) / block_n;

                SB_LOG_INFO("SYCL SYRK [tiled]: batch={} {}x{} k={}", batch_size, n, n, k);

                sycl::range<3> global(static_cast<size_t>(batch_size),
                                       static_cast<size_t>(n_tiles_m * wg_rows),
                                       static_cast<size_t>(n_tiles_n * wg_cols));
                sycl::range<3> local(1, static_cast<size_t>(wg_rows), static_cast<size_t>(wg_cols));

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);

                    sycl::local_accessor<T, 1> tileA(sycl::range<1>(block_m * tile_k), h);
                    sycl::local_accessor<T, 1> tileB(sycl::range<1>(tile_k * block_n), h);

                    h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                    {
                        const int64_t batch_id = item.get_group(0);
                        const int64_t row0 = item.get_group(1) * block_m;
                        const int64_t col0 = item.get_group(2) * block_n;
                        const int64_t local_row = item.get_local_id(1);
                        const int64_t local_col = item.get_local_id(2);
                        const int64_t local_linear = local_row * wg_cols + local_col;
                        constexpr int64_t wg_size = wg_rows * wg_cols;

                        const T* A = a + batch_id * str_a;
                        T* C = c + batch_id * str_c;

                        // A block wholly inside the unwanted triangle is skipped.
                        const bool tile_wanted = upper
                            ? (col0 + block_n - 1 >= row0)
                            : (row0 + block_m - 1 >= col0);
                        if (!tile_wanted)
                            return;

                        W acc[item_rows][item_cols];
                        for (int64_t ii = 0; ii < item_rows; ++ii)
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                                acc[ii][jj] = W(0);

                        for (int64_t k0 = 0; k0 < k; k0 += tile_k)
                        {
                            for (int64_t idx = local_linear; idx < block_m * tile_k; idx += wg_size)
                            {
                                const int64_t r = idx / tile_k;
                                const int64_t kk = idx % tile_k;
                                tileA[idx] =
                                    Detail::load_operand(A, row0 + r, k0 + kk, n, k, a_access);
                            }
                            for (int64_t idx = local_linear; idx < tile_k * block_n; idx += wg_size)
                            {
                                const int64_t kk = idx / block_n;
                                const int64_t cc = idx % block_n;
                                tileB[idx] = Conj::apply(
                                    Detail::load_operand(A, k0 + kk, col0 + cc, k, n, b_access));
                            }

                            sycl::group_barrier(item.get_group());

                            for (int64_t kk = 0; kk < tile_k; ++kk)
                            {
                                T rega[item_rows], regb[item_cols];
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                    rega[ii] = tileA[(local_row * item_rows + ii) * tile_k + kk];
                                for (int64_t jj = 0; jj < item_cols; ++jj)
                                    regb[jj] = tileB[kk * block_n + (local_col * item_cols + jj)];
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                    for (int64_t jj = 0; jj < item_cols; ++jj)
                                        acc[ii][jj] += multiply(W(rega[ii]), W(regb[jj]));
                            }

                            sycl::group_barrier(item.get_group());
                        }

                        for (int64_t ii = 0; ii < item_rows; ++ii)
                        {
                            const int64_t global_row = row0 + local_row * item_rows + ii;
                            if (global_row >= n)
                                continue;
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                            {
                                const int64_t global_col = col0 + local_col * item_cols + jj;
                                if (global_col >= n)
                                    continue;
                                // Upper keeps col >= row; lower keeps col <= row.
                                if (upper ? (global_col < global_row) : (global_col > global_row))
                                    continue;

                                const int64_t c_idx = c_access.index(global_row, global_col);
                                // A zero beta must not read the prior value of C.
                                const W updated = (beta == Scalar(0))
                                    ? W(multiply(WS(alpha), acc[ii][jj]))
                                    : W(multiply(WS(alpha), acc[ii][jj]) +
                                        multiply(WS(beta), W(C[c_idx])));
                                C[c_idx] = T((global_row == global_col) ? Conj::diagonal(updated)
                                                                        : updated);
                            }
                        }
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
