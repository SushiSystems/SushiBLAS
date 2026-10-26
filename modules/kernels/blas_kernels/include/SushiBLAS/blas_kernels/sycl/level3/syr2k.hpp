/**************************************************************************/
/* syr2k.hpp                                                              */
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
 * @file syr2k.hpp
 * @brief Defines the SYCL kernel of SYR2K and, under a conjugation policy, of HER2K.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Holds SYR2K's tile geometry: two register blocks in GEMM's byte budget.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct Syr2kTileConfig
                {
                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int64_t K_ITEM_ROWS =
                        register_block_dim<Accumulator<T>>(K_LOCAL_ACC_BYTES / 2, 4);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int64_t K_ITEM_COLS = K_ITEM_ROWS;

                    /** @brief Work-items down the row axis of a work-group. */
                    static constexpr int64_t K_WG_ROWS = 8;

                    /** @brief Work-items across the column axis of a work-group. */
                    static constexpr int64_t K_WG_COLS = 8;

                    /** @brief Work-items per work-group. */
                    static constexpr int64_t K_WG_ITEMS = K_WG_ROWS * K_WG_COLS;

                    /** @brief Rows of C covered by one work-group. */
                    static constexpr int64_t K_BLOCK_M = K_ITEM_ROWS * K_WG_ROWS;

                    /** @brief Columns of C covered by one work-group. */
                    static constexpr int64_t K_BLOCK_N = K_ITEM_COLS * K_WG_COLS;

                    /** @brief K-extent staged into local memory per iteration. */
                    static constexpr int64_t K_TILE_K = 16;

                    /** @brief Returns the bytes of local memory the four tiles of a group take. */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
                    {
                        return 2 * static_cast<std::size_t>(K_BLOCK_M * K_TILE_K + K_TILE_K * K_BLOCK_N) *
                               sizeof(T);
                    }
                };
            } // namespace Detail

            /**
             * @brief Names the work-group tile SYR2K stages in local memory, per element type.
             * @see kernels/README.md
             */
            struct Syr2kTile
            {
                /** @brief Returns what the tile of element type T needs from a device. */
                template<typename T>
                [[nodiscard]] static constexpr Device::LocalTileDemand demand() noexcept
                {
                    using Configuration = Detail::Syr2kTileConfig<T>;
                    return Device::LocalTileDemand{
                        Configuration::local_mem_bytes(),
                        static_cast<std::size_t>(Configuration::K_WG_ITEMS)};
                }
            };

            /**
             * @brief Computes C = alpha * op(A) * Conj(op(B))^T
             *        + Conj(alpha) * op(B) * Conj(op(A))^T + beta * C on one triangle of C.
             * @tparam Conj Unconjugated for SYR2K, Conjugated for HER2K.
             * @tparam Scalar The type of beta: T for SYR2K, the real component type for HER2K.
             * @throws std::runtime_error if the device cannot host Syr2kTile::demand<T>().
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            [[nodiscard]] sycl::event syr2k(sycl::queue& queue, Core::Layout layout,
                                            Core::Uplo uplo, Core::Transpose trans,
                                            int64_t n, int64_t k,
                                            T alpha, const T* a, int64_t lda, int64_t str_a,
                                            const T* b, int64_t ldb, int64_t str_b,
                                            Scalar beta, T* c, int64_t ldc, int64_t str_c,
                                            int64_t batch_size,
                                            const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;
                using Configuration = Detail::Syr2kTileConfig<T>;
                constexpr int64_t item_rows = Configuration::K_ITEM_ROWS;
                constexpr int64_t item_cols = Configuration::K_ITEM_COLS;
                constexpr int64_t wg_rows = Configuration::K_WG_ROWS;
                constexpr int64_t wg_cols = Configuration::K_WG_COLS;
                constexpr int64_t wg_items = Configuration::K_WG_ITEMS;
                constexpr int64_t block_m = Configuration::K_BLOCK_M;
                constexpr int64_t block_n = Configuration::K_BLOCK_N;
                constexpr int64_t tile_k = Configuration::K_TILE_K;

                const Device::Profile& profile = Device::profile_for(queue);
                const Device::LocalTileDemand demand = Syr2kTile::demand<T>();
                SB_THROW_IF(!Device::hosts_local_tile(profile, demand),
                            "SYCL SYR2K stages {} B of local memory in {}-item work-groups; the "
                            "device offers {} B and {}.",
                            demand.local_bytes, demand.work_group_items,
                            profile.local_mem_bytes, profile.max_work_group_size);

                const bool upper = (uplo == Core::Uplo::Upper);
                const OperandAccess a_rows = operand_access(layout, trans, lda);
                const OperandAccess b_rows = operand_access(layout, trans, ldb);
                const OperandAccess a_cols = a_rows.transposed();
                const OperandAccess b_cols = b_rows.transposed();
                const OperandAccess c_access =
                    operand_access(layout, Core::Transpose::NoTrans, ldc);

                const int64_t row_groups = Detail::group_count(n, block_m);
                const int64_t col_groups = Detail::group_count(n, block_n);

                const sycl::range<3> global(static_cast<std::size_t>(batch_size),
                                            static_cast<std::size_t>(row_groups * wg_rows),
                                            static_cast<std::size_t>(col_groups * wg_cols));
                const sycl::range<3> local(1, static_cast<std::size_t>(wg_rows),
                                           static_cast<std::size_t>(wg_cols));

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);

                    const sycl::range<1> row_tile(static_cast<std::size_t>(block_m * tile_k));
                    const sycl::range<1> col_tile(static_cast<std::size_t>(tile_k * block_n));
                    sycl::local_accessor<T, 1> tile_a_rows(row_tile, h);
                    sycl::local_accessor<T, 1> tile_b_rows(row_tile, h);
                    sycl::local_accessor<T, 1> tile_a_cols(col_tile, h);
                    sycl::local_accessor<T, 1> tile_b_cols(col_tile, h);

                    h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                    {
                        const int64_t batch_id = static_cast<int64_t>(item.get_group(0));
                        const int64_t row0 = static_cast<int64_t>(item.get_group(1)) * block_m;
                        const int64_t col0 = static_cast<int64_t>(item.get_group(2)) * block_n;
                        const int64_t local_row = static_cast<int64_t>(item.get_local_id(1));
                        const int64_t local_col = static_cast<int64_t>(item.get_local_id(2));
                        const int64_t local_linear = local_row * wg_cols + local_col;

                        const T* A = a + batch_id * str_a;
                        const T* B = b + batch_id * str_b;
                        T* C = c + batch_id * str_c;

                        // Every item of a group shares this test, so none waits at a barrier alone.
                        const bool tile_wanted = upper ? (col0 + block_n - 1 >= row0)
                                                       : (row0 + block_m - 1 >= col0);
                        if (!tile_wanted)
                            return;

                        W acc_ab[item_rows][item_cols];
                        W acc_ba[item_rows][item_cols];
                        for (int64_t ii = 0; ii < item_rows; ++ii)
                        {
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                            {
                                acc_ab[ii][jj] = W(0);
                                acc_ba[ii][jj] = W(0);
                            }
                        }

                        for (int64_t k0 = 0; k0 < k; k0 += tile_k)
                        {
                            for (int64_t idx = local_linear; idx < block_m * tile_k;
                                 idx += wg_items)
                            {
                                const int64_t r = idx / tile_k;
                                const int64_t kk = idx % tile_k;
                                tile_a_rows[idx] =
                                    Detail::load_operand(A, row0 + r, k0 + kk, n, k, a_rows);
                                tile_b_rows[idx] =
                                    Detail::load_operand(B, row0 + r, k0 + kk, n, k, b_rows);
                            }
                            for (int64_t idx = local_linear; idx < tile_k * block_n;
                                 idx += wg_items)
                            {
                                const int64_t kk = idx / block_n;
                                const int64_t cc = idx % block_n;
                                tile_a_cols[idx] = Conj::apply(
                                    Detail::load_operand(A, k0 + kk, col0 + cc, k, n, a_cols));
                                tile_b_cols[idx] = Conj::apply(
                                    Detail::load_operand(B, k0 + kk, col0 + cc, k, n, b_cols));
                            }

                            sycl::group_barrier(item.get_group());

                            // Staged out-of-range elements are zero, so the full tile sums exactly.
                            for (int64_t kk = 0; kk < tile_k; ++kk)
                            {
                                T row_a[item_rows];
                                T row_b[item_rows];
                                T col_a[item_cols];
                                T col_b[item_cols];
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                {
                                    const int64_t at = (local_row * item_rows + ii) * tile_k + kk;
                                    row_a[ii] = tile_a_rows[at];
                                    row_b[ii] = tile_b_rows[at];
                                }
                                for (int64_t jj = 0; jj < item_cols; ++jj)
                                {
                                    const int64_t at = kk * block_n + (local_col * item_cols + jj);
                                    col_a[jj] = tile_a_cols[at];
                                    col_b[jj] = tile_b_cols[at];
                                }
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                {
                                    for (int64_t jj = 0; jj < item_cols; ++jj)
                                    {
                                        acc_ab[ii][jj] += multiply(W(row_a[ii]), W(col_b[jj]));
                                        acc_ba[ii][jj] += multiply(W(row_b[ii]), W(col_a[jj]));
                                    }
                                }
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
                                if (upper ? (global_col < global_row) : (global_col > global_row))
                                    continue;

                                const int64_t c_idx = c_access.index(global_row, global_col);
                                const W scaled =
                                    multiply(W(alpha), acc_ab[ii][jj]) +
                                    multiply(Conj::apply(W(alpha)), acc_ba[ii][jj]);

                                // A zero beta must not read the prior value of C.
                                const W updated = (beta == Scalar(0))
                                    ? scaled
                                    : W(scaled + multiply(WS(beta), W(C[c_idx])));
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
