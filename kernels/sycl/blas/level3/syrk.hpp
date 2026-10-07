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
 * @brief Defines the tiled SYCL kernel of SYRK and its tile geometry.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>

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
                    static constexpr int64_t item_rows = 4;

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int64_t item_cols = 4;

                    /** @brief Work-items down the row axis of a work-group. */
                    static constexpr int64_t wg_rows = 8;

                    /** @brief Work-items across the column axis of a work-group. */
                    static constexpr int64_t wg_cols = 8;

                    /** @brief Rows of C covered by one work-group. */
                    static constexpr int64_t block_m = item_rows * wg_rows;

                    /** @brief Columns of C covered by one work-group. */
                    static constexpr int64_t block_n = item_cols * wg_cols;

                    /** @brief K-extent staged into local memory per iteration. */
                    static constexpr int64_t tile_k = 16;
                };
            } // namespace Detail

            /**
             * @brief Computes SYRK, C = alpha*op(A)*op(A)^T + beta*C, tiled through local memory,
             * on the requested triangle of C.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event syrk(sycl::queue& queue, Core::Layout layout,
                             bool upper, bool trans,
                             int64_t n, int64_t k,
                             T alpha, const T* a, int64_t lda, int64_t str_a,
                             T beta, T* c, int64_t ldc, int64_t str_c,
                             int64_t batch_size, const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                using Cfg = Detail::SyrkTileConfig<T>;
                constexpr int64_t item_rows = Cfg::item_rows;
                constexpr int64_t item_cols = Cfg::item_cols;
                constexpr int64_t wg_rows = Cfg::wg_rows;
                constexpr int64_t wg_cols = Cfg::wg_cols;
                constexpr int64_t block_m = Cfg::block_m;
                constexpr int64_t block_n = Cfg::block_n;
                constexpr int64_t tile_k = Cfg::tile_k;

                // op(A)^T is the factor a GEMM addresses with the opposite transpose flag.
                const Detail::OperandStrides a_strides = Detail::strides_for(layout, trans, lda);
                const Detail::OperandStrides b_strides = Detail::strides_for(layout, !trans, lda);
                const Detail::OperandStrides c_strides = Detail::strides_for(layout, false, ldc);

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
                        if (!tile_wanted) return;

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
                                tileA[idx] = Detail::load_operand(A, row0 + r, k0 + kk, n, k, a_strides);
                            }
                            for (int64_t idx = local_linear; idx < tile_k * block_n; idx += wg_size)
                            {
                                const int64_t kk = idx / block_n;
                                const int64_t cc = idx % block_n;
                                tileB[idx] = Detail::load_operand(A, k0 + kk, col0 + cc, k, n, b_strides);
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
                                        acc[ii][jj] += W(rega[ii]) * W(regb[jj]);
                            }

                            sycl::group_barrier(item.get_group());
                        }

                        for (int64_t ii = 0; ii < item_rows; ++ii)
                        {
                            const int64_t global_row = row0 + local_row * item_rows + ii;
                            if (global_row >= n) continue;
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                            {
                                const int64_t global_col = col0 + local_col * item_cols + jj;
                                if (global_col >= n) continue;
                                // Upper keeps col >= row; lower keeps col <= row.
                                if (upper ? (global_col < global_row) : (global_col > global_row)) continue;

                                const int64_t c_idx = global_row * c_strides.row +
                                                      global_col * c_strides.col;
                                // A zero beta must not read the prior value of C.
                                C[c_idx] = (beta == T(0))
                                    ? T(W(alpha) * acc[ii][jj])
                                    : T(W(alpha) * acc[ii][jj] + W(beta) * W(C[c_idx]));
                            }
                        }
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
