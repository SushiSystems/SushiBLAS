/**************************************************************************/
/* gemm.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Tiling/blocking algorithm ported from portBLAS's gemm_local.hpp (see
// NOTICE.md) into SushiBLAS's own flat template-function convention: a
// three-level-hierarchy Tile<> and expression-tree executor collapse here
// into one fixed-configuration kernel. Deliberately simplified vs. the
// upstream source for this first port - see NOTICE.md's "Deviations" note
// for exactly what was dropped (per-backend tuning table, cache-line-driven
// K-chunking, vectorized packet load/store, internal/edge fast-path split,
// double buffering) and why none of it is a correctness concern.

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                // Single fixed tile configuration (portBLAS's "larger matrices" default
                // is Tile<8,8,8,8>; this first port halves the register block to 4x4 to
                // keep private-memory pressure reasonable across all five dispatched
                // element types, including complex<double> - see NOTICE.md).
                template<typename T>
                struct GemmTileConfig
                {
                    static constexpr int64_t item_rows = 4;
                    static constexpr int64_t item_cols = 4;
                    static constexpr int64_t wg_rows = 8;
                    static constexpr int64_t wg_cols = 8;
                    static constexpr int64_t block_m = item_rows * wg_rows;   // 32
                    static constexpr int64_t block_n = item_cols * wg_cols;   // 32
                    static constexpr int64_t tile_k = 16;
                };

                template<typename T>
                inline T gemm_load_a(const T* a, Core::Layout layout, bool transA,
                                      int64_t row, int64_t col, int64_t m, int64_t k, int64_t lda)
                {
                    if (row >= m || col >= k) return T(0);
                    if (layout == Core::Layout::ROW_MAJOR)
                        return transA ? a[col * lda + row] : a[row * lda + col];
                    else
                        return transA ? a[row + col * lda] : a[col + row * lda];
                }

                template<typename T>
                inline T gemm_load_b(const T* b, Core::Layout layout, bool transB,
                                      int64_t row, int64_t col, int64_t k, int64_t n, int64_t ldb)
                {
                    if (row >= k || col >= n) return T(0);
                    if (layout == Core::Layout::ROW_MAJOR)
                        return transB ? b[col * ldb + row] : b[row * ldb + col];
                    else
                        return transB ? b[row + col * ldb] : b[col + row * ldb];
                }
            } // namespace Detail

            /**
            * @brief Tiled, local-memory-blocked SYCL GEMM.
            *
            * Local-memory tile of A/B loaded cooperatively by the work-group, each
            * work-item accumulating an item_rows x item_cols block of C in private
            * registers via an outer-product FMA loop over the K-tile, matching
            * portBLAS's gemm_local.hpp shape (see NOTICE.md for the exact origin and
            * this port's deliberate simplifications).
            */
            template<typename T>
            [[nodiscard]] sycl::event gemm(sycl::queue& queue, Core::Layout layout,
                            bool transA, bool transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size, const std::vector<sycl::event>& deps)
            {
                using Cfg = Detail::GemmTileConfig<T>;
                constexpr int64_t item_rows = Cfg::item_rows;
                constexpr int64_t item_cols = Cfg::item_cols;
                constexpr int64_t wg_rows = Cfg::wg_rows;
                constexpr int64_t wg_cols = Cfg::wg_cols;
                constexpr int64_t block_m = Cfg::block_m;
                constexpr int64_t block_n = Cfg::block_n;
                constexpr int64_t tile_k = Cfg::tile_k;

                const int64_t m_tiles = (m + block_m - 1) / block_m;
                const int64_t n_tiles = (n + block_n - 1) / block_n;

                SB_LOG_INFO("SYCL GEMM [tiled]: batch={} {}x{}x{}", batch_size, m, n, k);

                sycl::range<3> global(static_cast<size_t>(batch_size),
                                       static_cast<size_t>(m_tiles * wg_rows),
                                       static_cast<size_t>(n_tiles * wg_cols));
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
                        const T* B = b + batch_id * str_b;
                        T* C = c + batch_id * str_c;

                        T acc[item_rows][item_cols];
                        for (int64_t ii = 0; ii < item_rows; ++ii)
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                                acc[ii][jj] = T(0);

                        for (int64_t k0 = 0; k0 < k; k0 += tile_k)
                        {
                            // Cooperative load: block_m x tile_k elements of A, tile_k x
                            // block_n of B, wg_size threads each taking a grid-strided share.
                            for (int64_t idx = local_linear; idx < block_m * tile_k; idx += wg_size)
                            {
                                const int64_t r = idx / tile_k;
                                const int64_t kk = idx % tile_k;
                                tileA[idx] = Detail::gemm_load_a(A, layout, transA, row0 + r, k0 + kk, m, k, lda);
                            }
                            for (int64_t idx = local_linear; idx < tile_k * block_n; idx += wg_size)
                            {
                                const int64_t kk = idx / block_n;
                                const int64_t cc = idx % block_n;
                                tileB[idx] = Detail::gemm_load_b(B, layout, transB, k0 + kk, col0 + cc, k, n, ldb);
                            }

                            sycl::group_barrier(item.get_group());

                            // Out-of-range K padding above is zero, so summing the full
                            // tile_k width here (rather than clamping to the true
                            // remainder on the last tile) is still exact.
                            for (int64_t kk = 0; kk < tile_k; ++kk)
                            {
                                T rega[item_rows], regb[item_cols];
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                    rega[ii] = tileA[(local_row * item_rows + ii) * tile_k + kk];
                                for (int64_t jj = 0; jj < item_cols; ++jj)
                                    regb[jj] = tileB[kk * block_n + (local_col * item_cols + jj)];
                                for (int64_t ii = 0; ii < item_rows; ++ii)
                                    for (int64_t jj = 0; jj < item_cols; ++jj)
                                        acc[ii][jj] += rega[ii] * regb[jj];
                            }

                            sycl::group_barrier(item.get_group());
                        }

                        for (int64_t ii = 0; ii < item_rows; ++ii)
                        {
                            const int64_t global_row = row0 + local_row * item_rows + ii;
                            if (global_row >= m) continue;
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                            {
                                const int64_t global_col = col0 + local_col * item_cols + jj;
                                if (global_col >= n) continue;
                                const int64_t c_idx = (layout == Core::Layout::ROW_MAJOR)
                                    ? (global_row * ldc + global_col)
                                    : (global_row + global_col * ldc);
                                C[c_idx] = alpha * acc[ii][jj] + beta * C[c_idx];
                            }
                        }
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
