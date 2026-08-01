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

// No dedicated SYRK tiling kernel exists upstream (portBLAS's own
// doc/MissingFeatures.md lists SYRK as unimplemented - see NOTICE.md). This
// composes SushiBLAS's own ported GEMM tiling machinery instead, the same
// way portBLAS's SYMM composes onto its GEMM kernel: C = alpha*op(A)*op(A)^T
// is literally GEMM(A, A, transA=trans, transB=!trans) restricted to one
// triangle of C, so this reuses gemm.hpp's tile-loading helpers and adds the
// triangular output mask - original SushiBLAS work, not a port.

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <kernels/sycl/blas/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
            * @brief Tiled, local-memory-blocked SYCL SYRK: C = alpha*op(A)*op(A)^T + beta*C,
            * only the requested triangle of C referenced/written, matching standard BLAS
            * SYRK semantics. Built from gemm.hpp's tile-loading primitives with the second
            * GEMM operand set to A itself (see file header).
            */
            template<typename T>
            [[nodiscard]] sycl::event syrk(sycl::queue& queue, Core::Layout layout,
                             bool upper, bool trans,
                             int64_t n, int64_t k,
                             T alpha, const T* a, int64_t lda, int64_t str_a,
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

                const bool transB = !trans;
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

                        // Whole-tile early skip: this work-group's block lies entirely in the
                        // unwanted triangle, so it has nothing to load or compute.
                        const bool tile_wanted = upper
                            ? (col0 + block_n - 1 >= row0)
                            : (row0 + block_m - 1 >= col0);
                        if (!tile_wanted) return;

                        T acc[item_rows][item_cols];
                        for (int64_t ii = 0; ii < item_rows; ++ii)
                            for (int64_t jj = 0; jj < item_cols; ++jj)
                                acc[ii][jj] = T(0);

                        for (int64_t k0 = 0; k0 < k; k0 += tile_k)
                        {
                            for (int64_t idx = local_linear; idx < block_m * tile_k; idx += wg_size)
                            {
                                const int64_t r = idx / tile_k;
                                const int64_t kk = idx % tile_k;
                                tileA[idx] = Detail::gemm_load_a(A, layout, trans, row0 + r, k0 + kk, n, k, lda);
                            }
                            for (int64_t idx = local_linear; idx < tile_k * block_n; idx += wg_size)
                            {
                                const int64_t kk = idx / block_n;
                                const int64_t cc = idx % block_n;
                                tileB[idx] = Detail::gemm_load_b(A, layout, transB, k0 + kk, col0 + cc, k, n, lda);
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
                                        acc[ii][jj] += rega[ii] * regb[jj];
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
                                // Per-element triangle mask (matches Reference::syrk exactly):
                                // upper keeps col >= row, lower keeps col <= row.
                                if (upper ? (global_col < global_row) : (global_col > global_row)) continue;

                                const int64_t c_idx = (layout == Core::Layout::ROW_MAJOR)
                                    ? (global_row * ldc + global_col)
                                    : (global_row + global_col * ldc);
                                // beta==0 must not read the (possibly uninitialized) prior C
                                // value, matching standard BLAS semantics.
                                C[c_idx] = (beta == T(0)) ? alpha * acc[ii][jj] : alpha * acc[ii][jj] + beta * C[c_idx];
                            }
                        }
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
