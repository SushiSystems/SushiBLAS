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
 * @file gemm.hpp
 * @brief Declares the SYCL GEMM kernels and the tile geometry they launch with.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Returns one operand element, or zero for coordinates outside the matrix.
                 * @tparam Access Brick with `T at(const T*, int64_t, int64_t) const`, such as
                 *         OperandAccess or SymmetricAccess.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename Access>
                [[nodiscard]] inline T load_operand(const T* p, int64_t row, int64_t col,
                                                    int64_t rows, int64_t cols,
                                                    const Access& s) noexcept
                {
                    if (row >= rows || col >= cols) return T(0);
                    return s.at(p, row, col);
                }

                /**
                 * @brief Holds the accumulator byte budget per work-item of the LOCAL_TILED variant.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                inline constexpr std::size_t K_LOCAL_ACC_BYTES = 256;

                /**
                 * @brief Holds the accumulator byte budget per work-item of the NO_LOCAL variant.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                inline constexpr std::size_t K_NO_LOCAL_ACC_BYTES = 64;

                /**
                * @brief The largest square register block of @p T that fits a byte budget.
                *
                * @tparam T The accumulator element type.
                * @param budget_bytes Bytes of private memory the block may occupy.
                * @param cap Hard upper bound on the block edge, whatever the budget allows.
                * @return A power-of-two edge length in [1, @p cap].
                */
                template<typename T>
                [[nodiscard]] inline constexpr int register_block_dim(std::size_t budget_bytes,
                                                                      int cap) noexcept
                {
                    int dim = 1;
                    while (dim * 2 <= cap &&
                           static_cast<std::size_t>(dim * 2) *
                           static_cast<std::size_t>(dim * 2) * sizeof(T) <= budget_bytes)
                        dim *= 2;
                    return dim;
                }

                /**
                 * @brief Holds the compile-time geometry of the LOCAL_TILED variant for element type T.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                struct GemmLocalTile
                {
                    /** @brief Work-items down the row axis of a work-group. */
                    static constexpr int K_WG_ROWS = 8;

                    /** @brief Work-items across the column axis of a work-group. */
                    static constexpr int K_WG_COLS = 8;

                    /** @brief Work-items per work-group; 64 is accepted by every device. */
                    static constexpr int K_WG_ITEMS = K_WG_ROWS * K_WG_COLS;

                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_ROWS = register_block_dim<Accumulator<T>>(K_LOCAL_ACC_BYTES, 8);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_COLS = K_ITEM_ROWS;

                    /** @brief Rows of C covered by one work-group. */
                    static constexpr int K_BLOCK_M = K_WG_ROWS * K_ITEM_ROWS;

                    /** @brief Columns of C covered by one work-group. */
                    static constexpr int K_BLOCK_N = K_WG_COLS * K_ITEM_COLS;

                    /** @brief K-extent staged into local memory per iteration. */
                    static constexpr int K_TILE_K = 16;

                    /**
                    * @brief Leading dimension of the A tile, padded off a bank multiple.
                    *
                    * The +1 is not cargo cult; the arithmetic is worked out on the read
                    * site in @ref gemm_local_tiled.
                    */
                    static constexpr int K_TILE_A_LD = K_TILE_K + 1;

                    /**
                     * @brief Returns the local memory one work-group needs, in bytes.
                     * @see modules/kernels/blas_seam_kernels/README.md
                     */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
                    {
                        return (static_cast<std::size_t>(K_BLOCK_M) * K_TILE_A_LD +
                                static_cast<std::size_t>(K_TILE_K) * K_BLOCK_N) * sizeof(T);
                    }
                };

                /**
                 * @brief Holds the compile-time geometry of the NO_LOCAL variant for element type T.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                struct GemmNoLocalTile
                {
                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_ROWS = register_block_dim<Accumulator<T>>(K_NO_LOCAL_ACC_BYTES, 4);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_COLS = K_ITEM_ROWS;
                };

                /**
                * @struct GroupShape
                * @brief A two-dimensional work-group shape, in work-items.
                */
                struct GroupShape
                {
                    /** @brief Work-items down the row axis. */
                    std::size_t rows = 1;

                    /** @brief Work-items across the column axis. */
                    std::size_t cols = 1;
                };

                /**
                 * @brief Splits a work-item budget into the squarest power-of-two 2-D shape.
                 * @return A shape whose product is at most budget and never zero.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                [[nodiscard]] inline GroupShape group_shape(std::size_t budget) noexcept
                {
                    std::size_t side = 1;
                    while ((side * 2) * (side * 2) <= budget) side *= 2;

                    // A non-square budget leaves room for a second row of sub-groups, which is taken; see modules/kernels/blas_seam_kernels/README.md.
                    std::size_t rows = side;
                    if (rows * 2 * side <= budget) rows *= 2;

                    return GroupShape{rows, side};
                }

                /**
                 * @brief Returns the group count covering extent in steps of block, at least one.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                [[nodiscard]] inline constexpr int64_t group_count(int64_t extent,
                                                                   int64_t block) noexcept
                {
                    const int64_t count = (extent + block - 1) / block;
                    return count > 0 ? count : int64_t{1};
                }

                /**
                 * @brief Computes GEMM by tiling A and B through local memory, for devices with real scratchpad memory.
                 * @pre The local allocation fits the device; gemm checks this before it calls.
                 * @param beta Exactly zero skips reading C.
                 * @param c Must not overlap A or B.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event gemm_local_tiled(
                    sycl::queue& queue,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using W = Accumulator<T>;
                    using Configuration = GemmLocalTile<T>;

                    SB_ASSERT(Device::hosts_local_tile(
                                  Device::profile_for(queue),
                                  {Configuration::local_mem_bytes(),
                                   static_cast<std::size_t>(Configuration::K_WG_ITEMS)}),
                              "gemm_local_tiled launched on a device that cannot host its local tile");

                    const int64_t m_groups = group_count(m, Configuration::K_BLOCK_M);
                    const int64_t n_groups = group_count(n, Configuration::K_BLOCK_N);

                    const sycl::range<3> global(
                        static_cast<std::size_t>(batch_size),
                        static_cast<std::size_t>(m_groups * Configuration::K_WG_ROWS),
                        static_cast<std::size_t>(n_groups * Configuration::K_WG_COLS));
                    const sycl::range<3> local(
                        1, static_cast<std::size_t>(Configuration::K_WG_ROWS),
                        static_cast<std::size_t>(Configuration::K_WG_COLS));

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        sycl::local_accessor<T, 1> tile_a(
                            sycl::range<1>(static_cast<std::size_t>(Configuration::K_BLOCK_M) *
                                           Configuration::K_TILE_A_LD),
                            h);
                        sycl::local_accessor<T, 1> tile_b(
                            sycl::range<1>(static_cast<std::size_t>(Configuration::K_TILE_K) *
                                           Configuration::K_BLOCK_N),
                            h);

                        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                        {
                            // Only global offsets and strides are 64-bit; see modules/kernels/blas_seam_kernels/README.md.
                            const int64_t batch_id = static_cast<int64_t>(item.get_group(0));
                            const int64_t row0 =
                                static_cast<int64_t>(item.get_group(1)) * Configuration::K_BLOCK_M;
                            const int64_t col0 =
                                static_cast<int64_t>(item.get_group(2)) * Configuration::K_BLOCK_N;
                            const int local_row = static_cast<int>(item.get_local_id(1));
                            const int local_col = static_cast<int>(item.get_local_id(2));
                            const int local_linear =
                                local_row * Configuration::K_WG_COLS + local_col;

                            const T* A = a + batch_id * str_a;
                            const T* B = b + batch_id * str_b;

                            // Only C is __restrict__: a and b may alias; see modules/kernels/blas_seam_kernels/README.md.
                            T* __restrict__ C = c + batch_id * str_c;

                            W acc[Configuration::K_ITEM_ROWS][Configuration::K_ITEM_COLS];
                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    acc[ii][jj] = W(0);

                            // One local buffer and two barriers per K-step; see modules/kernels/blas_seam_kernels/README.md.
                            for (int64_t k0 = 0; k0 < k; k0 += Configuration::K_TILE_K)
                            {
                                // The whole group stages block_m x tile_k of A and tile_k x block_n of B; see modules/kernels/blas_seam_kernels/README.md.
                                for (int idx = local_linear;
                                     idx < Configuration::K_BLOCK_M * Configuration::K_TILE_K;
                                     idx += Configuration::K_WG_ITEMS)
                                {
                                    const int r = idx / Configuration::K_TILE_K;
                                    const int kk = idx % Configuration::K_TILE_K;
                                    tile_a[r * Configuration::K_TILE_A_LD + kk] =
                                        load_operand(A, row0 + r, k0 + kk, m, k, a_access);
                                }
                                for (int idx = local_linear;
                                     idx < Configuration::K_TILE_K * Configuration::K_BLOCK_N;
                                     idx += Configuration::K_WG_ITEMS)
                                {
                                    const int kk = idx / Configuration::K_BLOCK_N;
                                    const int cc = idx % Configuration::K_BLOCK_N;
                                    tile_b[idx] =
                                        load_operand(B, k0 + kk, col0 + cc, k, n, b_access);
                                }

                                sycl::group_barrier(item.get_group());

                                // Staged out-of-range elements are zero, so the full tile_k width sums exactly; see modules/kernels/blas_seam_kernels/README.md.
                                for (int kk = 0; kk < Configuration::K_TILE_K; ++kk)
                                {
                                    // tile_a_ld is padded to 17; see modules/kernels/blas_seam_kernels/README.md.
                                    T reg_a[Configuration::K_ITEM_ROWS];
                                    T reg_b[Configuration::K_ITEM_COLS];

                                    for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                        reg_a[ii] =
                                            tile_a[(local_row + ii * Configuration::K_WG_ROWS) *
                                                   Configuration::K_TILE_A_LD + kk];
                                    for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                        reg_b[jj] =
                                            tile_b[kk * Configuration::K_BLOCK_N +
                                                   (local_col + jj * Configuration::K_WG_COLS)];

                                    for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                        for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                            acc[ii][jj] += multiply(W(reg_a[ii]), W(reg_b[jj]));
                                }

                                sycl::group_barrier(item.get_group());
                            }

                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                            {
                                const int64_t global_row =
                                    row0 + local_row + ii * Configuration::K_WG_ROWS;
                                if (global_row >= m) continue;

                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                {
                                    const int64_t global_col =
                                        col0 + local_col + jj * Configuration::K_WG_COLS;
                                    if (global_col >= n) continue;

                                    const int64_t c_idx = c_access.index(global_row, global_col);

                                    // beta == 0 must not read the prior C value, which may be uninitialised; see modules/kernels/blas_seam_kernels/README.md.
                                    const T value = (beta == T(0))
                                        ? T(multiply(W(alpha), acc[ii][jj]))
                                        : T(multiply(W(alpha), acc[ii][jj]) +
                                            multiply(W(beta), W(C[c_idx])));

                                    C[c_idx] = epilogue(value, global_row, global_col);
                                }
                            }
                        });
                    });
                }

                /**
                 * @brief Computes GEMM without local memory or barriers, reading operands from global memory.
                 * @param beta Exactly zero skips reading C.
                 * @param c Must not overlap A or B.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event gemm_no_local(
                    sycl::queue& queue, const Device::Profile& profile,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using W = Accumulator<T>;
                    using Configuration = GemmNoLocalTile<T>;

                    const GroupShape group = group_shape(Device::work_group_size(profile));
                    const int wg_rows = static_cast<int>(group.rows);
                    const int wg_cols = static_cast<int>(group.cols);

                    const int64_t block_m =
                        static_cast<int64_t>(wg_rows) * Configuration::K_ITEM_ROWS;
                    const int64_t block_n =
                        static_cast<int64_t>(wg_cols) * Configuration::K_ITEM_COLS;

                    const int64_t m_groups = group_count(m, block_m);
                    const int64_t n_groups = group_count(n, block_n);

                    const sycl::range<3> global(static_cast<std::size_t>(batch_size),
                                                static_cast<std::size_t>(m_groups) * group.rows,
                                                static_cast<std::size_t>(n_groups) * group.cols);
                    const sycl::range<3> local(1, group.rows, group.cols);

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                        {
                            const int64_t batch_id = static_cast<int64_t>(item.get_group(0));

                            // Rows and columns form a strided lattice; see modules/kernels/blas_seam_kernels/README.md.
                            const int64_t row0 =
                                static_cast<int64_t>(item.get_group(1)) * block_m +
                                static_cast<int64_t>(item.get_local_id(1));
                            const int64_t col0 =
                                static_cast<int64_t>(item.get_group(2)) * block_n +
                                static_cast<int64_t>(item.get_local_id(2));

                            const T* A = a + batch_id * str_a;
                            const T* B = b + batch_id * str_b;

                            // Only C is __restrict__, because a and b may alias each other; see modules/kernels/blas_seam_kernels/README.md.
                            T* __restrict__ C = c + batch_id * str_c;

                            W acc[Configuration::K_ITEM_ROWS][Configuration::K_ITEM_COLS];
                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    acc[ii][jj] = W(0);

                            for (int64_t kk = 0; kk < k; ++kk)
                            {
                                T reg_a[Configuration::K_ITEM_ROWS];
                                T reg_b[Configuration::K_ITEM_COLS];

                                // load_operand's zero fill is the tail guard, so no remainder loop is needed; see modules/kernels/blas_seam_kernels/README.md.
                                for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                    reg_a[ii] = load_operand(A, row0 + ii * wg_rows, kk,
                                                             m, k, a_access);
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    reg_b[jj] = load_operand(B, kk, col0 + jj * wg_cols,
                                                             k, n, b_access);

                                for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                    for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                        acc[ii][jj] += multiply(W(reg_a[ii]), W(reg_b[jj]));
                            }

                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                            {
                                const int64_t global_row = row0 + ii * wg_rows;
                                if (global_row >= m) continue;

                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                {
                                    const int64_t global_col = col0 + jj * wg_cols;
                                    if (global_col >= n) continue;

                                    const int64_t c_idx = c_access.index(global_row, global_col);

                                    // beta == 0 must not read the prior C value, which may be uninitialised; see modules/kernels/blas_seam_kernels/README.md.
                                    const T value = (beta == T(0))
                                        ? T(multiply(W(alpha), acc[ii][jj]))
                                        : T(multiply(W(alpha), acc[ii][jj]) +
                                            multiply(W(beta), W(C[c_idx])));

                                    C[c_idx] = epilogue(value, global_row, global_col);
                                }
                            }
                        });
                    });
                }
            } // namespace Detail

            /**
             * @brief Computes `C = Epilogue(alpha * X * Y + beta * C)` for operands read through
             *        access bricks, with the kernel the profile selects.
             * @tparam AccessA Brick with `T at(const T*, int64_t, int64_t) const`; AccessB too.
             * @param beta Exactly zero skips reading C.
             * @param c Must not overlap a or b.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename T, typename AccessA, typename AccessB,
                     typename Epilogue = GemmIdentityEpilogue>
            [[nodiscard]] sycl::event gemm_accessed(sycl::queue& queue,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, AccessA a_access, int64_t str_a,
                            const T* b, AccessB b_access, int64_t str_b,
                            T beta, T* c, OperandAccess c_access, int64_t str_c,
                            int64_t batch_size, const std::vector<sycl::event>& deps,
                            Epilogue epilogue = Epilogue{})
            {
                using LocalConfiguration = Detail::GemmLocalTile<T>;
                constexpr std::size_t local_bytes = LocalConfiguration::local_mem_bytes();
                constexpr std::size_t local_items =
                    static_cast<std::size_t>(LocalConfiguration::K_WG_ITEMS);

                const Device::Profile& profile = Device::profile_for(queue);
                const Device::LocalTileDemand local_demand{local_bytes, local_items};
                const bool tile_hosted = Device::hosts_local_tile(profile, local_demand);

                // Both variants are instantiated here, whichever branch runs; see modules/kernels/blas_seam_kernels/README.md.
                if (profile.has_dedicated_local_memory && tile_hosted)
                {
                    SB_LOG_INFO("SYCL GEMM [local_tiled]: batch={} {}x{}x{} block={}x{} tile_k={} local={}B",
                                batch_size, m, n, k,
                                LocalConfiguration::K_BLOCK_M, LocalConfiguration::K_BLOCK_N,
                                LocalConfiguration::K_TILE_K, local_bytes);

                    return Detail::gemm_local_tiled<T>(
                        queue, m, n, k,
                        alpha, a, a_access, str_a,
                        b, b_access, str_b,
                        beta, c, c_access, str_c,
                        batch_size, deps, epilogue);
                }

                if (profile.has_dedicated_local_memory)
                    SB_LOG_WARN("SYCL GEMM: local tile wants {} B and {} work-items, device offers "
                                "{} B and {}; falling back to the no-local variant.",
                                local_bytes, local_items,
                                profile.local_mem_bytes, profile.max_work_group_size);

                SB_LOG_INFO("SYCL GEMM [no_local]: batch={} {}x{}x{} item_block={}x{}",
                            batch_size, m, n, k,
                            Detail::GemmNoLocalTile<T>::K_ITEM_ROWS,
                            Detail::GemmNoLocalTile<T>::K_ITEM_COLS);

                return Detail::gemm_no_local<T>(
                    queue, profile, m, n, k,
                    alpha, a, a_access, str_a,
                    b, b_access, str_b,
                    beta, c, c_access, str_c,
                    batch_size, deps, epilogue);
            }

            /**
             * @brief Computes `C = Epilogue(alpha * op(A) * op(B) + beta * C)` with the kernel the profile selects.
             * @tparam Epilogue Functor `T operator()(T value, int64_t row, int64_t col) const`, applied before a store.
             * @param beta Exactly zero skips reading C.
             * @param c Must not overlap A or B.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            [[nodiscard]] sycl::event gemm(sycl::queue& queue, Core::Layout layout,
                            Core::Transpose transA, Core::Transpose transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size, const std::vector<sycl::event>& deps,
                            Epilogue epilogue = Epilogue{})
            {
                // C is never transposed in BLAS, so it only ever sees the layout.
                return gemm_accessed<T>(queue, m, n, k,
                                        alpha, a, operand_access(layout, transA, lda), str_a,
                                        b, operand_access(layout, transB, ldb), str_b,
                                        beta, c,
                                        operand_access(layout, Core::Transpose::NoTrans, ldc),
                                        str_c, batch_size, deps, epilogue);
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
