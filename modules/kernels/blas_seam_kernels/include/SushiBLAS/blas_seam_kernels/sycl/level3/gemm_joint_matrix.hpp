/**************************************************************************/
/* gemm_joint_matrix.hpp                                                  */
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
 * @file gemm_joint_matrix.hpp
 * @brief Declares the HALF GEMM kernel that multiplies on the matrix units through joint_matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Holds the compile-time geometry of the joint_matrix HALF GEMM.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                struct GemmJointMatrixTile
                {
                    /** @brief Rows, columns and depth of one matrix-unit product. */
                    static constexpr int K_FRAGMENT = 16;

                    /** @brief Work-items of one sub-group, a CUDA warp. */
                    static constexpr int K_SUB_GROUP = 32;

                    /** @brief Fragments one sub-group owns along each side of its block of C. */
                    static constexpr int K_FRAGMENTS = 2;

                    /** @brief Sub-groups of a work-group along each side of its block of C. */
                    static constexpr int K_SUB_GROUPS = 2;

                    /** @brief Rows and columns of C one sub-group owns. */
                    static constexpr int K_SUB_BLOCK = K_FRAGMENT * K_FRAGMENTS;

                    /** @brief Rows and columns of C one work-group owns. */
                    static constexpr int K_BLOCK = K_SUB_BLOCK * K_SUB_GROUPS;

                    /** @brief Work-items of one work-group. */
                    static constexpr int K_WG_ITEMS = K_SUB_GROUP * K_SUB_GROUPS * K_SUB_GROUPS;

                    /** @brief HALF elements a leading dimension of A or B is a multiple of. */
                    static constexpr int64_t K_LEAD_MULTIPLE = 8;

                    /** @brief Bytes every fragment origin is aligned to. */
                    static constexpr std::uintptr_t K_ALIGNMENT = 32;
                };

                /**
                 * @brief Holds the FLOAT32 fragment each sub-group of a work-group stores before
                 *        it scales, rounds and writes it.
                 */
                struct alignas(GemmJointMatrixTile::K_ALIGNMENT) JointMatrixStoreTile
                {
                    /** @brief One 16 x 16 fragment per sub-group, row-major. */
                    float values[GemmJointMatrixTile::K_SUB_GROUPS *
                                 GemmJointMatrixTile::K_SUB_GROUPS]
                                [GemmJointMatrixTile::K_FRAGMENT * GemmJointMatrixTile::K_FRAGMENT];
                };

                /** @brief Returns whether one HALF operand meets the fragment load's rules. */
                [[nodiscard]] inline bool joint_matrix_operand(const sycl::half* p,
                                                               const OperandAccess& s,
                                                               int64_t stride_batch) noexcept
                {
                    using Tile = GemmJointMatrixTile;
                    const int64_t lead = s.row == 1 ? s.col : (s.col == 1 ? s.row : 0);
                    const int64_t fragment_elements = Tile::K_FRAGMENT;
                    return lead > 0 && lead % Tile::K_LEAD_MULTIPLE == 0 &&
                           stride_batch % fragment_elements == 0 &&
                           reinterpret_cast<std::uintptr_t>(p) % Tile::K_ALIGNMENT == 0;
                }

                /**
                 * @brief Returns whether a HALF product can run on gemm_joint_matrix.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                [[nodiscard]] inline bool joint_matrix_fits(const Device::Profile& profile,
                                                            int64_t m, int64_t n, int64_t k,
                                                            const sycl::half* a,
                                                            const OperandAccess& a_access,
                                                            int64_t str_a, const sycl::half* b,
                                                            const OperandAccess& b_access,
                                                            int64_t str_b) noexcept
                {
                    const int64_t fragment = GemmJointMatrixTile::K_FRAGMENT;
                    return profile.supports_half_matrix && m > 0 && n > 0 && k > 0 &&
                           m % fragment == 0 && n % fragment == 0 && k % fragment == 0 &&
                           joint_matrix_operand(a, a_access, str_a) &&
                           joint_matrix_operand(b, b_access, str_b);
                }

                /**
                 * @brief Holds one HALF GEMM on the matrix units as a kernel, A and B read in the
                 *        layouts the two template arguments name.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<sycl::ext::oneapi::experimental::matrix::layout LayoutA,
                         sycl::ext::oneapi::experimental::matrix::layout LayoutB,
                         typename Epilogue>
                struct JointMatrixKernel
                {
                    int64_t m;               /**< Rows of C. */
                    int64_t n;               /**< Columns of C. */
                    int64_t k;               /**< The depth of the product. */
                    sycl::half alpha;        /**< The scale of op(A) * op(B). */
                    const sycl::half* a;     /**< The first element of A. */
                    OperandAccess a_access;  /**< How op(A) is read. */
                    int64_t str_a;           /**< The batch stride of A. */
                    std::size_t a_stride;    /**< The fragment stride of A. */
                    const sycl::half* b;     /**< The first element of B. */
                    OperandAccess b_access;  /**< How op(B) is read. */
                    int64_t str_b;           /**< The batch stride of B. */
                    std::size_t b_stride;    /**< The fragment stride of B. */
                    sycl::half beta;         /**< The scale of C. */
                    sycl::half* c;           /**< The first element of C. */
                    OperandAccess c_access;  /**< How C is written. */
                    int64_t str_c;           /**< The batch stride of C. */
                    Epilogue epilogue;       /**< Applied to each element of C as it is stored. */

                    /**
                     * @brief Computes the 64 x 64 block of C that the work-group of item owns.
                     * @see modules/kernels/blas_seam_kernels/README.md
                     */
                    void operator()(sycl::nd_item<3> item) const
                    {
                        // The body exists for NVPTX alone; see the blas_seam_kernels README.
#if !defined(__SYCL_DEVICE_ONLY__) || defined(__NVPTX__)
                        using Tile = GemmJointMatrixTile;
                        namespace Matrix = sycl::ext::oneapi::experimental::matrix;
                        using W = Accumulator<sycl::half>;
                        using sycl::access::address_space;
                        using sycl::access::decorated;
                        using FragmentA = Matrix::joint_matrix<sycl::sub_group, sycl::half,
                                                               Matrix::use::a, Tile::K_FRAGMENT,
                                                               Tile::K_FRAGMENT, LayoutA>;
                        using FragmentB = Matrix::joint_matrix<sycl::sub_group, sycl::half,
                                                               Matrix::use::b, Tile::K_FRAGMENT,
                                                               Tile::K_FRAGMENT, LayoutB>;
                        using FragmentC = Matrix::joint_matrix<sycl::sub_group, float,
                                                               Matrix::use::accumulator,
                                                               Tile::K_FRAGMENT,
                                                               Tile::K_FRAGMENT>;

                        const sycl::sub_group sg = item.get_sub_group();
                        const int sub = static_cast<int>(sg.get_group_linear_id());
                        const int lane = static_cast<int>(sg.get_local_linear_id());
                        const int64_t batch_id = static_cast<int64_t>(item.get_group(0));
                        const int64_t row0 =
                            static_cast<int64_t>(item.get_group(1)) * Tile::K_BLOCK +
                            (sub / Tile::K_SUB_GROUPS) * Tile::K_SUB_BLOCK;
                        const int64_t col0 =
                            static_cast<int64_t>(item.get_group(2)) * Tile::K_BLOCK +
                            (sub % Tile::K_SUB_GROUPS) * Tile::K_SUB_BLOCK;

                        const sycl::half* A = a + batch_id * str_a;
                        const sycl::half* B = b + batch_id * str_b;
                        sycl::half* C = c + batch_id * str_c;

                        // m and n are multiples of 16: a fragment is wholly inside C or not.
                        bool row_inside[Tile::K_FRAGMENTS];
                        bool col_inside[Tile::K_FRAGMENTS];
                        for (int f = 0; f < Tile::K_FRAGMENTS; ++f)
                        {
                            row_inside[f] = row0 + f * Tile::K_FRAGMENT < m;
                            col_inside[f] = col0 + f * Tile::K_FRAGMENT < n;
                        }

                        FragmentC acc[Tile::K_FRAGMENTS][Tile::K_FRAGMENTS];
                        for (int i = 0; i < Tile::K_FRAGMENTS; ++i)
                            for (int j = 0; j < Tile::K_FRAGMENTS; ++j)
                                Matrix::joint_matrix_fill(sg, acc[i][j], 0.0f);

                        for (int64_t k0 = 0; k0 < k; k0 += Tile::K_FRAGMENT)
                        {
                            FragmentA fa[Tile::K_FRAGMENTS];
                            FragmentB fb[Tile::K_FRAGMENTS];
                            for (int f = 0; f < Tile::K_FRAGMENTS; ++f)
                            {
                                if (row_inside[f])
                                    Matrix::joint_matrix_load(
                                        sg, fa[f],
                                        sycl::address_space_cast<address_space::global_space,
                                                                 decorated::no>(
                                            A + a_access.index(row0 + f * Tile::K_FRAGMENT,
                                                               k0)),
                                        a_stride);
                                if (col_inside[f])
                                    Matrix::joint_matrix_load(
                                        sg, fb[f],
                                        sycl::address_space_cast<address_space::global_space,
                                                                 decorated::no>(
                                            B + b_access.index(k0,
                                                               col0 + f * Tile::K_FRAGMENT)),
                                        b_stride);
                            }
                            for (int i = 0; i < Tile::K_FRAGMENTS; ++i)
                                for (int j = 0; j < Tile::K_FRAGMENTS; ++j)
                                    if (row_inside[i] && col_inside[j])
                                        Matrix::joint_matrix_mad(sg, acc[i][j], fa[i], fb[j],
                                                                 acc[i][j]);
                        }

                        auto store = sycl::ext::oneapi::group_local_memory_for_overwrite<
                            JointMatrixStoreTile>(item.get_group());
                        float* slot = store.get_raw()->values[sub];
                        constexpr int K_FRAGMENT_ELEMENTS = Tile::K_FRAGMENT * Tile::K_FRAGMENT;

                        for (int i = 0; i < Tile::K_FRAGMENTS; ++i)
                            for (int j = 0; j < Tile::K_FRAGMENTS; ++j)
                            {
                                if (!row_inside[i] || !col_inside[j])
                                    continue;

                                Matrix::joint_matrix_store(
                                    sg, acc[i][j],
                                    sycl::address_space_cast<address_space::local_space,
                                                             decorated::no>(slot),
                                    Tile::K_FRAGMENT, Matrix::layout::row_major);
                                sycl::group_barrier(sg);

                                // beta == 0 never reads C, which may be uninitialised.
                                for (int e = lane; e < K_FRAGMENT_ELEMENTS;
                                     e += Tile::K_SUB_GROUP)
                                {
                                    const int64_t row =
                                        row0 + i * Tile::K_FRAGMENT + e / Tile::K_FRAGMENT;
                                    const int64_t col =
                                        col0 + j * Tile::K_FRAGMENT + e % Tile::K_FRAGMENT;
                                    const int64_t c_idx = c_access.index(row, col);
                                    const W scaled = W(alpha) * W(slot[e]);
                                    const W value = (beta == sycl::half(0))
                                        ? scaled
                                        : scaled + W(beta) * W(C[c_idx]);
                                    C[c_idx] = epilogue(sycl::half(value), row, col);
                                }
                                sycl::group_barrier(sg);
                            }
#else
                        (void)item;
#endif
                    }
                };

                /**
                 * @brief Computes a HALF GEMM on the matrix units, A and B read in the layouts
                 *        the two template arguments name.
                 * @pre joint_matrix_fits holds for the product.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<sycl::ext::oneapi::experimental::matrix::layout LayoutA,
                         sycl::ext::oneapi::experimental::matrix::layout LayoutB,
                         typename Epilogue>
                [[nodiscard]] sycl::event gemm_joint_matrix_as(
                    sycl::queue& queue, int64_t m, int64_t n, int64_t k,
                    sycl::half alpha, const sycl::half* a, OperandAccess a_access, int64_t str_a,
                    const sycl::half* b, OperandAccess b_access, int64_t str_b,
                    sycl::half beta, sycl::half* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using Tile = GemmJointMatrixTile;

                    const int64_t m_groups = (m + Tile::K_BLOCK - 1) / Tile::K_BLOCK;
                    const int64_t n_groups = (n + Tile::K_BLOCK - 1) / Tile::K_BLOCK;
                    const sycl::range<3> global(static_cast<std::size_t>(batch_size),
                                                static_cast<std::size_t>(m_groups),
                                                static_cast<std::size_t>(n_groups) *
                                                    Tile::K_WG_ITEMS);
                    const sycl::range<3> local(1, 1, Tile::K_WG_ITEMS);

                    // A fragment load strides by the step of the axis that is not contiguous.
                    const std::size_t a_stride =
                        static_cast<std::size_t>(a_access.col == 1 ? a_access.row : a_access.col);
                    const std::size_t b_stride =
                        static_cast<std::size_t>(b_access.col == 1 ? b_access.row : b_access.col);

                    const JointMatrixKernel<LayoutA, LayoutB, Epilogue> kernel{
                        m, n, k, alpha, a, a_access, str_a, a_stride, b, b_access, str_b,
                        b_stride, beta, c, c_access, str_c, epilogue};

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(sycl::nd_range<3>(global, local), kernel);
                    });
                }
            } // namespace Detail

            /**
             * @brief Computes `C = Epilogue(alpha * op(A) * op(B) + beta * C)` in HALF on the
             *        matrix units, accumulating in FLOAT32 and rounding once.
             * @pre Detail::joint_matrix_fits holds for the product.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename Epilogue>
            [[nodiscard]] sycl::event gemm_joint_matrix(
                sycl::queue& queue, int64_t m, int64_t n, int64_t k,
                sycl::half alpha, const sycl::half* a, OperandAccess a_access, int64_t str_a,
                const sycl::half* b, OperandAccess b_access, int64_t str_b,
                sycl::half beta, sycl::half* c, OperandAccess c_access, int64_t str_c,
                int64_t batch_size, const std::vector<sycl::event>& deps, Epilogue epilogue)
            {
                using sycl::ext::oneapi::experimental::matrix::layout;

                // A row of op(X) is contiguous when its column step is one.
                const bool a_rows = a_access.col == 1;
                const bool b_rows = b_access.col == 1;
                if (a_rows && b_rows)
                    return Detail::gemm_joint_matrix_as<layout::row_major, layout::row_major>(
                        queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                        beta, c, c_access, str_c, batch_size, deps, epilogue);
                if (a_rows)
                    return Detail::gemm_joint_matrix_as<layout::row_major, layout::col_major>(
                        queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                        beta, c, c_access, str_c, batch_size, deps, epilogue);
                if (b_rows)
                    return Detail::gemm_joint_matrix_as<layout::col_major, layout::row_major>(
                        queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                        beta, c, c_access, str_c, batch_size, deps, epilogue);
                return Detail::gemm_joint_matrix_as<layout::col_major, layout::col_major>(
                    queue, m, n, k, alpha, a, a_access, str_a, b, b_access, str_b,
                    beta, c, c_access, str_c, batch_size, deps, epilogue);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
