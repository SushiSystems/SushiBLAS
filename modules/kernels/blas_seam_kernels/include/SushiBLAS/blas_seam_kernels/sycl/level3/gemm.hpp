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
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_seam_kernels/blas_seam_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/arithmetic/multiply_add.hpp>
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
                    if (row >= rows || col >= cols)
                        return T(0);
                    return s.at(p, row, col);
                }

                /**
                 * @brief Returns element (row0 + r, col0 + c) of an inside block through at().
                 * @pre The whole block lies inside the matrix.
                 */
                template<typename T, typename Access>
                [[nodiscard]] inline T load_interior(const T* p, int64_t row0, int64_t col0, int r,
                                                     int c, const Access& s, bool) noexcept
                {
                    return s.at(p, row0 + r, col0 + c);
                }

                /**
                 * @brief Returns element (row0 + r, col0 + c) of a block inside the matrix.
                 * @param narrow Every offset inside the block fits in 32 bits, which then carry it.
                 * @pre The whole block lies inside the matrix.
                 */
                template<typename T>
                [[nodiscard]] inline T load_interior(const T* p, int64_t row0, int64_t col0, int r,
                                                     int c, const OperandAccess& s,
                                                     bool narrow) noexcept
                {
                    const T* origin = p + s.index(row0, col0);
                    if (!narrow)
                        return conjugate_if(s.conjugated, origin[s.index(r, c)]);
                    const int32_t offset = static_cast<int32_t>(r) * static_cast<int32_t>(s.row) +
                                           static_cast<int32_t>(c) * static_cast<int32_t>(s.col);
                    return conjugate_if(s.conjugated, origin[offset]);
                }

                /// @brief Returns whether every offset inside a rows x cols block fits in 32 bits.
                [[nodiscard]] inline constexpr bool block_fits_int32(const OperandAccess& s,
                                                                     int rows, int cols) noexcept
                {
                    const int64_t last = static_cast<int64_t>(rows - 1) * s.row +
                                         static_cast<int64_t>(cols - 1) * s.col;
                    return s.row > 0 && s.col > 0 && last <= INT32_MAX;
                }

                /// @brief Holds the lanes of one 128-bit load of T; 1 marks a type read by element.
                template<typename T>
                struct StagedLanes
                {
                    /** @brief Elements one load reads. */
                    static constexpr int K_VALUE = 1;
                };

                /// @brief Holds the four float lanes of one 128-bit load.
                template<>
                struct StagedLanes<float>
                {
                    /** @brief Elements one load reads. */
                    static constexpr int K_VALUE = 4;
                };

                /// @brief Holds the two double lanes of one 128-bit load.
                template<>
                struct StagedLanes<double>
                {
                    /** @brief Elements one load reads. */
                    static constexpr int K_VALUE = 2;
                };

                /// @brief Holds the eight HALF lanes of one 128-bit load.
                template<>
                struct StagedLanes<sycl::half>
                {
                    /** @brief Elements one load reads. */
                    static constexpr int K_VALUE = 8;
                };

                /**
                 * @brief Returns whether whole tiles of an operand can be staged by 128-bit loads.
                 * @param stride_batch Elements between the operands of two batch entries.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                [[nodiscard]] inline bool stages_by_vector(const T* p, const OperandAccess& s,
                                                           int64_t stride_batch) noexcept
                {
                    constexpr int64_t K_LANES = StagedLanes<T>::K_VALUE;
                    constexpr std::uintptr_t K_LOAD_BYTES = 16;
                    if (K_LANES == 1 || s.conjugated)
                        return false;

                    // The leading dimension is the stride of the axis that is not contiguous.
                    const int64_t lead = s.row == 1 ? s.col : (s.col == 1 ? s.row : 0);
                    return lead > 0 && lead % K_LANES == 0 && stride_batch % K_LANES == 0 &&
                           reinterpret_cast<std::uintptr_t>(p) % K_LOAD_BYTES == 0;
                }

                /// @brief Returns false: a brick other than OperandAccess is staged by element.
                template<typename T, typename Access>
                [[nodiscard]] inline bool stages_by_vector(const T*, const Access&,
                                                           int64_t) noexcept
                {
                    return false;
                }

                /**
                 * @brief Returns the StagedLanes<T> elements that start at p, in one load.
                 * @pre p is 16-byte aligned and points into global memory.
                 */
                template<typename T>
                [[nodiscard]] inline sycl::vec<T, StagedLanes<T>::K_VALUE> load_lanes(
                    const T* p) noexcept
                {
                    using sycl::access::address_space;
                    sycl::vec<T, StagedLanes<T>::K_VALUE> lanes;
                    lanes.load(0, sycl::address_space_cast<address_space::global_space,
                                                           sycl::access::decorated::no>(p));
                    return lanes;
                }

                /// @brief Returns false: a brick other than OperandAccess has no offsets to narrow.
                template<typename Access>
                [[nodiscard]] inline constexpr bool block_fits_int32(const Access&, int,
                                                                     int) noexcept
                {
                    return false;
                }

                /**
                 * @brief Holds how stage_block reads one operand, decided on the host.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                struct StagePlan
                {
                    bool columns_contiguous = false; /**< Neighbouring rows are adjacent. */
                    bool narrow = false;             /**< Offsets inside a block fit 32 bits. */
                    bool vector = false;             /**< Whole blocks take 128-bit loads. */
                };

                /// @brief Returns the StagePlan of an operand staged in Rows x Cols blocks.
                template<int Rows, int Cols, typename T, typename Access>
                [[nodiscard]] inline StagePlan stage_plan(const T* p, const Access& s,
                                                          int64_t stride_batch) noexcept
                {
                    return StagePlan{has_contiguous_columns(s), block_fits_int32(s, Rows, Cols),
                                     stages_by_vector(p, s, stride_batch)};
                }

                /**
                 * @brief Stages the Rows x Cols block of op(X) at (row0, col0) into a local tile.
                 * @param offset The tile's first element in the local buffer.
                 * @param linear The work-item's linear id in a group of Items work-items.
                 * @param rows The extent of op(X) down its rows, as cols is across its columns.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<int Rows, int Cols, int Ld, int Items, typename T, typename Access,
                         typename Tile>
                inline void stage_block(const Tile& tile, int offset, int linear, const T* p,
                                        int64_t row0, int64_t col0, int64_t rows, int64_t cols,
                                        const Access& s, const StagePlan& plan) noexcept
                {
                    // Whether the block is whole is the same for every work-item of the group.
                    const bool full = row0 + Rows <= rows && col0 + Cols <= cols;

                    if constexpr (StagedLanes<T>::K_VALUE > 1 &&
                                  std::is_same<Access, OperandAccess>::value)
                    {
                        constexpr int K_LANES = StagedLanes<T>::K_VALUE;
                        static_assert(Rows % K_LANES == 0 && Cols % K_LANES == 0,
                                      "a staged block must split into whole 128-bit loads");
                        if (full && plan.vector)
                        {
                            // One load takes K_LANES rows of a column, or K_LANES columns of a row.
                            const int row_steps = plan.columns_contiguous ? Rows / K_LANES : Rows;
                            const int col_steps = plan.columns_contiguous ? Cols : Cols / K_LANES;
                            for (int v = linear; v < Rows * Cols / K_LANES; v += Items)
                            {
                                const int r = plan.columns_contiguous ? (v % row_steps) * K_LANES
                                                                      : v / col_steps;
                                const int c = plan.columns_contiguous ? v / row_steps
                                                                      : (v % col_steps) * K_LANES;
                                const auto lanes = load_lanes(p + s.index(row0 + r, col0 + c));
                                for (int l = 0; l < K_LANES; ++l)
                                    tile[offset + (plan.columns_contiguous ? (r + l) * Ld + c
                                                                           : r * Ld + c + l)] =
                                        lanes[l];
                            }
                            return;
                        }
                    }

                    for (int idx = linear; idx < Rows * Cols; idx += Items)
                    {
                        const int r = plan.columns_contiguous ? idx % Rows : idx / Cols;
                        const int c = plan.columns_contiguous ? idx / Rows : idx % Cols;
                        tile[offset + r * Ld + c] =
                            full ? load_interior(p, row0, col0, r, c, s, plan.narrow)
                                 : load_operand(p, row0 + r, col0 + c, rows, cols, s);
                    }
                }

                /// @brief Returns whether neighbouring columns of op(X) are neighbouring elements.
                [[nodiscard]] inline constexpr bool has_contiguous_rows(
                    const OperandAccess& access) noexcept
                {
                    return access.col == 1;
                }

                /// @brief Returns false: a row read through another brick is not known contiguous.
                template<typename Access>
                [[nodiscard]] inline constexpr bool has_contiguous_rows(const Access&) noexcept
                {
                    return false;
                }

                /// @brief Returns whether neighbouring rows of op(X) are neighbouring elements.
                [[nodiscard]] inline constexpr bool has_contiguous_columns(
                    const OperandAccess& access) noexcept
                {
                    return access.row == 1;
                }

                /// @brief Returns false: a column read through another brick is not known contiguous.
                template<typename Access>
                [[nodiscard]] inline constexpr bool has_contiguous_columns(const Access&) noexcept
                {
                    return false;
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

                    /** @brief Leading dimension of the B tile, padded off a bank multiple. */
                    static constexpr int K_TILE_B_LD = K_BLOCK_N + 2;

                    /**
                     * @brief Returns the local memory one work-group needs, in bytes.
                     * @see modules/kernels/blas_seam_kernels/README.md
                     */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
                    {
                        return static_cast<std::size_t>(K_BUFFERS) *
                               (static_cast<std::size_t>(K_TILE_A_ELEMENTS) +
                                static_cast<std::size_t>(K_TILE_B_ELEMENTS)) * sizeof(T);
                    }

                    /** @brief Elements of one staged A tile, padding included. */
                    static constexpr int K_TILE_A_ELEMENTS = K_BLOCK_M * K_TILE_A_LD;

                    /** @brief Elements of one staged B tile, padding included. */
                    static constexpr int K_TILE_B_ELEMENTS = K_TILE_K * K_TILE_B_LD;

                    /** @brief Tiles of each operand: one is read while the next is staged. */
                    static constexpr int K_BUFFERS = 2;
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
                 * @brief Holds the compile-time geometry of the DEPTH_SPLIT variant for element type T.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                struct GemmDepthSplitTile
                {
                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_ROWS = register_block_dim<Accumulator<T>>(K_NO_LOCAL_ACC_BYTES, 4);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int K_ITEM_COLS = K_ITEM_ROWS;

                    /** @brief Elements of C one work-group produces. */
                    static constexpr int K_BLOCK_ITEMS = K_ITEM_ROWS * K_ITEM_COLS;

                    /** @brief Most work-items of a group that sit side by side along a row of op(B). */
                    static constexpr std::size_t K_MAX_COLUMN_LANES = 16;

                    /** @brief K-steps every work-item of a group must have before the variant is chosen. */
                    static constexpr int K_MIN_ITEM_DEPTH = 16;

                    /**
                     * @brief Returns the local memory one work-group of width work-items needs, in bytes.
                     * @see modules/kernels/blas_seam_kernels/README.md
                     */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes(std::size_t width) noexcept
                    {
                        return width * static_cast<std::size_t>(K_BLOCK_ITEMS) * sizeof(Accumulator<T>);
                    }
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
                    while ((side * 2) * (side * 2) <= budget)
                        side *= 2;

                    // A non-square budget leaves room for a second row of sub-groups, which is taken; see modules/kernels/blas_seam_kernels/README.md.
                    std::size_t rows = side;
                    if (rows * 2 * side <= budget)
                        rows *= 2;

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

                    SB_ASSERT(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, Device::hosts_local_tile(
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

                    // Each operand is staged along the axis its elements are adjacent on.
                    const StagePlan a_plan =
                        stage_plan<Configuration::K_BLOCK_M, Configuration::K_TILE_K>(
                            a, a_access, str_a);
                    const StagePlan b_plan =
                        stage_plan<Configuration::K_TILE_K, Configuration::K_BLOCK_N>(
                            b, b_access, str_b);

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        sycl::local_accessor<T, 1> tile_a(
                            sycl::range<1>(static_cast<std::size_t>(Configuration::K_BUFFERS) *
                                           Configuration::K_TILE_A_ELEMENTS),
                            h);
                        sycl::local_accessor<T, 1> tile_b(
                            sycl::range<1>(static_cast<std::size_t>(Configuration::K_BUFFERS) *
                                           Configuration::K_TILE_B_ELEMENTS),
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

                            // The whole group stages block_m x tile_k of A and tile_k x block_n of B; see modules/kernels/blas_seam_kernels/README.md.
                            const auto stage = [&](int buffer, int64_t depth)
                            {
                                stage_block<Configuration::K_BLOCK_M, Configuration::K_TILE_K,
                                            Configuration::K_TILE_A_LD,
                                            Configuration::K_WG_ITEMS>(
                                    tile_a, buffer * Configuration::K_TILE_A_ELEMENTS,
                                    local_linear, A, row0, depth, m, k, a_access, a_plan);
                                stage_block<Configuration::K_TILE_K, Configuration::K_BLOCK_N,
                                            Configuration::K_TILE_B_LD,
                                            Configuration::K_WG_ITEMS>(
                                    tile_b, buffer * Configuration::K_TILE_B_ELEMENTS,
                                    local_linear, B, depth, col0, k, n, b_access, b_plan);
                            };

                            // Two buffers and one barrier per K-step; see modules/kernels/blas_seam_kernels/README.md.
                            stage(0, 0);
                            sycl::group_barrier(item.get_group());
                            int current = 0;
                            for (int64_t k0 = 0; k0 < k; k0 += Configuration::K_TILE_K)
                            {
                                if (k0 + Configuration::K_TILE_K < k)
                                    stage(1 - current, k0 + Configuration::K_TILE_K);

                                const int a_base = current * Configuration::K_TILE_A_ELEMENTS;
                                const int b_base = current * Configuration::K_TILE_B_ELEMENTS;

                                // Staged out-of-range elements are zero, so the full tile_k width sums exactly; see modules/kernels/blas_seam_kernels/README.md.
                                for (int kk = 0; kk < Configuration::K_TILE_K; ++kk)
                                {
                                    // tile_a_ld is padded to 17; see modules/kernels/blas_seam_kernels/README.md.
                                    T reg_a[Configuration::K_ITEM_ROWS];
                                    T reg_b[Configuration::K_ITEM_COLS];

                                    for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                        reg_a[ii] =
                                            tile_a[a_base +
                                                   (local_row + ii * Configuration::K_WG_ROWS) *
                                                   Configuration::K_TILE_A_LD + kk];
                                    for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                        reg_b[jj] =
                                            tile_b[b_base + kk * Configuration::K_TILE_B_LD +
                                                   (local_col + jj * Configuration::K_WG_COLS)];

                                    for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                        for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                            acc[ii][jj] =
                                                Kernels::Detail::multiply_add(W(reg_a[ii]), W(reg_b[jj]), acc[ii][jj]);
                                }

                                // The next tile is staged, and no work-item reads this one again.
                                sycl::group_barrier(item.get_group());
                                current = 1 - current;
                            }

                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                            {
                                const int64_t global_row =
                                    row0 + local_row + ii * Configuration::K_WG_ROWS;
                                if (global_row >= m)
                                    continue;

                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                {
                                    const int64_t global_col =
                                        col0 + local_col + jj * Configuration::K_WG_COLS;
                                    if (global_col >= n)
                                        continue;

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
                                        acc[ii][jj] =
                                            Kernels::Detail::multiply_add(W(reg_a[ii]), W(reg_b[jj]), acc[ii][jj]);
                            }

                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                            {
                                const int64_t global_row = row0 + ii * wg_rows;
                                if (global_row >= m)
                                    continue;

                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                {
                                    const int64_t global_col = col0 + jj * wg_cols;
                                    if (global_col >= n)
                                        continue;

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
                 * @brief Holds the work-group geometry DEPTH_SPLIT launches one product with.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                struct DepthSplitPlan
                {
                    /** @brief Work-items in a work-group; 0 when the product keeps another variant. */
                    std::size_t width = 0;

                    /** @brief Work-items of a group that sit side by side along a row of op(B). */
                    std::size_t column_lanes = 1;

                    /** @brief Returns the work-items of a group that share the K extent. */
                    [[nodiscard]] constexpr std::size_t depth_lanes() const noexcept
                    {
                        return width / column_lanes;
                    }
                };

                /**
                 * @brief Plans DEPTH_SPLIT for one product; the width is 0 when the product keeps
                 *        one of the other two variants.
                 * @param b_rows_contiguous Whether neighbouring columns of op(B) are neighbouring elements.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                [[nodiscard]] inline DepthSplitPlan plan_depth_split(const Device::Profile& profile,
                                                                     int64_t m, int64_t n, int64_t k,
                                                                     int64_t batch_size,
                                                                     bool b_rows_contiguous) noexcept
                {
                    using Tiled = GemmLocalTile<T>;
                    using Split = GemmDepthSplitTile<T>;

                    if (!profile.has_dedicated_local_memory)
                        return DepthSplitPlan{};

                    const std::size_t width = Reduce::group_width(profile);
                    if (!Device::hosts_local_tile(profile, {Split::local_mem_bytes(width), width}))
                        return DepthSplitPlan{};

                    // A tiled launch with fewer groups than compute units leaves units idle.
                    const int64_t units = static_cast<int64_t>(profile.compute_units);
                    const int64_t tiled_groups = batch_size * group_count(m, Tiled::K_BLOCK_M) *
                                                 group_count(n, Tiled::K_BLOCK_N);
                    if (tiled_groups >= units)
                        return DepthSplitPlan{};

                    if (k < static_cast<int64_t>(width) * Split::K_MIN_ITEM_DEPTH)
                        return DepthSplitPlan{};

                    DepthSplitPlan plan{width, 1};
                    if (!b_rows_contiguous)
                        return plan;

                    // Column lanes double while every compute unit still has a group; see modules/kernels/blas_seam_kernels/README.md.
                    const int64_t row_blocks = batch_size * group_count(m, Split::K_ITEM_ROWS);
                    while (plan.column_lanes * 2 <= Split::K_MAX_COLUMN_LANES &&
                           plan.column_lanes * 2 <= width &&
                           row_blocks * group_count(n, static_cast<int64_t>(plan.column_lanes * 2) *
                                                           Split::K_ITEM_COLS) >= units)
                        plan.column_lanes *= 2;

                    return plan;
                }

                /**
                 * @brief Computes GEMM with one work-group per block of C, whose work-items share
                 *        the K extent and fold their sums in a fixed-order tree.
                 * @pre plan is a return of plan_depth_split for this product with a non-zero width.
                 * @param beta Exactly zero skips reading C.
                 * @param c Must not overlap A or B.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event gemm_depth_split(
                    sycl::queue& queue, DepthSplitPlan plan,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using W = Accumulator<T>;
                    using Configuration = GemmDepthSplitTile<T>;

                    const std::size_t width = plan.width;
                    const std::size_t column_lanes = plan.column_lanes;
                    const std::size_t depth_lanes = plan.depth_lanes();
                    const int64_t block_cols =
                        static_cast<int64_t>(column_lanes) * Configuration::K_ITEM_COLS;

                    const int64_t n_blocks = group_count(n, block_cols);
                    const int64_t blocks = group_count(m, Configuration::K_ITEM_ROWS) * n_blocks;
                    const std::size_t global =
                        static_cast<std::size_t>(batch_size * blocks) * width;

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        sycl::local_accessor<W, 1> tile(
                            sycl::range<1>(static_cast<std::size_t>(Configuration::K_BLOCK_ITEMS) *
                                           width),
                            h);

                        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global),
                                                         sycl::range<1>(width)),
                                       [=](sycl::nd_item<1> item)
                        {
                            const int64_t group = static_cast<int64_t>(item.get_group(0));
                            const int64_t batch_id = group / blocks;
                            const int64_t block = group % blocks;
                            const std::size_t lane = item.get_local_id(0);

                            // Neighbouring work-items take neighbouring columns, then neighbouring K-steps; see modules/kernels/blas_seam_kernels/README.md.
                            const std::size_t column_lane = lane % column_lanes;
                            const std::size_t depth_lane = lane / column_lanes;

                            const int64_t row0 = (block / n_blocks) * Configuration::K_ITEM_ROWS;
                            const int64_t col0 = (block % n_blocks) * block_cols +
                                static_cast<int64_t>(column_lane) * Configuration::K_ITEM_COLS;

                            const T* A = a + batch_id * str_a;
                            const T* B = b + batch_id * str_b;

                            // Only C is __restrict__: a and b may alias; see modules/kernels/blas_seam_kernels/README.md.
                            T* __restrict__ C = c + batch_id * str_c;

                            W acc[Configuration::K_ITEM_ROWS][Configuration::K_ITEM_COLS];
                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    acc[ii][jj] = W(0);

                            const int64_t step = static_cast<int64_t>(depth_lanes);
                            for (int64_t kk = static_cast<int64_t>(depth_lane); kk < k; kk += step)
                            {
                                T reg_a[Configuration::K_ITEM_ROWS];
                                T reg_b[Configuration::K_ITEM_COLS];

                                for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                    reg_a[ii] = load_operand(A, row0 + ii, kk, m, k, a_access);
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    reg_b[jj] = load_operand(B, kk, col0 + jj, k, n, b_access);

                                for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                    for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                        acc[ii][jj] =
                                            Kernels::Detail::multiply_add(W(reg_a[ii]), W(reg_b[jj]), acc[ii][jj]);
                            }

                            // Element e of a work-item's block keeps its partial at tile[e * width + lane].
                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                    tile[static_cast<std::size_t>(
                                             ii * Configuration::K_ITEM_COLS + jj) * width + lane] =
                                        acc[ii][jj];

                            sycl::group_barrier(item.get_group());

                            // depth_lanes is a power of two, so the halving tree pairs the depth lanes of one column lane in a fixed order.
                            for (std::size_t off = depth_lanes / 2; off > 0; off >>= 1)
                            {
                                if (depth_lane < off)
                                {
                                    for (std::size_t e = 0;
                                         e < static_cast<std::size_t>(Configuration::K_BLOCK_ITEMS);
                                         ++e)
                                        tile[e * width + lane] =
                                            tile[e * width + lane] +
                                            tile[e * width + lane + off * column_lanes];
                                }
                                sycl::group_barrier(item.get_group());
                            }

                            // The work-items of depth lane 0 hold the sums and store their own blocks.
                            if (depth_lane != 0)
                                return;

                            for (int ii = 0; ii < Configuration::K_ITEM_ROWS; ++ii)
                            {
                                const int64_t global_row = row0 + ii;
                                if (global_row >= m)
                                    continue;

                                for (int jj = 0; jj < Configuration::K_ITEM_COLS; ++jj)
                                {
                                    const int64_t global_col = col0 + jj;
                                    if (global_col >= n)
                                        continue;

                                    const int64_t c_idx = c_access.index(global_row, global_col);
                                    const W sum = tile[static_cast<std::size_t>(
                                        ii * Configuration::K_ITEM_COLS + jj) * width + lane];

                                    // beta == 0 must not read the prior C value, which may be uninitialised; see modules/kernels/blas_seam_kernels/README.md.
                                    const T value = (beta == T(0))
                                        ? T(multiply(W(alpha), sum))
                                        : T(multiply(W(alpha), sum) +
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

                const Detail::DepthSplitPlan split = Detail::plan_depth_split<T>(
                    profile, m, n, k, batch_size, Detail::has_contiguous_rows(b_access));
                if (split.width != 0)
                {
                    SB_LOG_DEBUG(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "SYCL GEMM [depth_split]: batch={} {}x{}x{} item_block={}x{} "
                                "width={} column_lanes={}",
                                batch_size, m, n, k,
                                Detail::GemmDepthSplitTile<T>::K_ITEM_ROWS,
                                Detail::GemmDepthSplitTile<T>::K_ITEM_COLS, split.width,
                                split.column_lanes);

                    return Detail::gemm_depth_split<T>(
                        queue, split, m, n, k,
                        alpha, a, a_access, str_a,
                        b, b_access, str_b,
                        beta, c, c_access, str_c,
                        batch_size, deps, epilogue);
                }

                // Every variant is instantiated here, whichever branch runs; see modules/kernels/blas_seam_kernels/README.md.
                if (profile.has_dedicated_local_memory && tile_hosted)
                {
                    SB_LOG_DEBUG(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "SYCL GEMM [local_tiled]: batch={} {}x{}x{} block={}x{} tile_k={} local={}B",
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
                    SB_LOG_WARN(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "SYCL GEMM: local tile wants {} B and {} work-items, device offers "
                                "{} B and {}; falling back to the no-local variant.",
                                local_bytes, local_items,
                                profile.local_mem_bytes, profile.max_work_group_size);

                SB_LOG_DEBUG(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "SYCL GEMM [no_local]: batch={} {}x{}x{} item_block={}x{}",
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
