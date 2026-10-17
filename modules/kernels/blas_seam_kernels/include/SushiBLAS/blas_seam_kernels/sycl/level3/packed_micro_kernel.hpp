/**************************************************************************/
/* packed_micro_kernel.hpp                                                */
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
 * @file packed_micro_kernel.hpp
 * @brief Declares the micro-kernel of a packed product, which accumulates one register block.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/arithmetic/multiply_add.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Loads Lanes consecutive reals as one vector.
                 * @pre p is aligned to the size of the vector.
                 */
                template<typename R, int Lanes>
                [[nodiscard]] inline sycl::vec<R, Lanes> load_lanes(const R* p) noexcept
                {
                    using Row = sycl::vec<R, Lanes>;
                    static_assert(sizeof(Row) == sizeof(R) * static_cast<std::size_t>(Lanes),
                                  "A row of lanes must be exactly its elements.");
                    static_assert(alignof(Row) <= K_PACKED_ALIGNMENT,
                                  "A row of lanes must fit the alignment of a packed panel.");
                    return *reinterpret_cast<const Row*>(p);
                }

                /**
                 * @brief Stores one vector as Lanes consecutive reals.
                 * @pre p is aligned to the size of the vector.
                 */
                template<typename R, int Lanes>
                inline void store_lanes(R* p, const sycl::vec<R, Lanes>& row) noexcept
                {
                    *reinterpret_cast<sycl::vec<R, Lanes>*>(p) = row;
                }

                /**
                 * @brief Adds depth outer products of real slices to a Rows x Cols micro-tile.
                 * @param a Holds depth slices of Rows reals.
                 * @param b Holds depth slices of Cols reals.
                 * @param tile Holds Rows rows of Cols reals; read before the first term, written
                 *        after the last.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename R, int Rows, int Cols, std::size_t... I>
                inline void accumulate_real(int64_t depth, const R* a, const R* b, R* tile,
                                            std::index_sequence<I...>) noexcept
                {
                    using Row = sycl::vec<R, Cols>;

                    std::array<Row, Rows> sum{load_lanes<R, Cols>(tile + I * Cols)...};
                    for (int64_t l = 0; l < depth; ++l)
                    {
                        const Row b_slice = load_lanes<R, Cols>(b + l * Cols);
                        const R* a_slice = a + l * Rows;
                        ((sum[I] = Kernels::Detail::multiply_add(Row(a_slice[I]), b_slice,
                                                                 sum[I])), ...);
                    }
                    (store_lanes<R, Cols>(tile + I * Cols, sum[I]), ...);
                }

                /**
                 * @brief Adds depth outer products of complex slices to a Rows x Cols micro-tile.
                 * @param a Holds depth slices of Rows real parts followed by Rows imaginary parts.
                 * @param b Holds depth slices of Cols real parts followed by Cols imaginary parts.
                 * @param tile Holds Rows rows of Cols real parts, then as many imaginary parts.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename R, int Rows, int Cols, std::size_t... I>
                inline void accumulate_complex(int64_t depth, const R* a, const R* b, R* tile,
                                               std::index_sequence<I...>) noexcept
                {
                    using Row = sycl::vec<R, Cols>;
                    constexpr int plane = Rows * Cols;

                    std::array<Row, Rows> sum_re{load_lanes<R, Cols>(tile + I * Cols)...};
                    std::array<Row, Rows> sum_im{load_lanes<R, Cols>(tile + plane + I * Cols)...};
                    for (int64_t l = 0; l < depth; ++l)
                    {
                        const Row b_re = load_lanes<R, Cols>(b + l * 2 * Cols);
                        const Row b_im = load_lanes<R, Cols>(b + l * 2 * Cols + Cols);
                        const R* a_slice = a + l * 2 * Rows;
                        (Kernels::Detail::multiply_add_parts(sum_re[I], sum_im[I], Row(a_slice[I]),
                                            Row(a_slice[Rows + I]), b_re, b_im), ...);
                    }
                    (store_lanes<R, Cols>(tile + I * Cols, sum_re[I]), ...);
                    (store_lanes<R, Cols>(tile + plane + I * Cols, sum_im[I]), ...);
                }

                /**
                 * @brief Adds depth terms of op(A) * op(B) to one micro-tile, in ascending order.
                 * @tparam W The accumulation type; its PackedRegisterBlock fixes the tile.
                 * @pre b and tile are aligned to K_PACKED_ALIGNMENT bytes.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename W>
                inline void packed_micro_kernel(int64_t depth, const RealType<W>* a,
                                                const RealType<W>* b, RealType<W>* tile) noexcept
                {
                    using Block = PackedRegisterBlock<W>;
                    using R = RealType<W>;
                    constexpr int rows = Block::K_ROWS;
                    constexpr int cols = Block::K_COLS;
                    const std::make_index_sequence<Block::K_ROWS> each_row{};

                    if constexpr (Core::is_complex_v<W>)
                        accumulate_complex<R, rows, cols>(depth, a, b, tile, each_row);
                    else
                        accumulate_real<R, rows, cols>(depth, a, b, tile, each_row);
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
