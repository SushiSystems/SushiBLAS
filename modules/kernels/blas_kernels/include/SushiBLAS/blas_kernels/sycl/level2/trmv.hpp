/**************************************************************************/
/* trmv.hpp                                                               */
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
 * @file trmv.hpp
 * @brief Defines the SYCL kernel of TRMV: a snapshot of x, then one row product per element.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/row_reduction.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/vector_snapshot.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Returns the term op(A)(row, k) * x(k) of a row product, in Accumulator<T>.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct TrmvTerm
                {
                    OperandAccess access;        /**< How op(A) is read. */
                    const T* a = nullptr;        /**< The triangular matrix. */
                    const T* snapshot = nullptr; /**< The values x held before the call. */
                    bool unit_diag = false;      /**< Whether the diagonal is one and never read. */

                    /** @brief Returns the term of column k in a row. */
                    [[nodiscard]] Accumulator<T> operator()(int64_t row, int64_t k) const
                    {
                        return k == row ? diagonal(row) : off_diagonal(row, k);
                    }

                    /** @brief Returns the term of the diagonal column of a row. */
                    [[nodiscard]] Accumulator<T> diagonal(int64_t row) const
                    {
                        using W = Accumulator<T>;
                        const T aval = unit_diag ? T(1) : access.at(a, row, row);
                        return multiply(W(aval), W(snapshot[row]));
                    }

                    /** @brief Returns the term of column k of a row, k not on the diagonal. */
                    [[nodiscard]] Accumulator<T> off_diagonal(int64_t row, int64_t k) const
                    {
                        using W = Accumulator<T>;
                        return multiply(W(access.at(a, row, k)), W(snapshot[k]));
                    }
                };

                /**
                 * @brief Stores one finished row product into x, rounded once to T.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct TrmvStore
                {
                    T* x = nullptr;   /**< The vector TRMV overwrites. */
                    int64_t incx = 1; /**< Distance between consecutive elements of x. */

                    /** @brief Stores the product of a row. */
                    void operator()(int64_t row, Accumulator<T> sum) const
                    {
                        x[row * incx] = T(sum);
                    }
                };

                /**
                 * @brief Computes one row product in one work-item, in ascending column order.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct TrmvRow
                {
                    TrmvTerm<T> term;    /**< The term of one column. */
                    TriangleRow columns; /**< The columns a row holds. */
                    TrmvStore<T> store;  /**< Where a finished row goes. */

                    /** @brief Computes and stores row index. */
                    void operator()(std::size_t index) const
                    {
                        const int64_t row = static_cast<int64_t>(index);
                        const RowWindow window = columns(row);

                        const bool diagonal_first = window.first == row;
                        const int64_t first = diagonal_first ? window.first + 1 : window.first;
                        const int64_t last = diagonal_first ? window.last : window.last - 1;

                        Accumulator<T> sum(0);
                        if (diagonal_first)
                            sum += term.diagonal(row);
                        for (int64_t k = first; k < last; ++k)
                            sum += term.off_diagonal(row, k);
                        if (!diagonal_first)
                            sum += term.diagonal(row);
                        store(row, sum);
                    }
                };
            } // namespace Detail

            /**
             * @brief Computes TRMV, x = op(A) * x, with a given work-group width per row.
             * @param group_width 0 for one work-item per row; else a power of two the device takes
             *        as a work-group size, and then n must be above zero.
             * @param work Device scratch of n elements that no other task touches; may be null when
             *        n is 0.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event trmv(sycl::queue& queue, std::size_t group_width,
                                           Core::Layout layout, Core::Uplo uplo,
                                           Core::Transpose trans, Core::Diag diag,
                                           int64_t n, const T* a, int64_t lda, T* x, int64_t incx,
                                           T* work, const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                const bool eff_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);
                const Detail::VectorSnapshot<T> snapshot{x, incx, work};
                const Detail::TrmvTerm<T> term{operand_access(layout, trans, lda), a, work,
                                               diag == Core::Diag::Unit};
                const Detail::TriangleRow columns{n, eff_upper};
                const Detail::TrmvStore<T> store{x, incx};
                const std::size_t rows = static_cast<std::size_t>(n);

                SB_LOG_INFO("SYCL TRMV: {}x{} group_width={}", n, n, group_width);

                const sycl::event copied = queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, rows, snapshot);
                });

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(copied);
                    if (group_width != 0)
                        Detail::reduce_row_windows_by_group<W>(h, n, group_width, columns, term,
                                                               store);
                    else
                        Launch::indexed(h, queue, rows, Detail::TrmvRow<T>{term, columns, store});
                });
            }

            /**
             * @brief Computes TRMV, x = op(A) * x, in the row shape the queue's device plans.
             * @param work Device scratch of n elements that no other task touches; may be null when
             *        n is 0.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event trmv(sycl::queue& queue, Core::Layout layout, Core::Uplo uplo,
                                           Core::Transpose trans, Core::Diag diag,
                                           int64_t n, const T* a, int64_t lda, T* x, int64_t incx,
                                           T* work, const std::vector<sycl::event>& deps)
            {
                // A row of op(A) is contiguous exactly when GEMV's row is; see kernels/README.md.
                const bool row_contiguous =
                    (layout == Core::Layout::RowMajor) != Core::is_transposed(trans);
                const std::size_t width = row_contiguous
                    ? Detail::row_reduction_width(Device::profile_for(queue), n)
                    : std::size_t{0};

                return trmv<T>(queue, width, layout, uplo, trans, diag, n, a, lda, x, incx, work,
                               deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
