/**************************************************************************/
/* triangular_product.hpp                                                 */
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
 * @file triangular_product.hpp
 * @brief Defines the SYCL triangular matrix-vector product TBMV and TPMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>
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
                 * @brief Computes one row of x = op(A) * x in one work-item, in ascending column
                 *        order, from a snapshot of x.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                struct TriangularProductRow
                {
                    Access access;               /**< How op(A) is read. */
                    const T* a = nullptr;        /**< The stored triangle. */
                    const T* snapshot = nullptr; /**< The values x held before the call. */
                    T* x = nullptr;              /**< The vector the product overwrites. */
                    int64_t incx = 1;            /**< Distance between consecutive elements of x. */
                    int64_t n = 0;               /**< Rows and columns of op(A). */
                    bool unit_diag = false;      /**< Whether the diagonal is one and never read. */

                    /** @brief Computes row index and stores it into x, rounded once to T. */
                    void operator()(std::size_t index) const
                    {
                        using W = Accumulator<T>;

                        const int64_t i = static_cast<int64_t>(index);
                        const int64_t end = access.end_column(i, n);

                        const int64_t first = access.first_column(i);

                        W sum(0);
                        RowCursor cursor = access.row_cursor(i, first);
                        for (int64_t k = first; k < end; ++k, cursor.advance())
                        {
                            const T aval = (k == i && unit_diag) ? T(1) : access.at(a, cursor);
                            sum += multiply(W(aval), W(snapshot[k]));
                        }
                        x[i * incx] = T(sum);
                    }
                };

                /**
                 * @brief Computes x = op(A) * x in place for a triangular op(A) read through
                 *        access, one work-item per row.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param work Device scratch of n elements that no other task touches; may be null
                 *        when n is 0.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                [[nodiscard]] sycl::event triangular_product(sycl::queue& queue,
                                                             const Access& access, bool unit_diag,
                                                             int64_t n, const T* a, T* x,
                                                             int64_t incx, T* work,
                                                             const std::vector<sycl::event>& deps)
                {
                    const VectorSnapshot<T> snapshot{x, incx, work};
                    const TriangularProductRow<T, Access> row{access, a, work, x, incx, n,
                                                              unit_diag};
                    const std::size_t rows = static_cast<std::size_t>(n);

                    const sycl::event copied = queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Launch::indexed(h, queue, rows, snapshot);
                    });

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(copied);
                        Launch::indexed(h, queue, rows, row);
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
