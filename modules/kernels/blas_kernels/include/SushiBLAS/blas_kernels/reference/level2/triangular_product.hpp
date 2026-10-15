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
 * @brief Defines the reference triangular matrix-vector product TBMV and TPMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /**
                 * @brief Computes x = op(A) * x in place for a triangular op(A) read through
                 *        access.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param op_upper Whether op(A) is upper triangular.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                void triangular_product(const Access& access, bool op_upper, bool unit_diag,
                                        int64_t n, const T* a, T* x, int64_t incx)
                {
                    using RW = ReferenceAccumulator<T>;

                    auto row = [&](int64_t i)
                    {
                        const int64_t end = access.end_column(i, n);

                        ReferenceSum<T> sum;
                        const int64_t first = access.first_column(i);
                        RowCursor cursor = access.row_cursor(i, first);
                        for (int64_t k = first; k < end; ++k, cursor.advance())
                        {
                            const T aval = (k == i && unit_diag) ? T(1) : access.at(a, cursor);
                            sum.add(RW(aval) * RW(x[k * incx]));
                        }
                        x[i * incx] = T(sum.value());
                    };

                    // Row i reads only elements of x that no earlier row has overwritten.
                    if (op_upper)
                    {
                        for (int64_t i = 0; i < n; ++i)
                            row(i);
                    }
                    else
                    {
                        for (int64_t i = n - 1; i >= 0; --i)
                            row(i);
                    }
                }
            } // namespace Detail
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
