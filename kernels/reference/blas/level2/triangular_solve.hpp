/**************************************************************************/
/* triangular_solve.hpp                                                   */
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
 * @file triangular_solve.hpp
 * @brief Defines the reference triangular substitution TBSV and TPSV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/kernels/blas/row_cursor.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /**
                 * @brief Solves op(A) * x = b in place for a triangular op(A) read through access,
                 *        where x holds b on entry.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param op_upper Whether op(A) is upper triangular.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                void triangular_solve(const Access& access, bool op_upper, bool unit_diag,
                                      int64_t n, const T* a, T* x, int64_t incx)
                {
                    using W = Accumulator<T>;
                    using RW = ReferenceAccumulator<T>;

                    const std::size_t span =
                        (n > 0) ? static_cast<std::size_t>((n - 1) * incx + 1) : std::size_t{0};
                    std::vector<W> wide(RunningSolution<T>::K_NEEDS_WORKSPACE ? span
                                                                              : std::size_t{0});
                    const RunningSolution<T> solution(x, wide.data(), 0);

                    auto solve = [&](int64_t i, int64_t first, int64_t end)
                    {
                        ReferenceSum<T> sum;
                        sum.add(RW(x[i * incx]));
                        RowCursor cursor = access.row_cursor(i, first);
                        for (int64_t c = first; c < end; ++c, cursor.advance())
                            sum.add(-(RW(access.at(a, cursor)) * RW(solution.get(c * incx))));

                        solution.set(i * incx,
                                     static_cast<W>(unit_diag
                                                        ? sum.value()
                                                        : sum.value() / RW(access.at(a, i, i))));
                    };

                    // Row i reads only elements of the solution that are already solved.
                    if (op_upper)
                    {
                        for (int64_t i = n - 1; i >= 0; --i)
                            solve(i, i + 1, access.end_column(i, n));
                    }
                    else
                    {
                        for (int64_t i = 0; i < n; ++i)
                            solve(i, access.first_column(i), i);
                    }
                }
            } // namespace Detail
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
