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
 * @brief Defines the SYCL triangular substitution TBSV and TPSV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Solves op(A) * x = b in place for a triangular op(A) read through access,
                 *        where x holds b on entry, in one sequential task.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param op_upper Whether op(A) is upper triangular.
                 * @param work (n - 1) * incx + 1 elements if RunningSolution<T> needs them.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                [[nodiscard]] sycl::event triangular_solve(sycl::queue& queue, const Access& access,
                                                           bool op_upper, bool unit_diag,
                                                           int64_t n, const T* a, T* x,
                                                           int64_t incx, Accumulator<T>* work,
                                                           const std::vector<sycl::event>& deps)
                {
                    using W = Accumulator<T>;

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.single_task([=]()
                        {
                            const RunningSolution<T> solution(x, work, 0);

                            auto solve = [=](int64_t i, int64_t first, int64_t end)
                            {
                                W sum = W(x[i * incx]);
                                for (int64_t c = first; c < end; ++c)
                                    sum -= multiply(W(access.at(a, i, c)),
                                                    solution.get(c * incx));

                                solution.set(i * incx,
                                             unit_diag ? sum
                                                       : divide(sum, W(access.at(a, i, i))));
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
                        });
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
