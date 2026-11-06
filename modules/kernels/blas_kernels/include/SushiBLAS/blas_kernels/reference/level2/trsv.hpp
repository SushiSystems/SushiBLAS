/**************************************************************************/
/* trsv.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file trsv.hpp
 * @brief Defines the reference kernel of TRSV.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes TRSV: solves op(A) * x = b in place for a dense triangular A, where
             *        x holds b on entry. Only the stored triangle of A is read.
             * @see kernels/README.md
             */
            template<typename T>
            void trsv(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                      int64_t n, const T* a, int64_t lda, T* x, int64_t incx)
            {
                using W = Accumulator<T>;
                using RW = ReferenceAccumulator<T>;

                const OperandAccess access = operand_access(layout, trans, lda);
                const bool unit_diag = (diag == Core::Diag::Unit);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);

                const std::size_t span =
                    (n > 0) ? static_cast<std::size_t>((n - 1) * incx + 1) : std::size_t{0};
                std::vector<W> wide(RunningSolution<T>::K_NEEDS_WORKSPACE ? span : std::size_t{0});
                const RunningSolution<T> solution(x, wide.data(), 0);

                auto solve = [&](int64_t i, int64_t first, int64_t end)
                {
                    ReferenceSum<T> sum;
                    sum.add(RW(x[i * incx]));
                    for (int64_t c = first; c < end; ++c)
                        sum.add(-(RW(access.at(a, i, c)) * RW(solution.get(c * incx))));

                    solution.set(i * incx,
                                 static_cast<W>(unit_diag ? sum.value()
                                                          : sum.value() / RW(access.at(a, i, i))));
                };

                // Row i reads only elements of the solution that are already solved.
                if (op_upper)
                {
                    for (int64_t i = n - 1; i >= 0; --i)
                        solve(i, i + 1, n);
                }
                else
                {
                    for (int64_t i = 0; i < n; ++i)
                        solve(i, 0, i);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
