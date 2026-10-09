/**************************************************************************/
/* sptrsv.hpp                                                             */
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
 * @file sptrsv.hpp
 * @brief Defines the SYCL kernel of the sparse triangular solve.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Solves op(A) * x = b for a triangular CSR A in one work-item, x replacing b.
             * @param uplo The triangle of A that is read; entries of the other one are skipped.
             * @param diag UNIT skips the stored diagonal and takes it as all ones.
             * @param work One Accumulator<T> per element of b when
             *        RunningSolution<T>::K_NEEDS_WORKSPACE holds; never read otherwise.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event sptrsv(sycl::queue& queue, Core::Uplo uplo,
                                             Core::Transpose trans, Core::Diag diag,
                                             CsrPattern<Index> pattern, const T* values, T* b,
                                             Accumulator<T>* work,
                                             const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                SB_LOG_INFO("SYCL SPTRSV: order {}, {} entries", pattern.rows, pattern.nnz);

                const int64_t n = pattern.rows;
                const bool lower = (uplo == Core::Uplo::Lower);
                const bool unit_diag = (diag == Core::Diag::Unit);
                const bool transposed = Core::is_transposed(trans);
                const bool conjugated = Core::is_conjugated(trans);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        const RunningSolution<T> X(b, work, 0);

                        if (!transposed)
                        {
                            // A lower triangle is solved from row 0 down, an upper one upwards.
                            for (int64_t step = 0; step < n; ++step)
                            {
                                const int64_t i = lower ? step : n - 1 - step;

                                W sum = W(b[i]);
                                W diagonal(0);
                                for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                                {
                                    const int64_t j = pattern.column(k);
                                    if (j == i)
                                        diagonal += W(values[k]);
                                    else if (lower ? j < i : j > i)
                                        sum -= multiply(W(values[k]), X.get(j));
                                }
                                X.set(i, unit_diag ? sum : divide(sum, diagonal));
                            }
                            return;
                        }

                        for (int64_t j = 0; j < n; ++j)
                            X.set(j, W(b[j]));

                        // op(A) of a lower triangle is upper, so its rows are solved from n - 1 up.
                        for (int64_t step = 0; step < n; ++step)
                        {
                            const int64_t i = lower ? n - 1 - step : step;

                            W diagonal(0);
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            {
                                if (pattern.column(k) == i)
                                    diagonal += W(conjugate_if(conjugated, values[k]));
                            }

                            const W solved = unit_diag ? X.get(i) : divide(X.get(i), diagonal);
                            X.set(i, solved);

                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            {
                                const int64_t j = pattern.column(k);
                                if (lower ? j < i : j > i)
                                    X.set(j, X.get(j) -
                                                 multiply(W(conjugate_if(conjugated, values[k])),
                                                          solved));
                            }
                        }
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
