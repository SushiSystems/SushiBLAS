/**************************************************************************/
/* spmm.hpp                                                               */
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
 * @file spmm.hpp
 * @brief Defines the SYCL kernel of the product of a sparse matrix and a dense matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>
#include "kernels/sparse/csr_pattern.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes C = alpha * op(A) * B + beta * C for a CSR matrix A and dense B, C.
             * @param beta Exactly zero skips reading C.
             * @param work One Accumulator<T> per element of C for a transposed product when
             *        RunningSolution<T>::K_NEEDS_WORKSPACE holds; never read otherwise.
             * @pre C overlaps neither B nor the arrays of A.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event spmm(sycl::queue& queue, Core::Layout layout,
                                           Core::Transpose trans, CsrPattern<Index> pattern,
                                           const T* values, int64_t nrhs, T alpha, const T* b,
                                           int64_t ldb, T beta, T* c, int64_t ldc,
                                           Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                SB_LOG_INFO("SYCL SPMM: {}x{}, {} entries, {} columns", pattern.rows, pattern.cols,
                            pattern.nnz, nrhs);

                const OperandAccess at_b = operand_access(layout, Core::Transpose::NoTrans, ldb);
                const OperandAccess at_c = operand_access(layout, Core::Transpose::NoTrans, ldc);
                const bool keeps_c = !(beta == T(0));

                if (!Core::is_transposed(trans))
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Launch::indexed(h, queue, static_cast<std::size_t>(pattern.rows * nrhs),
                                        [=](std::size_t index)
                        {
                            const int64_t i = static_cast<int64_t>(index) / nrhs;
                            const int64_t r = static_cast<int64_t>(index) % nrhs;

                            W sum(0);
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                                sum += multiply(W(values[k]),
                                                W(b[at_b.index(pattern.column(k), r)]));

                            const W scaled = multiply(W(alpha), sum);
                            const int64_t out = at_c.index(i, r);
                            c[out] = keeps_c ? static_cast<T>(scaled + multiply(W(beta), W(c[out])))
                                             : static_cast<T>(scaled);
                        });
                    });
                }

                const bool conjugated = Core::is_conjugated(trans);
                const int64_t outputs = pattern.cols;

                // One work-item owns one column of C, so no element has two writers.
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(nrhs), [=](std::size_t index)
                    {
                        const int64_t r = static_cast<int64_t>(index);
                        const RunningSolution<T> column(c, work, 0);

                        for (int64_t j = 0; j < outputs; ++j)
                        {
                            const int64_t out = at_c.index(j, r);
                            column.set(out, keeps_c ? multiply(W(beta), W(c[out])) : W(0));
                        }

                        for (int64_t i = 0; i < pattern.rows; ++i)
                        {
                            const W scaled = multiply(W(alpha), W(b[at_b.index(i, r)]));
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            {
                                const int64_t out = at_c.index(pattern.column(k), r);
                                column.set(out, column.get(out) +
                                                    multiply(W(conjugate_if(conjugated, values[k])),
                                                             scaled));
                            }
                        }
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
