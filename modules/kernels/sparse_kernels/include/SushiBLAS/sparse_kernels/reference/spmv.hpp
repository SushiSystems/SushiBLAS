/**************************************************************************/
/* spmv.hpp                                                               */
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
 * @file spmv.hpp
 * @brief Defines the reference kernel of the sparse matrix-vector product.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/structure.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes y = alpha * op(A) * x + beta * y for a CSR matrix A, row by row.
             * @param values One value per stored entry of pattern.
             * @param beta Exactly zero skips reading y.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void spmv(Core::Transpose trans, const CsrPattern<Index>& pattern, const T* values,
                      T alpha, const T* x, int64_t incx, T beta, T* y, int64_t incy)
            {
                using RW = ReferenceAccumulator<T>;

                SB_ASSERT(find_defect(pattern) == StructureDefect::None,
                          "spmv was given a CSR structure that breaks its contract");

                const bool keeps_y = !(beta == T(0));

                if (!Core::is_transposed(trans))
                {
                    for (int64_t i = 0; i < pattern.rows; ++i)
                    {
                        ReferenceSum<T> sum;
                        for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            sum.add(RW(values[k]) * RW(x[pattern.column(k) * incx]));

                        T& out = y[i * incy];
                        out = keeps_y ? static_cast<T>(RW(alpha) * sum.value() + RW(beta) * RW(out))
                                      : static_cast<T>(RW(alpha) * sum.value());
                    }
                    return;
                }

                // sums[j] collects column j of A in ascending row order.
                const bool conjugated = Core::is_conjugated(trans);
                std::vector<ReferenceSum<T>> sums(static_cast<std::size_t>(pattern.cols));
                for (int64_t i = 0; i < pattern.rows; ++i)
                {
                    const RW xi = RW(x[i * incx]);
                    for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                        sums[static_cast<std::size_t>(pattern.column(k))].add(
                            RW(conjugate_if(conjugated, values[k])) * xi);
                }

                for (int64_t j = 0; j < pattern.cols; ++j)
                {
                    const RW sum = sums[static_cast<std::size_t>(j)].value();
                    T& out = y[j * incy];
                    out = keeps_y ? static_cast<T>(RW(alpha) * sum + RW(beta) * RW(out))
                                  : static_cast<T>(RW(alpha) * sum);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
