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
 * @brief Defines the reference kernel of the product of a sparse matrix and a dense matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/structure.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes C = alpha * op(A) * B + beta * C for a CSR matrix A and dense B, C.
             * @param nrhs Columns of B and of C.
             * @param beta Exactly zero skips reading C.
             * @pre C overlaps neither B nor the arrays of A.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void spmm(Core::Layout layout, Core::Transpose trans, const CsrPattern<Index>& pattern,
                      const T* values, int64_t nrhs, T alpha, const T* b, int64_t ldb, T beta, T* c,
                      int64_t ldc)
            {
                using RW = ReferenceAccumulator<T>;

                SB_ASSERT(find_defect(pattern) == StructureDefect::None,
                          "spmm was given a CSR structure that breaks its contract");

                const OperandAccess at_b = operand_access(layout, Core::Transpose::NoTrans, ldb);
                const OperandAccess at_c = operand_access(layout, Core::Transpose::NoTrans, ldc);
                const bool keeps_c = !(beta == T(0));

                if (!Core::is_transposed(trans))
                {
                    for (int64_t i = 0; i < pattern.rows; ++i)
                    {
                        for (int64_t r = 0; r < nrhs; ++r)
                        {
                            ReferenceSum<T> sum;
                            for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                                sum.add(RW(values[k]) * RW(b[at_b.index(pattern.column(k), r)]));

                            T& out = c[at_c.index(i, r)];
                            out = keeps_c
                                ? static_cast<T>(RW(alpha) * sum.value() + RW(beta) * RW(out))
                                : static_cast<T>(RW(alpha) * sum.value());
                        }
                    }
                    return;
                }

                const bool conjugated = Core::is_conjugated(trans);
                for (int64_t r = 0; r < nrhs; ++r)
                {
                    // sums[j] collects column j of A against column r of B, in ascending row order.
                    std::vector<ReferenceSum<T>> sums(static_cast<std::size_t>(pattern.cols));
                    for (int64_t i = 0; i < pattern.rows; ++i)
                    {
                        const RW bi = RW(b[at_b.index(i, r)]);
                        for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                            sums[static_cast<std::size_t>(pattern.column(k))].add(
                                RW(conjugate_if(conjugated, values[k])) * bi);
                    }

                    for (int64_t j = 0; j < pattern.cols; ++j)
                    {
                        const RW sum = sums[static_cast<std::size_t>(j)].value();
                        T& out = c[at_c.index(j, r)];
                        out = keeps_c ? static_cast<T>(RW(alpha) * sum + RW(beta) * RW(out))
                                      : static_cast<T>(RW(alpha) * sum);
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
