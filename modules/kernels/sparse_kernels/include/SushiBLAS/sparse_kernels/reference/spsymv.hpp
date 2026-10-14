/**************************************************************************/
/* spsymv.hpp                                                             */
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
 * @file spsymv.hpp
 * @brief Defines the reference kernel of the symmetric and Hermitian sparse matrix-vector product.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/structure.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes y = alpha * A * x + beta * y for an A given by one stored triangle.
             * @tparam Conj Unconjugated for a symmetric A, Conjugated for a Hermitian one.
             * @param uplo The triangle of pattern that is read; entries of the other are skipped.
             * @param beta Exactly zero skips reading y.
             * @pre pattern is square.
             * @see kernels/README.md
             */
            template<typename T, typename Index, typename Conj = Unconjugated>
            void spsymv(Core::Uplo uplo, const CsrPattern<Index>& pattern, const T* values, T alpha,
                        const T* x, int64_t incx, T beta, T* y, int64_t incy)
            {
                using RW = ReferenceAccumulator<T>;

                SB_ASSERT(find_defect(pattern) == StructureDefect::None,
                          "spsymv was given a CSR structure that breaks its contract");

                const int64_t n = pattern.rows;
                const bool lower = (uplo == Core::Uplo::Lower);
                const bool keeps_y = !(beta == T(0));

                // sums[i] collects row i of the whole matrix: its stored entries and their mirrors.
                std::vector<ReferenceSum<T>> sums(static_cast<std::size_t>(n));
                for (int64_t i = 0; i < n; ++i)
                {
                    const RW xi = RW(x[i * incx]);
                    for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                    {
                        const int64_t j = pattern.column(k);
                        const RW value = RW(values[k]);
                        if (j == i)
                        {
                            sums[static_cast<std::size_t>(i)].add(Conj::diagonal(value) * xi);
                        }
                        else if (lower ? j < i : j > i)
                        {
                            sums[static_cast<std::size_t>(i)].add(value * RW(x[j * incx]));
                            sums[static_cast<std::size_t>(j)].add(Conj::apply(value) * xi);
                        }
                    }
                }

                for (int64_t i = 0; i < n; ++i)
                {
                    const RW sum = sums[static_cast<std::size_t>(i)].value();
                    T& out = y[i * incy];
                    out = keeps_y ? static_cast<T>(RW(alpha) * sum + RW(beta) * RW(out))
                                  : static_cast<T>(RW(alpha) * sum);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
