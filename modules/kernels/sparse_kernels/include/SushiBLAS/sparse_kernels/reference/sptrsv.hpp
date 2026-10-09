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
 * @brief Defines the reference kernel of the sparse triangular solve.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/running_solution.hpp>
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
             * @brief Solves op(A) * x = b for a triangular CSR matrix A, overwriting b with x.
             * @param uplo The triangle of A that is read; entries of the other one are skipped.
             * @param diag UNIT skips the stored diagonal and takes it as all ones.
             * @pre pattern is square and b holds pattern.rows adjacent elements.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void sptrsv(Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                        const CsrPattern<Index>& pattern, const T* values, T* b)
            {
                using W = Accumulator<T>;
                using RW = ReferenceAccumulator<T>;

                SB_ASSERT(find_defect(pattern) == StructureDefect::None,
                          "sptrsv was given a CSR structure that breaks its contract");

                const int64_t n = pattern.rows;
                const bool lower = (uplo == Core::Uplo::Lower);
                const bool unit_diag = (diag == Core::Diag::Unit);
                const bool conjugated = Core::is_conjugated(trans);

                std::vector<W> wide(RunningSolution<T>::K_NEEDS_WORKSPACE
                                        ? static_cast<std::size_t>(n) : std::size_t{0});
                const RunningSolution<T> X(b, wide.data(), 0);

                if (!Core::is_transposed(trans))
                {
                    // A lower triangle is solved from row 0 down, an upper one from row n - 1 up.
                    for (int64_t step = 0; step < n; ++step)
                    {
                        const int64_t i = lower ? step : n - 1 - step;

                        ReferenceSum<T> sum;
                        sum.add(RW(b[i]));
                        RW diagonal(0);
                        for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                        {
                            const int64_t j = pattern.column(k);
                            if (j == i)
                                diagonal += RW(values[k]);
                            else if (lower ? j < i : j > i)
                                sum.add(-(RW(values[k]) * RW(X.get(j))));
                        }
                        X.set(i, static_cast<W>(unit_diag ? sum.value() : sum.value() / diagonal));
                    }
                    return;
                }

                // pending[j] holds b[j] less every term the rows solved so far have sent to it.
                std::vector<ReferenceSum<T>> pending(static_cast<std::size_t>(n));
                for (int64_t j = 0; j < n; ++j)
                    pending[static_cast<std::size_t>(j)].add(RW(b[j]));

                // op(A) of a lower triangle is upper, so its rows are solved from n - 1 up.
                for (int64_t step = 0; step < n; ++step)
                {
                    const int64_t i = lower ? n - 1 - step : step;

                    RW diagonal(0);
                    for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                    {
                        if (pattern.column(k) == i)
                            diagonal += RW(conjugate_if(conjugated, values[k]));
                    }

                    const RW total = pending[static_cast<std::size_t>(i)].value();
                    X.set(i, static_cast<W>(unit_diag ? total : total / diagonal));

                    const RW solved = RW(X.get(i));
                    for (int64_t k = pattern.row_begin(i); k < pattern.row_end(i); ++k)
                    {
                        const int64_t j = pattern.column(k);
                        if (lower ? j < i : j > i)
                            pending[static_cast<std::size_t>(j)].add(
                                -(RW(conjugate_if(conjugated, values[k])) * solved));
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
