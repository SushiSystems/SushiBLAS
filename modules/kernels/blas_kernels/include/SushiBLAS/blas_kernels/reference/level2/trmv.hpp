/**************************************************************************/
/* trmv.hpp                                                               */
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
 * @file trmv.hpp
 * @brief Defines the reference kernel of TRMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes TRMV, x = op(A)*x, in place without a copy of x.
             * @see kernels/README.md
             */
            template<typename T>
            void trmv(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                      int64_t n, const T* a, int64_t lda, T* x, int64_t incx)
            {
                using RW = ReferenceAccumulator<T>;

                const OperandAccess access = operand_access(layout, trans, lda);
                const bool unit_diag = (diag == Core::Diag::Unit);
                const bool eff_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::Upper);

                if (eff_upper)
                {
                    for (int64_t i = 0; i < n; ++i)
                    {
                        ReferenceSum<T> sum;
                        for (int64_t k = i; k < n; ++k)
                        {
                            T aval = (k == i && unit_diag) ? T(1) : access.at(a, i, k);
                            sum.add(RW(aval) * RW(x[k * incx]));
                        }
                        x[i * incx] = T(sum.value());
                    }
                }
                else
                {
                    for (int64_t i = n - 1; i >= 0; --i)
                    {
                        ReferenceSum<T> sum;
                        for (int64_t k = 0; k <= i; ++k)
                        {
                            T aval = (k == i && unit_diag) ? T(1) : access.at(a, i, k);
                            sum.add(RW(aval) * RW(x[k * incx]));
                        }
                        x[i * incx] = T(sum.value());
                    }
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
