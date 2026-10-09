/**************************************************************************/
/* gemv.hpp                                                               */
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
 * @file gemv.hpp
 * @brief Defines the reference kernel of GEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes GEMV, y = alpha*op(A)*x + beta*y, each element in a ReferenceSum.
             * @param m Rows of A as stored; op(A) is n x m when trans transposes.
             * @param n Columns of A as stored.
             */
            template<typename T>
            void gemv(Core::Layout layout, Core::Transpose trans, int64_t m, int64_t n,
                      T alpha, const T* a, int64_t lda,
                      const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                using RW = ReferenceAccumulator<T>;

                const OperandAccess access = operand_access(layout, trans, lda);
                const bool transposed = Core::is_transposed(trans);
                const int64_t out_len = transposed ? n : m;
                const int64_t in_len = transposed ? m : n;

                for (int64_t o = 0; o < out_len; ++o)
                {
                    ReferenceSum<T> sum;
                    for (int64_t k = 0; k < in_len; ++k)
                        sum.add(RW(access.at(a, o, k)) * RW(x[k * incx]));

                    // beta==0 must not read the (possibly uninitialized) prior y value.
                    y[o * incy] = (beta == T(0))
                        ? T(RW(alpha) * sum.value())
                        : T(RW(alpha) * sum.value() + RW(beta) * RW(y[o * incy]));
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
