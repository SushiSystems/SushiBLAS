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
 * @file gemv.hpp
 * @brief Defines the reference kernel of GEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/matrix_vector_index.hpp>

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
                const Indexing::VectorExtents extents = Indexing::vector_extents(transposed, m, n);
                const int64_t out_len = extents.out_len;
                const int64_t in_len = extents.in_len;

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
