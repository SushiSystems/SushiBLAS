/****************************************************************/
/* gemv.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file gemv.hpp
 * @brief Defines the reference kernel of GEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference GEMV: y = alpha*op(A)*x + beta*y.
            * A is m x n physically; op(A) is n x m if transA else m x n.
            */
            template<typename T>
            void gemv(Core::Layout layout, bool transA, int64_t m, int64_t n,
                      T alpha, const T* a, int64_t lda,
                      const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };

                int64_t out_len = transA ? n : m;
                int64_t in_len = transA ? m : n;

                for (int64_t o = 0; o < out_len; ++o)
                {
                    T sum(0);
                    for (int64_t k = 0; k < in_len; ++k)
                    {
                        T aval = transA ? A_at(k, o) : A_at(o, k);
                        sum += aval * x[k * incx];
                    }
                    // beta==0 must not read the (possibly uninitialized) prior y value.
                    y[o * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[o * incy];
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
