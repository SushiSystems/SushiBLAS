/****************************************************************/
/* syr2.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file syr2.hpp
 * @brief Defines the reference kernel of SYR2.
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
            * @brief Pure C++ reference SYR2: A += alpha*(x*y^T + y*x^T), only
            * the requested triangle written.
            */
            template<typename T>
            void syr2(Core::Layout layout, bool upper, int64_t n,
                      T alpha, const T* x, int64_t incx,
                      const T* y, int64_t incy,
                      T* a, int64_t lda)
            {
                auto A_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t jstart = upper ? i : 0;
                    int64_t jend = upper ? n : i + 1;
                    for (int64_t j = jstart; j < jend; ++j)
                        A_ref(i, j) += alpha * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
