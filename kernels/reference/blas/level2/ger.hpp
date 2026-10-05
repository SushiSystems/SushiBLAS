/****************************************************************/
/* ger.hpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
            * @brief Pure C++ reference GER (unconjugated rank-1 update): A += alpha * x * y^T.
            */
            template<typename T>
            void ger(Core::Layout layout, int64_t m, int64_t n,
                     T alpha, const T* x, int64_t incx,
                     const T* y, int64_t incy,
                     T* a, int64_t lda)
            {
                auto A_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };

                for (int64_t i = 0; i < m; ++i)
                    for (int64_t j = 0; j < n; ++j)
                        A_ref(i, j) += alpha * x[i * incx] * y[j * incy];
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
