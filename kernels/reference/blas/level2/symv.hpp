/**************************************************************************/
/* symv.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
            * @brief Pure C++ reference SYMV: y = alpha*A*x + beta*y, A symmetric
            * with only the requested triangle physically read.
            */
            template<typename T>
            void symv(Core::Layout layout, bool upper, int64_t n,
                      T alpha, const T* a, int64_t lda,
                      const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                auto sym_at = [&](int64_t i, int64_t j) -> T
                {
                    if (upper) return (i <= j) ? A_at(i, j) : A_at(j, i);
                    return (i >= j) ? A_at(i, j) : A_at(j, i);
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    T sum(0);
                    for (int64_t j = 0; j < n; ++j) sum += sym_at(i, j) * x[j * incx];
                    y[i * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i * incy];
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
