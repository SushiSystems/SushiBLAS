/**************************************************************************/
/* syr2.hpp                                                               */
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
