/**************************************************************************/
/* ger.hpp                                                                */
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
