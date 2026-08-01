/**************************************************************************/
/* syrk.hpp                                                               */
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

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference implementation of SYRK.
            * 
            * Supports Row-Major and Column-Major layouts.
            */
            template<typename T>
            void syrk(Core::Layout layout, bool upper, bool trans,
                      int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda,
                      T beta, T* c, int64_t ldc) 
            {
                for (int64_t i = 0; i < n; ++i) 
                {
                    for (int64_t j = 0; j < n; ++j) 
                    {
                        if ((upper && j < i) || (!upper && j > i)) 
                            continue;

                        T sum(0);
                        for (int64_t l = 0; l < k; ++l) 
                        {
                            T valA_1, valA_2;
                            if (layout == Core::Layout::ROW_MAJOR) 
                            {
                                valA_1 = trans ? a[l * lda + i] : a[i * lda + l];
                                valA_2 = trans ? a[l * lda + j] : a[j * lda + l];
                            } else 
                            {
                                valA_1 = trans ? a[i + l * lda] : a[l + i * lda];
                                valA_2 = trans ? a[j + l * lda] : a[l + j * lda];
                            }
                            sum += valA_1 * valA_2;
                        }

                        if (layout == Core::Layout::ROW_MAJOR) 
                            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
                        else 
                            c[i + j * ldc] = alpha * sum + beta * c[i + j * ldc];
                    }
                }
            }

            /**
            * @brief Batch SYRK reference implementation.
            */
            template<typename T>
            void syrk(Core::Layout layout, bool upper, bool trans,
                      int64_t n, int64_t k,
                      T alpha, const T* a, int64_t lda, int64_t str_a,
                      T beta, T* c, int64_t ldc, int64_t str_c,
                      int64_t batch_size) 
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) 
                {
                    syrk(layout, upper, trans, n, k, alpha, 
                         a + b_idx * str_a, lda, 
                         beta, c + b_idx * str_c, ldc);
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
