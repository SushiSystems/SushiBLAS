/**************************************************************************/
/* rotm.hpp                                                               */
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
 * @file rotm.hpp
 * @brief Defines the reference kernel of ROTM.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_kernels/rotm_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes ROTM: applies the modified Givens transformation H to every pair
             *        (x_i, y_i).
             * @param param Five elements: the flag, then h11, h21, h12 and h22.
             * @see kernels/README.md
             */
            template<typename T>
            void rotm(int64_t n, T* x, int64_t incx, T* y, int64_t incy, const T* param)
            {
                using W = Accumulator<T>;

                const W flag = W(param[0]);
                if (Indexing::rotm_is_identity(flag))
                    return;

                const W h11 = W(param[1]);
                const W h21 = W(param[2]);
                const W h12 = W(param[3]);
                const W h22 = W(param[4]);

                for (int64_t i = 0; i < n; ++i)
                {
                    const Indexing::RotmPair<W> rotated = Indexing::rotm_apply(
                        flag, h11, h21, h12, h22, W(x[i * incx]), W(y[i * incy]));
                    x[i * incx] = T(rotated.x);
                    y[i * incy] = T(rotated.y);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
