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
                if (flag == W(-2))
                    return;

                const W h11 = W(param[1]);
                const W h21 = W(param[2]);
                const W h12 = W(param[3]);
                const W h22 = W(param[4]);

                for (int64_t i = 0; i < n; ++i)
                {
                    const W w = W(x[i * incx]);
                    const W z = W(y[i * incy]);

                    if (flag < W(0))
                    {
                        x[i * incx] = T(w * h11 + z * h12);
                        y[i * incy] = T(w * h21 + z * h22);
                    }
                    else if (flag == W(0))
                    {
                        x[i * incx] = T(w + z * h12);
                        y[i * incy] = T(w * h21 + z);
                    }
                    else
                    {
                        x[i * incx] = T(w * h11 + z);
                        y[i * incy] = T(-w + z * h22);
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
