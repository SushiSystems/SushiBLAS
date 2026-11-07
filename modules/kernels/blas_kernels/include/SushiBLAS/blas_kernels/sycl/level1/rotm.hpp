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
 * @brief Defines the SYCL kernel of ROTM.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_kernels/rotm_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes ROTM: applies the modified Givens transformation H to every pair
             *        (x_i, y_i), one work-item per pair.
             * @param param Five elements on the device: the flag, then h11, h21, h12 and h22.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event rotm(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y,
                                           int64_t incy, const T* param,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(n), [=](std::size_t item)
                    {
                        const W flag = W(param[0]);
                        if (Indexing::rotm_is_identity(flag))
                            return;

                        const int64_t i = static_cast<int64_t>(item);
                        const Indexing::RotmPair<W> rotated = Indexing::rotm_apply(
                            flag, W(param[1]), W(param[2]), W(param[3]), W(param[4]),
                            W(x[i * incx]), W(y[i * incy]));
                        x[i * incx] = T(rotated.x);
                        y[i * incy] = T(rotated.y);
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
