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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/support/accumulator.hpp>

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
                        if (flag == W(-2))
                            return;

                        const int64_t i = static_cast<int64_t>(item);
                        const W h11 = W(param[1]);
                        const W h21 = W(param[2]);
                        const W h12 = W(param[3]);
                        const W h22 = W(param[4]);
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
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
