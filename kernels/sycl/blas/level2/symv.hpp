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
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
            * @brief SYMV: y = alpha*A*x + beta*y, A symmetric, only the requested
            * triangle physically read. One work-item per output element.
            *
            * Same launch shape and the same reasoning as `gemv` above: `Launch::indexed`
            * for the device-dependent grid, and not `Launch::map` because each output is
            * a reduction over a row rather than a function of one input element.
            */
            template<typename T>
            [[nodiscard]] sycl::event symv(sycl::queue& queue, Core::Layout layout, bool upper,
                             int64_t n, T alpha, const T* a, int64_t lda,
                             const T* x, int64_t incx, T beta, T* y, int64_t incy,
                             const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL SYMV: {}x{}", n, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        auto A_at = [=](int64_t r, int64_t c) -> T
                        {
                            return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                        };
                        auto sym_at = [=](int64_t r, int64_t c) -> T
                        {
                            if (upper) return (r <= c) ? A_at(r, c) : A_at(c, r);
                            return (r >= c) ? A_at(r, c) : A_at(c, r);
                        };
                        T sum(0);
                        for (int64_t j = 0; j < n; ++j) sum += sym_at(i, j) * x[j * incx];
                        y[i * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i * incy];
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
