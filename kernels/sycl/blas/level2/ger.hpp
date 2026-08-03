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
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
            * @brief GER (unconjugated rank-1 update): A += alpha * x * y^T,
            * one work-item per output element (elementwise on A, no hazard).
            *
            * Left on a `sycl::range<2>` on purpose, unlike `gemv`/`symv`.
            * <SushiBLAS/engine/launch_1d.hpp> plans one-dimensional geometries; this
            * kernel's index *is* two-dimensional, because `A` is addressed by `(i, j)`
            * under a runtime layout and the two axes are strided differently. Flattening
            * it to `i * n + j` to force it through `Launch::indexed` would trade a
            * geometry the backend can pick a 2-D work-group for against a division and a
            * modulo per work-item, and would buy the grid-stride shape for a kernel whose
            * launch is already sized by the output matrix rather than by a vector length.
            * If this ever needs the device split, the honest fix is a `plan_2d` next to
            * `plan_1d`, not a reshape here.
            */
            template<typename T>
            [[nodiscard]] sycl::event ger(sycl::queue& queue, Core::Layout layout,
                            int64_t m, int64_t n, T alpha,
                            const T* x, int64_t incx, const T* y, int64_t incy,
                            T* a, int64_t lda, const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL GER: {}x{}", m, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(n)), [=](sycl::id<2> idx)
                    {
                        int64_t i = idx[0], j = idx[1];
                        T& ref = (layout == Core::Layout::ROW_MAJOR) ? a[i * lda + j] : a[i + j * lda];
                        ref += alpha * x[i * incx] * y[j * incy];
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
