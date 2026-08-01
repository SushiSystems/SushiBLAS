/**************************************************************************/
/* trmv.hpp                                                               */
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
            * @brief TRMV: x = op(A)*x, A triangular, in-place, no temp buffer.
            *
            * x[i] depends only on x[k] for k on one side of i (determined by
            * op(A)'s effective triangle: "eff_upper" means the support is
            * k>=i). Walking i from the "open" end towards the "closed" end
            * (increasing for eff_upper, decreasing otherwise) guarantees every
            * x[k] a step needs is still pristine when that step runs, since
            * indices already overwritten are always outside the remaining
            * steps' dependency range. This must run as a single sequential
            * task - the write/read dependency between steps isn't parallel.
            */
            template<typename T>
            [[nodiscard]] sycl::event trmv(sycl::queue& queue, Core::Layout layout, bool upper, bool transA, bool unit_diag,
                             int64_t n, const T* a, int64_t lda, T* x, int64_t incx,
                             const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL TRMV: {}x{}", n, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        auto A_at = [=](int64_t r, int64_t c) -> T
                        {
                            return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                        };
                        auto A_eff = [=](int64_t i, int64_t k) -> T
                        {
                            return transA ? A_at(k, i) : A_at(i, k);
                        };
                        bool eff_upper = transA ? !upper : upper;

                        if (eff_upper)
                        {
                            for (int64_t i = 0; i < n; ++i)
                            {
                                T sum(0);
                                for (int64_t k = i; k < n; ++k)
                                {
                                    T aval = (k == i && unit_diag) ? T(1) : A_eff(i, k);
                                    sum += aval * x[k * incx];
                                }
                                x[i * incx] = sum;
                            }
                        }
                        else
                        {
                            for (int64_t i = n - 1; i >= 0; --i)
                            {
                                T sum(0);
                                for (int64_t k = 0; k <= i; ++k)
                                {
                                    T aval = (k == i && unit_diag) ? T(1) : A_eff(i, k);
                                    sum += aval * x[k * incx];
                                }
                                x[i * incx] = sum;
                            }
                        }
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
