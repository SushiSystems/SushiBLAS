/**************************************************************************/
/* level1.hpp                                                             */
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
/* permissions and limitations under the License.                        */
/**************************************************************************/

// SYCL Level 1 kernels. The elementwise ops (axpy/copy/scal/swap/rot) are
// embarrassingly parallel over the vector, one work-item per element. The
// reductions (asum/nrm2/dot/iamax) run as a single serial device task -
// Level 1 vectors are small relative to Level 3 workloads in this library,
// so a tree reduction isn't worth the complexity yet (can revisit if a
// profiling need for large-vector Level 1 reductions ever shows up).

#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <kernels/reference/blas/level1/level1.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            template<typename T>
            [[nodiscard]] sycl::event axpy(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx)
                    {
                        int64_t i = idx[0];
                        y[i * incy] += alpha * x[i * incx];
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event copy(sycl::queue& queue, int64_t n, const T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx)
                    {
                        int64_t i = idx[0];
                        y[i * incy] = x[i * incx];
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event scal(sycl::queue& queue, int64_t n, T alpha, T* x, int64_t incx, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx)
                    {
                        int64_t i = idx[0];
                        x[i * incx] *= alpha;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event swap(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx)
                    {
                        int64_t i = idx[0];
                        T tmp = x[i * incx];
                        x[i * incx] = y[i * incy];
                        y[i * incy] = tmp;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event rot(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy,
                             typename Reference::RealOf<T>::type c, typename Reference::RealOf<T>::type s,
                             const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx)
                    {
                        int64_t i = idx[0];
                        T xi = x[i * incx];
                        T yi = y[i * incy];
                        x[i * incx] = c * xi + s * yi;
                        y[i * incy] = c * yi - s * xi;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event asum(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        typename Reference::RealOf<T>::type sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += Reference::abs1(x[i * incx]);
                        *result = sum;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event nrm2(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        typename Reference::RealOf<T>::type sum(0);
                        for (int64_t i = 0; i < n; ++i)
                        {
                            auto v = Reference::abs1(x[i * incx]);
                            sum += v * v;
                        }
                        *result = sycl::sqrt(sum);
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event dot(sycl::queue& queue, int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        T sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
                        *result = sum;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<T>* x, int64_t incx, const std::complex<T>* y, int64_t incy, std::complex<T>* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        std::complex<T> sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += std::conj(x[i * incx]) * y[i * incy];
                        *result = sum;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event iamax(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        int64_t best_idx = 0;
                        typename Reference::RealOf<T>::type best_val(0);
                        for (int64_t i = 0; i < n; ++i)
                        {
                            auto v = Reference::abs1(x[i * incx]);
                            if (i == 0 || v > best_val)
                            {
                                best_val = v;
                                best_idx = i;
                            }
                        }
                        *result = best_idx;
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
