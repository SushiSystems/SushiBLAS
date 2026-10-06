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
 * @file level1.hpp
 * @brief Defines the SYCL kernels of the BLAS level 1 routines.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include <kernels/reference/blas/level1/level1.hpp>

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
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
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
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
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
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
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
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
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
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
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

            /**
             * @brief Computes the Euclidean norm through Reduce::ScaledSumOfSquares.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event nrm2(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, const std::vector<sycl::event>& deps)
            {
                using R = typename Reference::RealOf<T>::type;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        Reduce::ScaledSumOfSquares<R> acc;
                        for (int64_t i = 0; i < n; ++i)
                        {
                            const T v = x[i * incx];
                            if constexpr (Core::is_complex_v<T>)
                            {
                                acc.accumulate(v.real());
                                acc.accumulate(v.imag());
                            }
                            else
                            {
                                acc.accumulate(v);
                            }
                        }
                        *result = acc.value();
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
