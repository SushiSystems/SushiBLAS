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
 * @brief Defines the reference kernels of the BLAS level 1 routines.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>
#include <type_traits>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            template<typename T> struct RealOf { using type = T; };
            template<typename T> struct RealOf<std::complex<T>> { using type = T; };

            template<typename T>
            inline typename RealOf<T>::type abs1(const T& v) { return static_cast<typename RealOf<T>::type>(std::abs(v)); }

            template<typename T>
            inline T abs1(const std::complex<T>& v) { return std::abs(v.real()) + std::abs(v.imag()); }

            template<typename T>
            void axpy(int64_t n, T alpha, const T* x, int64_t incx, T* y, int64_t incy)
            {
                for (int64_t i = 0; i < n; ++i) y[i * incy] += alpha * x[i * incx];
            }

            template<typename T>
            void copy(int64_t n, const T* x, int64_t incx, T* y, int64_t incy)
            {
                for (int64_t i = 0; i < n; ++i) y[i * incy] = x[i * incx];
            }

            template<typename T>
            void scal(int64_t n, T alpha, T* x, int64_t incx)
            {
                for (int64_t i = 0; i < n; ++i) x[i * incx] *= alpha;
            }

            template<typename T>
            void swap(int64_t n, T* x, int64_t incx, T* y, int64_t incy)
            {
                for (int64_t i = 0; i < n; ++i) std::swap(x[i * incx], y[i * incy]);
            }

            template<typename T>
            void rot(int64_t n, T* x, int64_t incx, T* y, int64_t incy, typename RealOf<T>::type c, typename RealOf<T>::type s)
            {
                for (int64_t i = 0; i < n; ++i)
                {
                    T xi = x[i * incx];
                    T yi = y[i * incy];
                    x[i * incx] = c * xi + s * yi;
                    y[i * incy] = c * yi - s * xi;
                }
            }

            template<typename T>
            void asum(int64_t n, const T* x, int64_t incx, typename RealOf<T>::type* result)
            {
                typename RealOf<T>::type sum(0);
                for (int64_t i = 0; i < n; ++i) sum += abs1(x[i * incx]);
                *result = sum;
            }

            /**
             * @brief Computes the Euclidean norm in the scaling form xNRM2 specifies.
             * @see kernels/README.md
             */
            template<typename T>
            void nrm2(int64_t n, const T* x, int64_t incx, typename RealOf<T>::type* result)
            {
                using R = typename RealOf<T>::type;

                R scale(0), ssq(0);

                auto fold = [&scale, &ssq](R value)
                {
                    const R a = std::abs(value);
                    if (a == R(0)) return;

                    // A non-finite element is folded in unscaled; see kernels/README.md.
                    if (!std::isfinite(a))
                    {
                        if (scale == R(0)) scale = R(1);
                        ssq += a * a;
                        return;
                    }

                    if (a > scale)
                    {
                        const R r = scale / a;
                        ssq = R(1) + ssq * r * r;
                        scale = a;
                    }
                    else
                    {
                        const R r = a / scale;
                        ssq += r * r;
                    }
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    const T v = x[i * incx];
                    if constexpr (std::is_same_v<T, std::complex<R>>)
                    {
                        fold(v.real());
                        fold(v.imag());
                    }
                    else
                    {
                        fold(v);
                    }
                }

                // scale is zero only when nothing was folded, in which case ssq is zero too.
                *result = scale * static_cast<R>(std::sqrt(ssq));
            }

            // Unconjugated dot product: result = sum(x_i * y_i).
            template<typename T>
            void dot(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result)
            {
                T sum(0);
                for (int64_t i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
                *result = sum;
            }

            // Conjugated dot product (DOTC): result = sum(conj(x_i) * y_i).
            template<typename T>
            void dotc(int64_t n, const std::complex<T>* x, int64_t incx, const std::complex<T>* y, int64_t incy, std::complex<T>* result)
            {
                std::complex<T> sum(0);
                for (int64_t i = 0; i < n; ++i) sum += std::conj(x[i * incx]) * y[i * incy];
                *result = sum;
            }

            // Smallest index achieving the maximum of |Re(x_i)| + |Im(x_i)|.
            template<typename T>
            void iamax(int64_t n, const T* x, int64_t incx, int64_t* result)
            {
                int64_t best_idx = 0;
                typename RealOf<T>::type best_val(0);
                for (int64_t i = 0; i < n; ++i)
                {
                    auto v = abs1(x[i * incx]);
                    if (i == 0 || v > best_val)
                    {
                        best_val = v;
                        best_idx = i;
                    }
                }
                *result = best_idx;
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
