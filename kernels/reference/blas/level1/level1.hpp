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
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Pure C++ reference implementations of BLAS Level 1 (vector-vector) ops.
// These are simple, unblocked, single-threaded kernels - Level 1 is not the
// performance-critical tier of this library (that's Level 3, see gemm.hpp/
// syrk.hpp), so correctness-first serial loops are the right tradeoff here.

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

            // Euclidean norm in the scaling form xNRM2 specifies, not `sqrt(sum(x*x))`.
            //
            // Two bugs were in that one line. The squaring overflowed: an fp32 element past
            // ~1.8e19 squares to infinity, so the norm of a vector whose largest entry is
            // 1e20 came back `inf` where the true answer is about 1e20 - and the mirror
            // case underflowed, an element below ~1e-22 squaring to zero so that a vector
            // of denormals came back exactly 0. Carrying a running `scale` and summing
            // `(x / scale)^2` squares only values near 1, so neither end can happen; the
            // result overflows only when the true norm does.
            //
            // And `abs1` was the wrong magnitude for a complex element. It is BLAS's
            // "|Re| + |Im|", which is what xASUM and IxAMAX are defined on, but the 2-norm
            // needs `Re^2 + Im^2` - using abs1 added a spurious `2|Re||Im|` per element, so
            // a complex nrm2 was simply the wrong number (`sqrt(2)`x too large for x = 1+i).
            // Folding the two components separately is both the correct magnitude and the
            // form that stays overflow-safe, and it is what LAPACK's own classq does.
            //
            // This is the reference kernel, so it is written the textbook way - the running
            // maximum as the scale, no compensation - deliberately independent of the SYCL
            // kernel's octave-rounded, Neumaier-corrected variant rather than sharing code
            // with it. A ground truth that shares an implementation with the thing it
            // checks is not a check.
            template<typename T>
            void nrm2(int64_t n, const T* x, int64_t incx, typename RealOf<T>::type* result)
            {
                using R = typename RealOf<T>::type;

                R scale(0), ssq(0);

                auto fold = [&scale, &ssq](R value)
                {
                    const R a = std::abs(value);
                    if (a == R(0)) return;

                    // An infinity or a NaN has no magnitude to be relative to; folding it
                    // in unscaled saturates the sum, which is the answer BLAS gives.
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
