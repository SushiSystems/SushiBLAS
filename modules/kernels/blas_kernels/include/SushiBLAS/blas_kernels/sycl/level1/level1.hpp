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
#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>

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
                        y[i * incy] += multiply(alpha, x[i * incx]);
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
                        x[i * incx] = multiply(x[i * incx], alpha);
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
                        x[i * incx] = multiply(c, xi) + multiply(s, yi);
                        y[i * incy] = multiply(c, yi) - multiply(s, xi);
                    });
                });
            }

            /**
             * @brief Holds the bytes of scratch per work-group partial that every Level 1 reduction fits in.
             * @see kernels/README.md
             */
            inline constexpr std::size_t K_LEVEL1_PARTIAL_BYTES = 16;

            /**
             * @brief Returns how many W partials fit in @p scratch_bytes.
             * @tparam W The partial type; must fit in K_LEVEL1_PARTIAL_BYTES.
             */
            template<typename W>
            [[nodiscard]] constexpr std::size_t partial_capacity(std::size_t scratch_bytes) noexcept
            {
                static_assert(sizeof(W) <= K_LEVEL1_PARTIAL_BYTES,
                              "A Level 1 partial outgrew K_LEVEL1_PARTIAL_BYTES.");
                return scratch_bytes / sizeof(W);
            }

            /**
             * @brief Computes the sum of magnitudes through Reduce::CompensatedSum.
             * @param scratch Device scratch for the partials; may be null when scratch_bytes is zero.
             * @note The result may differ in the last bits from a serial left-to-right sum.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event asum(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, void* scratch, std::size_t scratch_bytes, const std::vector<sycl::event>& deps)
            {
                using R = typename Reference::RealOf<T>::type;
                using W = Reduce::AccumulationOf<R>;

                return Reduce::reduce_indexed<Reduce::CompensatedSum>(queue, static_cast<std::size_t>(n),
                    [=](std::size_t i) { return static_cast<W>(Reference::abs1(x[static_cast<int64_t>(i) * incx])); },
                    result, static_cast<W*>(scratch), partial_capacity<W>(scratch_bytes),
                    [](W v, std::size_t) { return v; }, deps);
            }

            /**
             * @brief Computes the Euclidean norm through Reduce::ScaledSumOfSquares.
             * @param scratch Device scratch for the partials; may be null when scratch_bytes is zero.
             * @note On the two-stage shape the result may differ in the last bits from a serial fold.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event nrm2(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, void* scratch, std::size_t scratch_bytes, const std::vector<sycl::event>& deps)
            {
                using R = typename Reference::RealOf<T>::type;
                using W = Reduce::AccumulationOf<R>;

                // A complex element contributes its real part, then its imaginary part.
                constexpr std::size_t K_PARTS = Core::is_complex_v<T> ? 2 : 1;

                return Reduce::reduce_indexed<Reduce::ScaledSumOfSquares>(queue, static_cast<std::size_t>(n) * K_PARTS,
                    [=](std::size_t k) -> W
                    {
                        const T v = x[static_cast<int64_t>(k / K_PARTS) * incx];
                        if constexpr (Core::is_complex_v<T>)
                            return static_cast<W>((k % 2 == 0) ? v.real() : v.imag());
                        else
                            return static_cast<W>(v);
                    },
                    result, static_cast<W*>(scratch), partial_capacity<W>(scratch_bytes),
                    [](W v, std::size_t) { return v; }, deps);
            }

            /**
             * @brief Computes sum(Conj(x_i) * y_i) through a compensated sum.
             * @tparam Conj Unconjugated for dot and dotu, Conjugated for dotc.
             * @param scratch Device scratch for the partials; may be null when scratch_bytes is 0.
             * @note The result may differ in the last bits from a serial left-to-right sum.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event dot(sycl::queue& queue, int64_t n, const T* x, int64_t incx,
                                          const T* y, int64_t incy, T* result, void* scratch,
                                          std::size_t scratch_bytes,
                                          const std::vector<sycl::event>& deps)
            {
                using W = Reduce::AccumulationOf<T>;

                const auto term = [=](std::size_t k)
                {
                    const int64_t i = static_cast<int64_t>(k);
                    return multiply(Conj::apply(static_cast<W>(x[i * incx])),
                                    static_cast<W>(y[i * incy]));
                };
                const auto finish = [](W v, std::size_t) { return v; };

                if constexpr (Core::is_complex_v<T>)
                    return Reduce::reduce_indexed<Reduce::CompensatedComplexSum>(
                        queue, static_cast<std::size_t>(n), term, result, static_cast<W*>(scratch),
                        partial_capacity<W>(scratch_bytes), finish, deps);
                else
                    return Reduce::reduce_indexed<Reduce::CompensatedSum>(
                        queue, static_cast<std::size_t>(n), term, result, static_cast<W*>(scratch),
                        partial_capacity<W>(scratch_bytes), finish, deps);
            }

            /**
             * @brief Finds the index of the largest magnitude through Reduce::RunningArgMax.
             * @param scratch Device scratch for the partials; may be null when scratch_bytes is zero.
             * @note Ties go to the lowest index; a NaN wins only at index 0, as in the reference kernel.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event iamax(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t* result, void* scratch, std::size_t scratch_bytes, const std::vector<sycl::event>& deps)
            {
                using R = typename Reference::RealOf<T>::type;
                using P = Reduce::IndexedValue<R>;

                return Reduce::reduce_indexed<Reduce::RunningArgMax>(queue, static_cast<std::size_t>(n),
                    [=](std::size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        return P{static_cast<R>(Reference::abs1(x[i * incx])), i};
                    },
                    result, static_cast<P*>(scratch), partial_capacity<P>(scratch_bytes),
                    // An empty input answers index 0, as the reference kernel does.
                    [](P v, std::size_t) { return v.index < 0 ? int64_t{0} : v.index; }, deps);
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
