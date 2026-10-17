/**************************************************************************/
/* multiply_add.hpp                                                       */
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
 * @file multiply_add.hpp
 * @brief Declares the multiply-add every product kernel accumulates with, one rounding a step.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Detail
        {
            /** @brief Names the element type of a scalar or of a SYCL vector. */
            template<typename V>
            struct LaneOf
            {
                /** @brief The element type. */
                using type = V;
            };

            /** @brief Names R as the element type of sycl::vec<R, N>. */
            template<typename R, int N>
            struct LaneOf<sycl::vec<R, N>>
            {
                /** @brief The element type. */
                using type = R;
            };

            /**
             * @brief Returns `a * b + c`, rounded once for a floating-point scalar or vector.
             * @see modules/foundation/arithmetic/README.md
             */
            template<typename V>
            [[nodiscard]] inline V multiply_add(const V& a, const V& b, const V& c) noexcept
            {
                if constexpr (std::is_floating_point_v<typename LaneOf<V>::type>)
                    return sycl::fma(a, b, c);
                else
                    return a * b + c;
            }

            /**
             * @brief Adds the product of (a_re, a_im) and (b_re, b_im) to a complex sum held as parts.
             * @see modules/foundation/arithmetic/README.md
             */
            template<typename V>
            inline void multiply_add_parts(V& sum_re, V& sum_im, const V& a_re, const V& a_im,
                                           const V& b_re, const V& b_im) noexcept
            {
                sum_re = multiply_add(V(-a_im), b_im, multiply_add(a_re, b_re, sum_re));
                sum_im = multiply_add(a_im, b_re, multiply_add(a_re, b_im, sum_im));
            }

            /**
             * @brief Returns `a * b + c` for complex values, by the parts of multiply_add_parts.
             * @see modules/foundation/arithmetic/README.md
             */
            template<typename R>
            [[nodiscard]] inline std::complex<R> multiply_add(const std::complex<R>& a,
                                                              const std::complex<R>& b,
                                                              const std::complex<R>& c) noexcept
            {
                R sum_re = c.real();
                R sum_im = c.imag();
                multiply_add_parts<R>(sum_re, sum_im, a.real(), a.imag(), b.real(), b.imag());
                return std::complex<R>(sum_re, sum_im);
            }
        } // namespace Detail
    } // namespace Kernels
} // namespace SushiBLAS
