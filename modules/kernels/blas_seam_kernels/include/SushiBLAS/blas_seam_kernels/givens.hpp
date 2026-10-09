/**************************************************************************/
/* givens.hpp                                                             */
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
/*                                                                        */
/* Transcribed from the reference BLAS of LAPACK, Copyright (c) 1992-2025 */
/* The University of Tennessee and others; modified BSD, see NOTICE.md    */
/**************************************************************************/

/**
 * @file givens.hpp
 * @brief Declares GivensRotation and the function that builds one from two scalars.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <complex>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the plane rotation that maps (a, b) to (r, 0), as xROTG returns it.
         * @tparam T float, double, std::complex<float> or std::complex<double>.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T>
        struct GivensRotation
        {
            T r{};           /**< The first component after the rotation; a receives it. */
            T z{};           /**< The value c and s can be rebuilt from; zero for a complex T. */
            RealType<T> c{}; /**< The cosine, which is real for every T. */
            T s{};           /**< The sine. */
        };

        namespace Detail
        {
            /** @brief Returns 2 raised to exponent, exactly. */
            template<typename R>
            [[nodiscard]] constexpr R two_to(int exponent) noexcept
            {
                R value(1);
                for (int i = 0; i < exponent; ++i)
                    value *= R(2);
                for (int i = 0; i > exponent; --i)
                    value /= R(2);
                return value;
            }

            /**
             * @brief Holds the bounds xROTG scales its operands into for a real type R.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename R>
            struct SafeScaling
            {
                /** @brief Holds the smallest normal value of R. */
                static constexpr R K_MINIMUM = two_to<R>(std::numeric_limits<R>::min_exponent - 1);

                /** @brief Holds the largest power of two of R. */
                static constexpr R K_MAXIMUM = two_to<R>(std::numeric_limits<R>::max_exponent - 1);

                /** @brief Holds the square root of K_MINIMUM, itself a power of two. */
                static constexpr R K_ROOT_MINIMUM =
                    two_to<R>((std::numeric_limits<R>::min_exponent - 1) / 2);
            };

            /** @brief Returns the magnitude of a real value. */
            template<typename R>
            [[nodiscard]] constexpr R magnitude(R value) noexcept
            {
                return (value < R(0)) ? -value : value;
            }

            /** @brief Returns the larger of two real values, the second when they do not order. */
            template<typename R>
            [[nodiscard]] constexpr R larger(R a, R b) noexcept
            {
                return (a > b) ? a : b;
            }

            /** @brief Returns value moved into the interval from K_MINIMUM to K_MAXIMUM. */
            template<typename R>
            [[nodiscard]] constexpr R safe_scale(R value) noexcept
            {
                if (value < SafeScaling<R>::K_MINIMUM)
                    return SafeScaling<R>::K_MINIMUM;
                return (value > SafeScaling<R>::K_MAXIMUM) ? SafeScaling<R>::K_MAXIMUM : value;
            }

            /** @brief Returns the squared magnitude of a complex value. */
            template<typename R>
            [[nodiscard]] constexpr R squared_magnitude(const std::complex<R>& value) noexcept
            {
                return value.real() * value.real() + value.imag() * value.imag();
            }

            /** @brief Returns a complex value divided by a real one, component by component. */
            template<typename R>
            [[nodiscard]] constexpr std::complex<R> over(const std::complex<R>& value,
                                                         R divisor) noexcept
            {
                return std::complex<R>(value.real() / divisor, value.imag() / divisor);
            }
        } // namespace Detail

        /**
         * @brief Returns the rotation of two real scalars by the algorithm of LAPACK 3.10 xROTG.
         * @tparam Root A type whose static apply(R) returns the square root of its argument.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename Root, typename R>
        [[nodiscard]] GivensRotation<R> givens_rotation(R a, R b) noexcept
        {
            const R a_norm = Detail::magnitude(a);
            const R b_norm = Detail::magnitude(b);

            GivensRotation<R> g;
            if (b_norm == R(0))
            {
                g.r = a;
                g.z = R(0);
                g.c = R(1);
                g.s = R(0);
                return g;
            }
            if (a_norm == R(0))
            {
                g.r = b;
                g.z = R(1);
                g.c = R(0);
                g.s = R(1);
                return g;
            }

            const R scale = Detail::safe_scale(Detail::larger(a_norm, b_norm));
            const R sign_source = (a_norm > b_norm) ? a : b;
            const R sigma = (sign_source < R(0)) ? R(-1) : R(1);
            const R a_scaled = a / scale;
            const R b_scaled = b / scale;

            g.r = sigma * (scale * Root::apply(a_scaled * a_scaled + b_scaled * b_scaled));
            g.c = a / g.r;
            g.s = b / g.r;
            if (a_norm > b_norm)
                g.z = g.s;
            else
                g.z = (g.c != R(0)) ? R(1) / g.c : R(1);
            return g;
        }

        namespace Detail
        {
            /**
             * @brief Returns the rotation of a zero f and a non-zero g.
             * @pre g is not zero.
             */
            template<typename Root, typename R>
            [[nodiscard]] GivensRotation<std::complex<R>>
            givens_of_zero_first(const std::complex<R>& g) noexcept
            {
                GivensRotation<std::complex<R>> out;
                out.c = R(0);

                if (g.real() == R(0) || g.imag() == R(0))
                {
                    const R d = magnitude(g.real() == R(0) ? g.imag() : g.real());
                    out.r = std::complex<R>(d, R(0));
                    out.s = over(conjugate(g), d);
                    return out;
                }

                const R g1 = larger(magnitude(g.real()), magnitude(g.imag()));
                const R root_maximum = Root::apply(SafeScaling<R>::K_MAXIMUM / R(2));
                if (g1 > SafeScaling<R>::K_ROOT_MINIMUM && g1 < root_maximum)
                {
                    const R d = Root::apply(squared_magnitude(g));
                    out.r = std::complex<R>(d, R(0));
                    out.s = over(conjugate(g), d);
                    return out;
                }

                const R u = safe_scale(g1);
                const std::complex<R> gs = over(g, u);
                const R d = Root::apply(squared_magnitude(gs));
                out.r = std::complex<R>(d * u, R(0));
                out.s = over(conjugate(gs), d);
                return out;
            }

            /**
             * @brief Returns the rotation of f and g once both are scaled so their squares hold.
             * @param f2 The squared magnitude of f, with any extra scale of f already in h2.
             * @param h2 The squared magnitude of the scaled pair (f, g).
             * @param root_maximum The bound the branch that called this tests f and g against.
             * @pre Neither f nor g is zero.
             */
            template<typename Root, typename R>
            [[nodiscard]] GivensRotation<std::complex<R>>
            givens_of_scaled(const std::complex<R>& f, const std::complex<R>& g, R f2, R h2,
                             R root_maximum) noexcept
            {
                GivensRotation<std::complex<R>> out;

                if (f2 >= h2 * SafeScaling<R>::K_MINIMUM)
                {
                    out.c = Root::apply(f2 / h2);
                    out.r = over(f, out.c);
                    if (f2 > SafeScaling<R>::K_ROOT_MINIMUM && h2 < root_maximum * R(2))
                        out.s = multiply(conjugate(g), over(f, Root::apply(f2 * h2)));
                    else
                        out.s = multiply(conjugate(g), over(out.r, h2));
                    return out;
                }

                const R d = Root::apply(f2 * h2);
                out.c = f2 / d;
                out.r = (out.c >= SafeScaling<R>::K_MINIMUM) ? over(f, out.c)
                                                             : multiply(h2 / d, f);
                out.s = multiply(conjugate(g), over(f, d));
                return out;
            }
        } // namespace Detail

        /**
         * @brief Returns the rotation of two complex scalars by the algorithm of LAPACK 3.10
         *        xROTG.
         * @tparam Root A type whose static apply(R) returns the square root of its argument.
         * @return A rotation whose z is zero: the complex routine leaves b as it is.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename Root, typename R>
        [[nodiscard]] GivensRotation<std::complex<R>>
        givens_rotation(const std::complex<R>& f, const std::complex<R>& g) noexcept
        {
            const bool f_is_zero = (f.real() == R(0) && f.imag() == R(0));
            const bool g_is_zero = (g.real() == R(0) && g.imag() == R(0));

            if (g_is_zero)
            {
                GivensRotation<std::complex<R>> out;
                out.r = f;
                out.c = R(1);
                return out;
            }
            if (f_is_zero)
                return Detail::givens_of_zero_first<Root>(g);

            const R f1 = Detail::larger(Detail::magnitude(f.real()), Detail::magnitude(f.imag()));
            const R g1 = Detail::larger(Detail::magnitude(g.real()), Detail::magnitude(g.imag()));
            const R root_minimum = Detail::SafeScaling<R>::K_ROOT_MINIMUM;
            const R root_maximum = Root::apply(Detail::SafeScaling<R>::K_MAXIMUM / R(4));

            if (f1 > root_minimum && f1 < root_maximum && g1 > root_minimum && g1 < root_maximum)
            {
                const R f2 = Detail::squared_magnitude(f);
                const R h2 = f2 + Detail::squared_magnitude(g);
                return Detail::givens_of_scaled<Root>(f, g, f2, h2, root_maximum);
            }

            const R u = Detail::safe_scale(Detail::larger(f1, g1));
            const std::complex<R> gs = Detail::over(g, u);
            const R g2 = Detail::squared_magnitude(gs);

            // f is scaled by v when it is too small to share u with g, and w carries v / u.
            const bool separate = (f1 / u < root_minimum);
            const R v = separate ? Detail::safe_scale(f1) : u;
            const R w = separate ? v / u : R(1);
            const std::complex<R> fs = Detail::over(f, v);
            const R f2 = Detail::squared_magnitude(fs);
            const R h2 = separate ? f2 * (w * w) + g2 : f2 + g2;

            GivensRotation<std::complex<R>> out =
                Detail::givens_of_scaled<Root>(fs, gs, f2, h2, root_maximum);
            out.c = out.c * w;
            out.r = multiply(u, out.r);
            return out;
        }
    } // namespace Kernels
} // namespace SushiBLAS
