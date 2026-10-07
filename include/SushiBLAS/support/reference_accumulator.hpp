/**************************************************************************/
/* reference_accumulator.hpp                                              */
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
 * @file reference_accumulator.hpp
 * @brief Declares the type and the running sum a reference kernel accumulates an element type in.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <complex>
#include <type_traits>
#include <SushiBLAS/core/traits.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names the type a reference kernel sums a real T in: double.
         * @tparam T A floating-point element type; an integer has no reference sum.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename T>
        struct ReferenceAccumulatorOf
        {
            static_assert(!std::is_integral_v<T>,
                          "A reference kernel sums HALF, FLOAT32, FLOAT64 and the complex types.");

            /** @brief The accumulation type. */
            using type = double;
        };

        /** @brief Sums a complex element in std::complex<double>. */
        template<typename R>
        struct ReferenceAccumulatorOf<std::complex<R>>
        {
            /** @brief The accumulation type. */
            using type = std::complex<double>;
        };

        /** @brief Shorthand for `ReferenceAccumulatorOf<T>::type`. */
        template<typename T>
        using ReferenceAccumulator = typename ReferenceAccumulatorOf<T>::type;

        namespace Detail
        {
            /**
             * @brief Holds a running sum of doubles, and under Compensated the rounding error of
             *        every addition.
             * @see include/SushiBLAS/support/README.md
             */
            template<bool Compensated>
            class RunningDouble final
            {
                public:
                    /** @brief Adds term to the sum. */
                    void add(double term) noexcept
                    {
                        const double total = sum_ + term;
                        if constexpr (Compensated)
                        {
                            // TwoSum: both differences are exact only under strict IEEE arithmetic.
                            const double term_part = total - sum_;
                            const double sum_part = total - term_part;
                            error_ += (sum_ - sum_part) + (term - term_part);
                        }
                        sum_ = total;
                    }

                    /** @brief Returns the sum, with the collected error added when it is finite. */
                    [[nodiscard]] double value() const noexcept
                    {
                        if constexpr (Compensated)
                            return std::isfinite(error_) ? sum_ + error_ : sum_;
                        else
                            return sum_;
                    }

                private:
                    /// @brief The rounded sum of every term added so far.
                    double sum_ = 0.0;

                    /// @brief The summed rounding errors of the additions; zero unless Compensated.
                    double error_ = 0.0;
            };
        } // namespace Detail

        /**
         * @brief Holds the running sum of a reference kernel: in double for HALF, FLOAT32 and
         *        COMPLEX32, compensated for FLOAT64 and COMPLEX64.
         * @tparam T The element type of the tensor the kernel serves.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename T>
        class ReferenceSum final
        {
            public:
                /** @brief The type a term is added in and the sum is read in. */
                using Wide = ReferenceAccumulator<T>;

                /** @brief Holds whether Wide is T itself, so the sum keeps its rounding errors. */
                static constexpr bool K_IS_COMPENSATED = std::is_same_v<Wide, T>;

                /** @brief Adds term to the sum. */
                void add(const Wide& term) noexcept
                {
                    if constexpr (Core::is_complex_v<Wide>)
                    {
                        real_.add(term.real());
                        imaginary_.add(term.imag());
                    }
                    else
                    {
                        real_.add(term);
                    }
                }

                /** @brief Returns the sum of every term added so far. */
                [[nodiscard]] Wide value() const noexcept
                {
                    if constexpr (Core::is_complex_v<Wide>)
                        return Wide(real_.value(), imaginary_.value());
                    else
                        return real_.value();
                }

            private:
                /// @brief The sum of the real parts.
                Detail::RunningDouble<K_IS_COMPENSATED> real_;

                /// @brief The sum of the imaginary parts; stays zero for a real Wide.
                Detail::RunningDouble<K_IS_COMPENSATED> imaginary_;
        };
    } // namespace Kernels
} // namespace SushiBLAS
