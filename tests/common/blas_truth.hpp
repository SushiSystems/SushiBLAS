/**************************************************************************/
/* blas_truth.hpp                                                         */
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
 * @file blas_truth.hpp
 * @brief Declares the double-precision truth a BLAS parity test measures both kernels against.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a sum whose rounding errors were recovered, and the magnitude behind it. */
        struct TruthValue
        {
            double value = 0.0;     /**< The sum of the products. */
            double magnitude = 0.0; /**< The sum of the absolute values of the products. */
        };

        /**
         * @brief Sums products of doubles as if in twice the working precision.
         * @see tests/common/README.md
         */
        class TruthSum final
        {
            public:
                /** @brief Adds a * b, keeping the rounding error of the product and of the sum. */
                void add_product(double a, double b) noexcept
                {
                    const double product = a * b;
                    add(product);
                    add(std::fma(a, b, -product));
                    magnitude_ += std::fabs(product);
                }

                /** @brief Returns the sum and the magnitude of every product added so far. */
                [[nodiscard]] TruthValue value() const noexcept
                {
                    return TruthValue{sum_ + error_, magnitude_};
                }

            private:
                /** @brief Adds term, keeping the rounding error of the addition. */
                void add(double term) noexcept
                {
                    const double total = sum_ + term;
                    const double term_part = total - sum_;
                    const double sum_part = total - term_part;
                    error_ += (sum_ - sum_part) + (term - term_part);
                    sum_ = total;
                }

                /// @brief The rounded sum.
                double sum_ = 0.0;

                /// @brief The rounding errors of the additions, summed.
                double error_ = 0.0;

                /// @brief The sum of the absolute values of the products.
                double magnitude_ = 0.0;
        };

        /** @brief Returns the unit roundoff of Real: half the gap between 1 and the next value. */
        template<typename Real>
        [[nodiscard]] constexpr double unit_roundoff() noexcept
        {
            return static_cast<double>(std::numeric_limits<Real>::epsilon()) / 2.0;
        }

        /**
         * @brief Returns the largest error `roundings` rounded operations can leave on a sum of
         *        terms whose absolute values add up to magnitude.
         * @pre roundings * unit is below 1.
         */
        [[nodiscard]] inline double rounding_bound(std::size_t roundings, double unit,
                                                   double magnitude) noexcept
        {
            const double scaled = static_cast<double>(roundings) * unit;
            return scaled / (1.0 - scaled) * magnitude;
        }
    } // namespace Test
} // namespace SushiBLAS
