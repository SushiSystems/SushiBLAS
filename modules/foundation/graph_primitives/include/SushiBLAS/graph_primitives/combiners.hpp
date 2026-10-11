/**************************************************************************/
/* combiners.hpp                                                          */
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
 * @file combiners.hpp
 * @brief Declares the Sum, Product, Minimum and Maximum combiners a reduction folds with.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Adds two values; a sum of no elements is the value-initialised T.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T>
        struct Sum
        {
            /**
             * @brief Returns the sum of two values.
             * @param a The accumulator of a fold.
             * @param b The next element.
             */
            [[nodiscard]] T operator()(const T& a, const T& b) const
            {
                return a + b;
            }

            /** @brief Returns the value a sum of no elements produces. */
            [[nodiscard]] static constexpr T identity()
            {
                return T{};
            }
        };

        /**
         * @brief Multiplies two values; a product of no elements is one.
         * @tparam T Element type; must be constructible from the integer one.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T>
        struct Product
        {
            /**
             * @brief Returns the product of two values.
             * @param a The accumulator of a fold.
             * @param b The next element.
             */
            [[nodiscard]] T operator()(const T& a, const T& b) const
            {
                return a * b;
            }

            /** @brief Returns the value a product of no elements produces. */
            [[nodiscard]] static constexpr T identity()
            {
                return static_cast<T>(1);
            }
        };

        /**
         * @brief Keeps the smaller of two values; a minimum of no elements is the largest T.
         * @tparam T Element type; must have a std::numeric_limits.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T>
        struct Minimum
        {
            /**
             * @brief Returns b when it compares less than a, and a otherwise.
             * @param a The accumulator of a fold.
             * @param b The next element.
             */
            [[nodiscard]] T operator()(const T& a, const T& b) const
            {
                return b < a ? b : a;
            }

            /** @brief Returns the value a minimum of no elements produces. */
            [[nodiscard]] static constexpr T identity()
            {
                return (std::numeric_limits<T>::max)();
            }
        };

        /**
         * @brief Keeps the larger of two values; a maximum of no elements is the lowest T.
         * @tparam T Element type; must have a std::numeric_limits.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T>
        struct Maximum
        {
            /**
             * @brief Returns b when a compares less than it, and a otherwise.
             * @param a The accumulator of a fold.
             * @param b The next element.
             */
            [[nodiscard]] T operator()(const T& a, const T& b) const
            {
                return a < b ? b : a;
            }

            /** @brief Returns the value a maximum of no elements produces. */
            [[nodiscard]] static constexpr T identity()
            {
                return (std::numeric_limits<T>::lowest)();
            }
        };

    } // namespace GraphPrimitives
} // namespace SushiBLAS
