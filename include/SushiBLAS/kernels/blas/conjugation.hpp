/**************************************************************************/
/* conjugation.hpp                                                        */
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
 * @file conjugation.hpp
 * @brief Declares the conjugation policies that turn a symmetric kernel into its Hermitian sibling.
 * @author Mustafa Garip
 */

#pragma once

#include <type_traits>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Leaves every element as it is: the policy of an unconjugated or symmetric routine.
         * @see include/SushiBLAS/kernels/README.md
         */
        struct Unconjugated
        {
            /** @brief Returns the second factor of a product, unchanged. */
            template<typename T>
            [[nodiscard]] static constexpr T apply(T value) noexcept
            {
                return value;
            }

            /** @brief Returns a diagonal element, unchanged. */
            template<typename T>
            [[nodiscard]] static constexpr T diagonal(T value) noexcept
            {
                return value;
            }
        };

        /**
         * @brief Conjugates: the policy of dotc, gerc, hemv, her, her2 and herk.
         * @see include/SushiBLAS/kernels/README.md
         */
        struct Conjugated
        {
            /** @brief Returns the conjugate of the factor a Hermitian product conjugates. */
            template<typename T>
            [[nodiscard]] static constexpr T apply(T value) noexcept
            {
                return conjugate(value);
            }

            /** @brief Returns a diagonal element of a Hermitian matrix: its real part. */
            template<typename T>
            [[nodiscard]] static constexpr T diagonal(T value) noexcept
            {
                return real_part(value);
            }
        };

        /**
         * @brief Names the policy a kernel is instantiated with for element type T.
         * @return Conj for a complex T, Unconjugated for a real one.
         */
        template<typename T, typename Conj>
        using ConjugationFor = std::conditional_t<Core::is_complex_v<T>, Conj, Unconjugated>;

        /**
         * @brief Names the type of alpha and beta in a rank-1 or rank-k update under a policy.
         * @return The real component type under Conjugated (her, herk), T otherwise (syr, syrk).
         */
        template<typename T, typename Conj>
        using RankUpdateScalar =
            std::conditional_t<std::is_same_v<Conj, Conjugated>, RealType<T>, T>;
    } // namespace Kernels
} // namespace SushiBLAS
