/**************************************************************************/
/* chunked_dot.hpp                                                        */
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
 * @file chunked_dot.hpp
 * @brief Declares chunked_dot, the one summation order of every GMRES dot product.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/arithmetic/no_contraction.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /** @brief Holds the elements one chunk of a dot product sums, on every device. */
        constexpr std::size_t K_DOT_CHUNK = 256;

        /** @brief Returns the chunks a dot product of @p n elements has. */
        [[nodiscard]] inline std::size_t dot_chunks(std::size_t n) noexcept
        {
            return (n + K_DOT_CHUNK - 1) / K_DOT_CHUNK;
        }

        /**
         * @brief Returns the sum of a[i] * b[i] over chunk @p chunk of @p n, in index order.
         * @note Each product is rounded before it is added.
         */
        template<typename T>
        [[nodiscard]] inline T chunk_dot(const T* a, const T* b, std::size_t n,
                                         std::size_t chunk) noexcept
        {
            SB_FP_CONTRACT_OFF
            const std::size_t begin = chunk * K_DOT_CHUNK;
            const std::size_t end = n < begin + K_DOT_CHUNK ? n : begin + K_DOT_CHUNK;
            T partial = T(0);
            for (std::size_t i = begin; i < end; ++i)
            {
                const T product = a[i] * b[i];
                partial += product;
            }
            return partial;
        }

        /**
         * @brief Returns the dot product of two n-vectors: each chunk in index order, then the
         *        chunk sums in index order.
         * @see modules/kernels/linalg_kernels/README.md
         */
        template<typename T>
        [[nodiscard]] inline T chunked_dot(const T* a, const T* b, std::size_t n) noexcept
        {
            T total = T(0);
            for (std::size_t chunk = 0; chunk < dot_chunks(n); ++chunk)
                total += chunk_dot(a, b, n, chunk);
            return total;
        }
    } // namespace Kernels
} // namespace SushiBLAS
