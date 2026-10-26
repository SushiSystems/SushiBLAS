/**************************************************************************/
/* lu.hpp                                                                 */
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
 * @file lu.hpp
 * @brief Defines the reference kernels of the batched dense LU factorisation and its solve.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <SushiBLAS/arithmetic/no_contraction.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /** @brief Swaps rows @p first and @p second of a row-major matrix @p width wide. */
                template<typename T>
                void swap_rows(T* a, std::size_t width, std::size_t first,
                               std::size_t second) noexcept
                {
                    if (first == second)
                        return;
                    std::swap_ranges(a + first * width, a + first * width + width,
                                     a + second * width);
                }

                /**
                 * @brief Factors one row-major n * n matrix in place; returns 1 on a small pivot.
                 * @param pivots n elements; pivots[i] is the row swapped with row i at step i.
                 */
                template<typename T>
                std::uint32_t lu_factor_one(T* a, std::uint32_t* pivots, std::size_t n) noexcept
                {
                    SB_FP_CONTRACT_OFF
                    T largest = T(0);
                    for (std::size_t i = 0; i < n * n; ++i)
                        largest = std::max(largest, static_cast<T>(std::abs(a[i])));
                    const T threshold =
                        static_cast<T>(n) * std::numeric_limits<T>::epsilon() * largest;

                    for (std::size_t k = 0; k < n; ++k)
                    {
                        // Strictly greater keeps the lowest row index on a tie.
                        std::size_t pivot_row = k;
                        T pivot_magnitude = std::abs(a[k * n + k]);
                        for (std::size_t i = k + 1; i < n; ++i)
                            if (std::abs(a[i * n + k]) > pivot_magnitude)
                            {
                                pivot_row = i;
                                pivot_magnitude = std::abs(a[i * n + k]);
                            }

                        if (!(pivot_magnitude > threshold))
                        {
                            for (std::size_t i = k; i < n; ++i)
                                pivots[i] = static_cast<std::uint32_t>(i);
                            return 1U;
                        }

                        pivots[k] = static_cast<std::uint32_t>(pivot_row);
                        swap_rows(a, n, k, pivot_row);
                        const T inverse = T(1) / a[k * n + k];
                        for (std::size_t i = k + 1; i < n; ++i)
                            a[i * n + k] *= inverse;
                        for (std::size_t i = k + 1; i < n; ++i)
                            for (std::size_t j = k + 1; j < n; ++j)
                            {
                                const T product = a[i * n + k] * a[k * n + j];
                                a[i * n + j] -= product;
                            }
                    }
                    return 0U;
                }

                /** @brief Replaces the n * width right-hand sides at @p x by the solution. */
                template<typename T>
                void lu_solve_one(const T* lu, const std::uint32_t* pivots, T* x, std::size_t n,
                                  std::size_t width) noexcept
                {
                    SB_FP_CONTRACT_OFF
                    for (std::size_t i = 0; i < n; ++i)
                        swap_rows(x, width, i, pivots[i]);
                    for (std::size_t i = 1; i < n; ++i)
                        for (std::size_t k = 0; k < i; ++k)
                            for (std::size_t j = 0; j < width; ++j)
                            {
                                const T product = lu[i * n + k] * x[k * width + j];
                                x[i * width + j] -= product;
                            }
                    for (std::size_t i = n; i-- > 0;)
                    {
                        for (std::size_t k = i + 1; k < n; ++k)
                            for (std::size_t j = 0; j < width; ++j)
                            {
                                const T product = lu[i * n + k] * x[k * width + j];
                                x[i * width + j] -= product;
                            }
                        const T inverse = T(1) / lu[i * n + i];
                        for (std::size_t j = 0; j < width; ++j)
                            x[i * width + j] *= inverse;
                    }
                }
            } // namespace Detail

            /**
             * @brief Factors P A = L U with partial pivoting for each of @p batch matrices.
             * @param singular 1 where a pivot is not above n * eps * max|a|; that matrix stops.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            void lu_factor(T* matrices, std::uint32_t* pivots, std::uint32_t* singular,
                           std::size_t n, std::size_t batch) noexcept
            {
                for (std::size_t b = 0; b < batch; ++b)
                    singular[b] = Detail::lu_factor_one(matrices + b * n * n, pivots + b * n, n);
            }

            /**
             * @brief Solves A X = B for each of @p batch matrices lu_factor factored, in place.
             * @pre No matrix of the batch was reported singular.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            void lu_solve(const T* factors, const std::uint32_t* pivots, T* rhs, std::size_t n,
                          std::size_t rhs_count, std::size_t batch) noexcept
            {
                for (std::size_t b = 0; b < batch; ++b)
                    Detail::lu_solve_one(factors + b * n * n, pivots + b * n,
                                         rhs + b * n * rhs_count, n, rhs_count);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
