/**************************************************************************/
/* lu_index.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */

/**
 * @file lu_index.hpp
 * @brief Declares the pivot rule and the cell coordinates of the batched LU factorisation and its solve, shared by the reference and SYCL kernels.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <cstddef>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds the row a pivot search chose and the magnitude found there. */
            template<typename T>
            struct LuPivot
            {
                std::size_t row;   /**< The chosen row. */
                T magnitude;       /**< The absolute value at that row. */
            };

            /** @brief Holds a row and a column of a matrix. */
            struct LuCell
            {
                std::size_t row;      /**< The row. */
                std::size_t column;   /**< The column. */
            };

            /** @brief Holds the matrix of a batch and the right-hand side column one work-item solves. */
            struct LuSolveCell
            {
                std::size_t matrix;   /**< The matrix of the batch. */
                std::size_t column;   /**< The right-hand side column. */
            };

            /** @brief Returns the absolute value of @p value without a library call. */
            template<typename T>
            [[nodiscard]] constexpr T lu_magnitude(T value) noexcept
            {
                return value < T(0) ? -value : value;
            }

            /** @brief Returns `n * eps * largest`, the bound a pivot must exceed. */
            template<typename T>
            [[nodiscard]] constexpr T lu_threshold(std::size_t n, T largest) noexcept
            {
                return static_cast<T>(n) * std::numeric_limits<T>::epsilon() * largest;
            }

            /** @brief Returns whether a pivot of @p magnitude is above @p threshold; a NaN is not. */
            template<typename T>
            [[nodiscard]] constexpr bool lu_pivot_usable(T magnitude, T threshold) noexcept
            {
                return magnitude > threshold;
            }

            /**
             * @brief Returns the row from @p step down with the largest magnitude in column @p step.
             * @note Strictly greater wins, so a tie keeps the lowest row index.
             * @pre @p step is below @p n.
             */
            template<typename T>
            [[nodiscard]] constexpr LuPivot<T> lu_pivot_search(const T* a, std::size_t n,
                                                               std::size_t step) noexcept
            {
                LuPivot<T> best{step, lu_magnitude(a[step * n + step])};
                for (std::size_t i = step + 1; i < n; ++i)
                {
                    const T magnitude = lu_magnitude(a[i * n + step]);
                    if (magnitude > best.magnitude)
                        best = LuPivot<T>{i, magnitude};
                }
                return best;
            }

            /**
             * @brief Returns the cell of the trailing block that flat index @p flat names at @p step.
             * @param width The block's side: `n - step - 1`, above zero.
             */
            [[nodiscard]] constexpr LuCell lu_trailing_cell(std::size_t step, std::size_t width,
                                                            std::size_t flat) noexcept
            {
                return LuCell{step + 1 + flat / width, step + 1 + flat % width};
            }

            /** @brief Returns the matrix and the column of flat solve index @p flat. */
            [[nodiscard]] constexpr LuSolveCell lu_solve_cell(std::size_t flat,
                                                              std::size_t rhs_count) noexcept
            {
                return LuSolveCell{flat / rhs_count, flat % rhs_count};
            }
        } // namespace Indexing
    } // namespace Kernels
} // namespace SushiBLAS
