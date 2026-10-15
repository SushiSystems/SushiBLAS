/**************************************************************************/
/* dense_oracle.hpp                                                       */
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
 * @file dense_oracle.hpp
 * @brief Declares the plain dense loops a BLAS kernel test takes its expected values from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns element (i, j) of a dense matrix of n columns given row by row.
         */
        template<typename T>
        [[nodiscard]] T element(const std::vector<T>& dense, int64_t n, int64_t i, int64_t j)
        {
            return dense[static_cast<std::size_t>(i * n + j)];
        }

        /**
         * @brief Returns alpha * op(A) * x + beta * y for an m x n matrix A given row by row.
         * @param transposed Uses A^T, which is n x m, in place of A.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> dense_matvec(const std::vector<T>& a, int64_t m, int64_t n,
                                                  bool transposed, T alpha,
                                                  const std::vector<T>& x, T beta,
                                                  const std::vector<T>& y)
        {
            const int64_t rows = transposed ? n : m;
            const int64_t columns = transposed ? m : n;
            std::vector<T> out(static_cast<std::size_t>(rows));
            for (int64_t i = 0; i < rows; ++i)
            {
                T sum(0);
                for (int64_t k = 0; k < columns; ++k)
                    sum += (transposed ? element(a, n, k, i) : element(a, n, i, k)) *
                           x[static_cast<std::size_t>(k)];
                out[static_cast<std::size_t>(i)] =
                    alpha * sum + beta * y[static_cast<std::size_t>(i)];
            }
            return out;
        }

        /**
         * @brief Returns the n x n symmetric matrix one triangle of a dense matrix stands for.
         * @param upper Reads the upper triangle; otherwise the lower.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> symmetric_from(const std::vector<T>& dense, int64_t n,
                                                    bool upper)
        {
            std::vector<T> out(dense.size());
            for (int64_t i = 0; i < n; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    const bool stored = upper ? (i <= j) : (i >= j);
                    out[static_cast<std::size_t>(i * n + j)] =
                        stored ? element(dense, n, i, j) : element(dense, n, j, i);
                }
            }
            return out;
        }

        /**
         * @brief Returns the triangle of an n x n dense matrix, with zeros elsewhere.
         * @param unit Replaces the diagonal with ones.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> triangle_of(const std::vector<T>& dense, int64_t n,
                                                 bool upper, bool unit)
        {
            std::vector<T> out(dense.size(), T(0));
            for (int64_t i = 0; i < n; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    if (upper ? (i <= j) : (i >= j))
                        out[static_cast<std::size_t>(i * n + j)] =
                            (unit && i == j) ? T(1) : element(dense, n, i, j);
                }
            }
            return out;
        }

        /**
         * @brief Returns the band of an m x n dense matrix, kl below and ku above the diagonal.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> band_of(const std::vector<T>& dense, int64_t m, int64_t n,
                                             int64_t kl, int64_t ku)
        {
            std::vector<T> out(dense.size(), T(0));
            for (int64_t i = 0; i < m; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    if (j - i <= ku && i - j <= kl)
                        out[static_cast<std::size_t>(i * n + j)] = element(dense, n, i, j);
                }
            }
            return out;
        }

        /**
         * @brief Adds alpha * (x * y^T + y * x^T) to one triangle of an n x n dense matrix.
         * @param rank_two Adds only alpha * x * x^T, ignoring y, when false.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> triangle_update(std::vector<T> dense, int64_t n, bool upper,
                                                     T alpha, const std::vector<T>& x,
                                                     const std::vector<T>& y, bool rank_two)
        {
            for (int64_t i = 0; i < n; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    if (!(upper ? (i <= j) : (i >= j)))
                        continue;
                    const T xi = x[static_cast<std::size_t>(i)];
                    const T xj = x[static_cast<std::size_t>(j)];
                    T term = xi * xj;
                    if (rank_two)
                        term = xi * y[static_cast<std::size_t>(j)] +
                               y[static_cast<std::size_t>(i)] * xj;
                    dense[static_cast<std::size_t>(i * n + j)] += alpha * term;
                }
            }
            return dense;
        }
    } // namespace Test
} // namespace SushiBLAS
