/**************************************************************************/
/* small_integers.hpp                                                     */
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
 * @file small_integers.hpp
 * @brief Declares small_integers and unit_diagonal_matrix, the operands a kernel test fills.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns count values from -3 to 3, drawn from a generator seeded with seed.
         * @return Values a float holds exactly, so every small sum of their products is exact.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> small_integers(std::size_t count, std::uint32_t seed)
        {
            std::mt19937 generator(seed);
            std::uniform_int_distribution<int> draw(-3, 3);
            std::vector<T> values(count);
            for (T& value : values)
                value = static_cast<T>(draw(generator));
            return values;
        }

        /**
         * @brief Returns an n x n matrix of small integers with ones on the diagonal.
         * @return Row by row; a triangle of it solves to integers, so a solve is exact.
         */
        template<typename T>
        [[nodiscard]] std::vector<T> unit_diagonal_matrix(int64_t n, std::uint32_t seed)
        {
            std::vector<T> matrix = small_integers<T>(static_cast<std::size_t>(n * n), seed);
            for (int64_t i = 0; i < n; ++i)
                matrix[static_cast<std::size_t>(i * n + i)] = T(1);
            return matrix;
        }
    } // namespace Test
} // namespace SushiBLAS
