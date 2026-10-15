/**************************************************************************/
/* integer_csr.hpp                                                        */
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
 * @file integer_csr.hpp
 * @brief Declares IntegerCsr, a CSR matrix of small integers that a sparse kernel test fills.
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
        /** @brief Holds a zero-based CSR matrix whose stored values are integers from -3 to 3. */
        struct IntegerCsr
        {
            int64_t rows = 0;             /**< Number of rows. */
            int64_t cols = 0;             /**< Number of columns. */
            std::vector<int64_t> offsets; /**< rows + 1 offsets into columns and values. */
            std::vector<int64_t> columns; /**< Each entry's column, ascending in a row. */
            std::vector<double> values;   /**< The value of each stored entry. */

            /** @brief Returns the number of stored entries. */
            [[nodiscard]] int64_t nnz() const noexcept
            {
                return static_cast<int64_t>(values.size());
            }

            /** @brief Returns the matrix dense, row by row, with zeros where nothing is stored. */
            [[nodiscard]] std::vector<double> dense() const
            {
                std::vector<double> out(static_cast<std::size_t>(rows * cols), 0.0);
                for (int64_t i = 0; i < rows; ++i)
                    for (int64_t e = offsets[static_cast<std::size_t>(i)];
                         e < offsets[static_cast<std::size_t>(i + 1)]; ++e)
                        out[static_cast<std::size_t>(i * cols +
                                                     columns[static_cast<std::size_t>(e)])] =
                            values[static_cast<std::size_t>(e)];
                return out;
            }
        };

        /**
         * @brief Returns a rows x cols CSR matrix with about a quarter of its entries stored.
         * @param with_diagonal Stores every diagonal entry as well, with the value 1.
         */
        [[nodiscard]] inline IntegerCsr integer_csr(int64_t rows, int64_t cols, std::uint32_t seed,
                                                    bool with_diagonal)
        {
            std::mt19937 generator(seed);
            std::uniform_int_distribution<int> value(-3, 3);
            std::uniform_int_distribution<int> keep(0, 3);
            IntegerCsr m;
            m.rows = rows;
            m.cols = cols;
            m.offsets.push_back(0);
            for (int64_t i = 0; i < rows; ++i)
            {
                for (int64_t j = 0; j < cols; ++j)
                {
                    const bool diagonal = with_diagonal && i == j;
                    if (!diagonal && keep(generator) != 0)
                        continue;
                    m.columns.push_back(j);
                    m.values.push_back(diagonal ? 1.0 : static_cast<double>(value(generator)));
                }
                m.offsets.push_back(static_cast<int64_t>(m.columns.size()));
            }
            return m;
        }
    } // namespace Test
} // namespace SushiBLAS
