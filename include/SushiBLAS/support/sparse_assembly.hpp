/**************************************************************************/
/* sparse_assembly.hpp                                                    */
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
 * @file sparse_assembly.hpp
 * @brief Declares the host-side assembly of CSR arrays from triplets, and its input checks.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Sparse
    {
        /**
         * @brief Holds the three arrays of one CSR matrix in host memory.
         * @tparam Index int32_t or int64_t.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename Index, typename T>
        struct HostCsr
        {
            std::vector<Index> row_offsets;    /**< rows + 1 offsets, the first the index base. */
            std::vector<Index> column_indices; /**< The column of each entry, ascending in a row. */
            std::vector<T> values;             /**< The value of each entry. */
        };

        /**
         * @brief Throws unless Index can hold largest.
         * @tparam Index int32_t or int64_t.
         * @param what Names the quantity in the message, such as "entry count".
         * @throws std::runtime_error naming the quantity and the limit of Index.
         */
        template<typename Index>
        void require_index_capacity(int64_t largest, const char* what)
        {
            static_assert(std::is_same_v<Index, int32_t> || std::is_same_v<Index, int64_t>,
                          "A sparse index is an int32_t or an int64_t.");

            const int64_t limit = static_cast<int64_t>(std::numeric_limits<Index>::max());
            SB_THROW_IF(largest > limit,
                        "The {} needs the index {}, above the largest value of its index type, {}.",
                        what, largest, limit);
        }

        /**
         * @brief Throws unless every stored position lies inside the vector and none repeats.
         * @param first_index The stored value that names position 0.
         * @throws std::runtime_error naming the first entry that breaks either rule.
         */
        template<typename Index>
        void require_distinct_positions(int64_t size, int64_t first_index,
                                        const std::vector<Index>& indices)
        {
            SB_THROW_IF(size < 0, "A sparse vector cannot have the size {}.", size);
            require_index_capacity<Index>(size - 1 + first_index, "last position");

            std::vector<bool> seen(static_cast<std::size_t>(size), false);
            for (std::size_t k = 0; k < indices.size(); ++k)
            {
                const int64_t stored = static_cast<int64_t>(indices[k]);
                const int64_t position = stored - first_index;
                SB_THROW_IF(position < 0 || position >= size,
                            "Entry {} holds the index {}, outside [{}, {}).",
                            k, stored, first_index, first_index + size);
                SB_THROW_IF(seen[static_cast<std::size_t>(position)],
                            "Entry {} repeats the index {}; a sparse vector stores a position "
                            "once.",
                            k, stored);
                seen[static_cast<std::size_t>(position)] = true;
            }
        }

        /**
         * @brief Returns the sum of values[first, last): the element itself when there is one,
         *        and a Kernels::ReferenceSum rounded once to T when there are more.
         * @pre first is below last.
         */
        template<typename T>
        [[nodiscard]] T sum_of_run(const std::vector<T>& values, std::size_t first,
                                   std::size_t last)
        {
            using RW = Kernels::ReferenceAccumulator<T>;

            if (last - first == 1)
                return values[first];

            Kernels::ReferenceSum<T> sum;
            for (std::size_t k = first; k < last; ++k)
                sum.add(static_cast<RW>(values[k]));
            return static_cast<T>(sum.value());
        }

        /**
         * @brief Assembles triplets into the three arrays of a CSR matrix, adding repeated pairs.
         * @param first_index The stored value that names row 0 and column 0, on input and output.
         * @return Each (row, column) pair once, by ascending column in a row; the values of a
         *         repeated pair are added in input order.
         * @throws std::runtime_error on a negative extent, arrays of different lengths, a triplet
         *         outside the matrix, or a count Index cannot hold.
         */
        template<typename Index, typename T>
        [[nodiscard]] HostCsr<Index, T> assemble_csr(int64_t rows, int64_t cols,
                                                     int64_t first_index,
                                                     const std::vector<Index>& row_indices,
                                                     const std::vector<Index>& column_indices,
                                                     const std::vector<T>& values)
        {
            SB_THROW_IF(rows < 0 || cols < 0, "A sparse matrix cannot be {} x {}.", rows, cols);
            SB_THROW_IF(row_indices.size() != values.size() ||
                            column_indices.size() != values.size(),
                        "The triplet arrays hold {} rows, {} columns and {} values; the three "
                        "counts must match.",
                        row_indices.size(), column_indices.size(), values.size());

            const std::size_t entries = values.size();
            require_index_capacity<Index>(cols - 1 + first_index, "last column");

            std::vector<int64_t> column_start(static_cast<std::size_t>(cols) + 1, 0);
            std::vector<int64_t> row_start(static_cast<std::size_t>(rows) + 1, 0);
            for (std::size_t k = 0; k < entries; ++k)
            {
                const int64_t row = static_cast<int64_t>(row_indices[k]) - first_index;
                const int64_t column = static_cast<int64_t>(column_indices[k]) - first_index;
                SB_THROW_IF(row < 0 || row >= rows || column < 0 || column >= cols,
                            "Triplet {} names the entry ({}, {}), outside a {} x {} matrix whose "
                            "indices start at {}.",
                            k, static_cast<int64_t>(row_indices[k]),
                            static_cast<int64_t>(column_indices[k]), rows, cols, first_index);
                ++column_start[static_cast<std::size_t>(column) + 1];
                ++row_start[static_cast<std::size_t>(row) + 1];
            }
            for (std::size_t c = 0; c < static_cast<std::size_t>(cols); ++c)
                column_start[c + 1] += column_start[c];
            for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r)
                row_start[r + 1] += row_start[r];

            // by_column lists the triplets by ascending column, input order kept inside a column.
            std::vector<std::size_t> by_column(entries);
            for (std::size_t k = 0; k < entries; ++k)
            {
                const std::size_t column =
                    static_cast<std::size_t>(static_cast<int64_t>(column_indices[k]) - first_index);
                by_column[static_cast<std::size_t>(column_start[column]++)] = k;
            }

            // Placing by_column in order keeps each row's entries by ascending column.
            HostCsr<Index, T> csr;
            csr.column_indices.resize(entries);
            csr.values.resize(entries);
            for (const std::size_t k : by_column)
            {
                const int64_t row = static_cast<int64_t>(row_indices[k]) - first_index;
                const std::size_t slot =
                    static_cast<std::size_t>(row_start[static_cast<std::size_t>(row)]++);
                csr.column_indices[slot] = column_indices[k];
                csr.values[slot] = values[k];
            }

            // row_start[r] is now the end of row r; kept never passes the entry being read.
            csr.row_offsets.resize(static_cast<std::size_t>(rows) + 1);
            std::size_t kept = 0;
            std::size_t begin = 0;
            for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r)
            {
                const std::size_t end = static_cast<std::size_t>(row_start[r]);
                csr.row_offsets[r] = static_cast<Index>(static_cast<int64_t>(kept) + first_index);

                std::size_t k = begin;
                while (k < end)
                {
                    std::size_t run_end = k + 1;
                    while (run_end < end && csr.column_indices[run_end] == csr.column_indices[k])
                        ++run_end;

                    const T folded = sum_of_run(csr.values, k, run_end);
                    csr.column_indices[kept] = csr.column_indices[k];
                    csr.values[kept] = folded;
                    ++kept;
                    k = run_end;
                }
                begin = end;
            }

            require_index_capacity<Index>(static_cast<int64_t>(kept) + first_index, "entry count");
            csr.row_offsets[static_cast<std::size_t>(rows)] =
                static_cast<Index>(static_cast<int64_t>(kept) + first_index);
            csr.column_indices.resize(kept);
            csr.values.resize(kept);
            return csr;
        }
    } // namespace Sparse
} // namespace SushiBLAS
