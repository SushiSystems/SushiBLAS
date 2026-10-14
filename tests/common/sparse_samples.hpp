/**************************************************************************/
/* sparse_samples.hpp                                                     */
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
 * @file sparse_samples.hpp
 * @brief Declares the host matrices the sparse tests build from and the host products they expect.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <random>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Names the element type and the index type one sparse suite runs with.
         */
        template<typename T, typename Index>
        struct SparseCase
        {
            /** @brief The element type of the values. */
            using Value = T;

            /** @brief The element type of the indices. */
            using IndexType = Index;
        };

        /** @brief Narrows a host expectation to the element type of a tensor. */
        template<typename T>
        [[nodiscard]] T narrowed(const std::complex<double>& value)
        {
            if constexpr (Core::is_complex_v<T>)
                return T(static_cast<typename T::value_type>(value.real()),
                         static_cast<typename T::value_type>(value.imag()));
            else
                return static_cast<T>(static_cast<float>(value.real()));
        }

        /** @brief Returns the integers start, start + 1, ... as complex doubles. */
        [[nodiscard]] inline std::vector<std::complex<double>> counting(std::size_t count,
                                                                        double start)
        {
            std::vector<std::complex<double>> values(count);
            for (std::size_t i = 0; i < count; ++i)
                values[i] = std::complex<double>(start + static_cast<double>(i), 0.0);
            return values;
        }

        /// @brief The element and index types a sparse suite covers: every dtype, both widths.
        using SparseCases = ::testing::Types<SparseCase<float, int32_t>,
                                             SparseCase<double, int64_t>,
                                             SparseCase<sycl::half, int64_t>,
                                             SparseCase<std::complex<float>, int64_t>,
                                             SparseCase<std::complex<double>, int32_t>>;

        /// @brief The two execution modes every sparse routine is tested in.
        constexpr ExecutionMode K_BOTH_MODES[2] = {ExecutionMode::ForceReference,
                                                   ExecutionMode::ForceSYCL};

        /// @brief The two index bases every sparse routine is tested in.
        constexpr Core::IndexBase K_BOTH_BASES[2] = {Core::IndexBase::Zero, Core::IndexBase::One};

        /**
         * @brief Holds a sparse matrix as zero-based triplets in host memory.
         * @see tests/common/README.md
         */
        template<typename T>
        struct HostTriplets
        {
            int64_t rows = 0;                    /**< Number of rows. */
            int64_t cols = 0;                    /**< Number of columns. */
            std::vector<int64_t> row_indices;    /**< The zero-based row of each entry. */
            std::vector<int64_t> column_indices; /**< The zero-based column of each entry. */
            std::vector<T> values;               /**< The value of each entry. */
        };

        /**
         * @brief Returns the 4 x 5 sample matrix: seven entries, row 1 empty, no two values equal.
         * @return Entries (0,0)=1 (0,3)=2 (2,1)=3 (2,2)=4 (2,4)=5 (3,0)=6 (3,4)=7, listed out of
         *         row order; a complex T adds the imaginary parts 2, -1, 1, 0, -2, 3, -1.
         * @see tests/common/README.md
         */
        template<typename T>
        [[nodiscard]] HostTriplets<T> sample_matrix()
        {
            using R = Kernels::RealType<T>;

            HostTriplets<T> m;
            m.rows = 4;
            m.cols = 5;
            m.row_indices = {3, 0, 2, 2, 0, 3, 2};
            m.column_indices = {4, 3, 2, 4, 0, 0, 1};

            const double real_parts[7] = {7, 2, 4, 5, 1, 6, 3};
            const double imaginary_parts[7] = {-1, -1, 0, -2, 2, 3, 1};
            for (std::size_t k = 0; k < 7; ++k)
            {
                if constexpr (Core::is_complex_v<T>)
                    m.values.push_back(T(static_cast<R>(real_parts[k]),
                                         static_cast<R>(imaginary_parts[k])));
                else
                    m.values.push_back(static_cast<T>(static_cast<float>(real_parts[k])));
            }
            return m;
        }

        /** @brief Returns the zero-based row offsets of sample_matrix() in CSR form. */
        [[nodiscard]] inline std::vector<int64_t> sample_row_offsets()
        {
            return {0, 2, 2, 5, 7};
        }

        /** @brief Returns the zero-based columns of sample_matrix() in CSR form, row by row. */
        [[nodiscard]] inline std::vector<int64_t> sample_columns()
        {
            return {0, 3, 1, 2, 4, 0, 4};
        }

        /**
         * @brief Returns the values of sample_matrix() in CSR form, row by row.
         * @return The real parts 1 to 7; a complex T adds the imaginary parts of sample_matrix().
         */
        template<typename T>
        [[nodiscard]] std::vector<std::complex<double>> sample_values()
        {
            const double imaginary_parts[7] = {2, -1, 1, 0, -2, 3, -1};

            std::vector<std::complex<double>> values(7);
            for (std::size_t k = 0; k < 7; ++k)
                values[k] = std::complex<double>(static_cast<double>(k + 1),
                                                 Core::is_complex_v<T> ? imaginary_parts[k] : 0.0);
            return values;
        }

        /**
         * @brief Returns a matrix of small integers in a pattern a seed fixes.
         * @param per_row Entries drawn for each row; two may fall on one column.
         * @return Values in [-3, 3], so every sum of up to 2^20 products with such a vector is
         *         exact in FLOAT32.
         */
        template<typename T>
        [[nodiscard]] HostTriplets<T> integer_valued_matrix(int64_t rows, int64_t cols,
                                                            int64_t per_row, std::uint32_t seed)
        {
            std::mt19937 generator(seed);
            std::uniform_int_distribution<int64_t> column(0, cols - 1);
            std::uniform_int_distribution<int> value(-3, 3);

            HostTriplets<T> m;
            m.rows = rows;
            m.cols = cols;
            for (int64_t i = 0; i < rows; ++i)
            {
                for (int64_t t = 0; t < per_row; ++t)
                {
                    m.row_indices.push_back(i);
                    m.column_indices.push_back(column(generator));
                    m.values.push_back(static_cast<T>(static_cast<float>(value(generator))));
                }
            }
            return m;
        }

        /** @brief Returns zero-based indices as the values a matrix of base stores, in Index. */
        template<typename Index>
        [[nodiscard]] std::vector<Index> stored_indices(const std::vector<int64_t>& zero_based,
                                                        Core::IndexBase base)
        {
            std::vector<Index> stored;
            stored.reserve(zero_based.size());
            for (const int64_t index : zero_based)
                stored.push_back(static_cast<Index>(index + Core::first_index(base)));
            return stored;
        }

        /** @brief Widens a real or complex host value to std::complex<double>. */
        template<typename T>
        [[nodiscard]] std::complex<double> widened(const T& value)
        {
            if constexpr (Core::is_complex_v<T>)
                return std::complex<double>(static_cast<double>(value.real()),
                                            static_cast<double>(value.imag()));
            else
                return std::complex<double>(static_cast<double>(static_cast<float>(value)), 0.0);
        }

        /**
         * @brief Returns alpha * op(A) * x + beta * y, computed on the host in double.
         * @param x One element per column of op(A).
         * @param y One element per row of op(A); the value before the call.
         */
        template<typename T>
        [[nodiscard]] std::vector<std::complex<double>> host_product(
            const HostTriplets<T>& a, Core::Transpose trans, std::complex<double> alpha,
            const std::vector<std::complex<double>>& x, std::complex<double> beta,
            const std::vector<std::complex<double>>& y)
        {
            std::vector<std::complex<double>> sums(y.size(), std::complex<double>(0.0, 0.0));
            for (std::size_t k = 0; k < a.values.size(); ++k)
            {
                const std::size_t row = static_cast<std::size_t>(a.row_indices[k]);
                const std::size_t column = static_cast<std::size_t>(a.column_indices[k]);
                const std::complex<double> value = widened(a.values[k]);

                if (!Core::is_transposed(trans))
                    sums[row] += value * x[column];
                else
                    sums[column] +=
                        (Core::is_conjugated(trans) ? std::conj(value) : value) * x[row];
            }

            std::vector<std::complex<double>> result(y.size());
            for (std::size_t i = 0; i < y.size(); ++i)
                result[i] = alpha * sums[i] + beta * y[i];
            return result;
        }
    } // namespace Test
} // namespace SushiBLAS
