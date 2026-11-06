/**************************************************************************/
/* test_sparse_samples.cpp                                                */
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
/**************************************************************************/

/**
 * @file test_sparse_samples.cpp
 * @brief Tests the sample sparse matrices, the index storage and the host product they give.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Expects every entry of a triplet matrix to lie inside its shape.
        template <typename T>
        static void expect_entries_inside_shape(const HostTriplets<T>& m)
        {
            for (std::size_t k = 0; k < m.values.size(); ++k)
            {
                EXPECT_GE(m.row_indices[k], int64_t{0}) << "entry " << k;
                EXPECT_LT(m.row_indices[k], m.rows) << "entry " << k;
                EXPECT_GE(m.column_indices[k], int64_t{0}) << "entry " << k;
                EXPECT_LT(m.column_indices[k], m.cols) << "entry " << k;
            }
        }

        /// @brief Expects every value to lie in the closed range [-3, 3].
        template <typename T>
        static void expect_values_within_three(const std::vector<T>& values)
        {
            for (const T value : values)
            {
                EXPECT_GE(value, static_cast<T>(-3));
                EXPECT_LE(value, static_cast<T>(3));
            }
        }

        /// @brief Returns the entries of one row of a triplet matrix as column -> value, ascending by column.
        static std::map<int64_t, double> entries_of_row(const HostTriplets<double>& m, int64_t row)
        {
            std::map<int64_t, double> entries;
            for (std::size_t k = 0; k < m.values.size(); ++k)
            {
                if (m.row_indices[k] == row)
                {
                    entries[m.column_indices[k]] = m.values[k];
                }
            }
            return entries;
        }

        /// @brief Expects the CSR slice [first, last) to hold exactly the entries of a row, by column and value.
        static void expect_csr_slice_is(const std::map<int64_t, double>& entries, int64_t first,
                                        int64_t last)
        {
            const std::vector<int64_t> columns = sample_columns();
            const std::vector<std::complex<double>> values = sample_values<double>();

            ASSERT_EQ(last - first, static_cast<int64_t>(entries.size()));
            int64_t e = first;
            for (const auto& [column, value] : entries)
            {
                EXPECT_EQ(columns[static_cast<std::size_t>(e)], column) << "entry " << e;
                EXPECT_EQ(values[static_cast<std::size_t>(e)].real(), value) << "entry " << e;
                ++e;
            }
        }

        /// @brief Returns the 2 by 2 complex matrix whose host products the tests below expect.
        static HostTriplets<std::complex<double>> small_complex_matrix()
        {
            HostTriplets<std::complex<double>> a;
            a.rows = 2;
            a.cols = 2;
            a.row_indices = {0, 1};
            a.column_indices = {1, 0};
            a.values = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 0.0)};
            return a;
        }

        TEST(Small_SparseSamples, TheSampleMatrixHoldsSevenEntriesInsideItsShape)
        {
            const HostTriplets<double> m = sample_matrix<double>();

            EXPECT_EQ(m.rows, int64_t{4});
            EXPECT_EQ(m.cols, int64_t{5});
            ASSERT_EQ(m.values.size(), std::size_t{7});
            ASSERT_EQ(m.row_indices.size(), std::size_t{7});
            ASSERT_EQ(m.column_indices.size(), std::size_t{7});
            expect_entries_inside_shape(m);
        }

        TEST(Small_SparseSamples, AComplexSampleCarriesImaginaryParts)
        {
            const HostTriplets<std::complex<float>> m = sample_matrix<std::complex<float>>();

            EXPECT_EQ(m.values[0], std::complex<float>(7.0f, -1.0f));
            EXPECT_EQ(sample_values<double>()[0], std::complex<double>(1.0, 0.0));
            EXPECT_EQ(sample_values<std::complex<double>>()[0], std::complex<double>(1.0, 2.0));
        }

        TEST(Small_SparseSamples, TheSampleCSRArraysAgree)
        {
            const HostTriplets<double> m = sample_matrix<double>();
            const std::vector<int64_t> offsets = sample_row_offsets();

            ASSERT_EQ(offsets.size(), static_cast<std::size_t>(m.rows) + 1);
            EXPECT_EQ(offsets.back(), static_cast<int64_t>(m.values.size()));
            for (int64_t row = 0; row < m.rows; ++row)
            {
                expect_csr_slice_is(entries_of_row(m, row), offsets[static_cast<std::size_t>(row)],
                                    offsets[static_cast<std::size_t>(row + 1)]);
            }
        }

        TEST(Small_SparseSamples, StoredIndicesAddTheFirstIndexOfTheBase)
        {
            const std::vector<int64_t> zero_based{0, 2, 5};

            EXPECT_EQ(stored_indices<int32_t>(zero_based, Core::IndexBase::Zero),
                      (std::vector<int32_t>{0, 2, 5}));
            EXPECT_EQ(stored_indices<int64_t>(zero_based, Core::IndexBase::One),
                      (std::vector<int64_t>{1, 3, 6}));
        }

        TEST(Small_SparseSamples, CountingGivesConsecutiveIntegersFromAStart)
        {
            const std::vector<std::complex<double>> values = counting(3, 2.0);

            EXPECT_EQ(values, (std::vector<std::complex<double>>{{2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}));
        }

        TEST(Small_SparseSamples, NarrowingThenWideningRoundTripsExactValues)
        {
            EXPECT_EQ(narrowed<float>(std::complex<double>(3.0, 0.0)), 3.0f);
            EXPECT_EQ(widened(narrowed<std::complex<float>>(std::complex<double>(1.5, -2.0))),
                      std::complex<double>(1.5, -2.0));
        }

        TEST(Small_SparseSamples, AnIntegerMatrixHoldsTheRowsItWasAskedFor)
        {
            const HostTriplets<float> m = integer_valued_matrix<float>(6, 9, 3, 42);

            EXPECT_EQ(m.row_indices, (std::vector<int64_t>{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                                           4, 4, 4, 5, 5, 5}));
            EXPECT_EQ(m.values.size(), std::size_t{18});
            expect_entries_inside_shape(m);
        }

        TEST(Small_SparseSamples, AnIntegerMatrixHoldsValuesFromMinusThreeToThree)
        {
            expect_values_within_three(integer_valued_matrix<float>(6, 9, 3, 42).values);
        }

        TEST(Small_SparseSamples, AnIntegerMatrixIsRepeatableForASeed)
        {
            const HostTriplets<float> first = integer_valued_matrix<float>(6, 9, 3, 42);
            const HostTriplets<float> second = integer_valued_matrix<float>(6, 9, 3, 42);

            EXPECT_EQ(first.row_indices, second.row_indices);
            EXPECT_EQ(first.column_indices, second.column_indices);
            EXPECT_EQ(first.values, second.values);
        }

        TEST(Small_SparseSamples, TheHostProductAppliesNoTransposeAndTheScalars)
        {
            const std::vector<std::complex<double>> x{{1.0, 0.0}, {3.0, 0.0}};
            const std::vector<std::complex<double>> y{{10.0, 0.0}, {20.0, 0.0}};
            const std::complex<double> one(1.0, 0.0);

            EXPECT_EQ(host_product(small_complex_matrix(), Core::Transpose::NoTrans, one, x, one, y),
                      (std::vector<std::complex<double>>{{13.0, 3.0}, {22.0, 0.0}}));
        }

        TEST(Small_SparseSamples, TheHostProductAppliesTranspose)
        {
            const std::vector<std::complex<double>> x{{1.0, 0.0}, {3.0, 0.0}};
            const std::vector<std::complex<double>> y{{10.0, 0.0}, {20.0, 0.0}};
            const std::complex<double> one(1.0, 0.0);

            EXPECT_EQ(host_product(small_complex_matrix(), Core::Transpose::Trans, one, x, 0.0, y),
                      (std::vector<std::complex<double>>{{6.0, 0.0}, {1.0, 1.0}}));
        }

        TEST(Small_SparseSamples, TheHostProductAppliesConjugateTranspose)
        {
            const std::vector<std::complex<double>> x{{1.0, 0.0}, {3.0, 0.0}};
            const std::vector<std::complex<double>> y{{10.0, 0.0}, {20.0, 0.0}};
            const std::complex<double> one(1.0, 0.0);

            EXPECT_EQ(host_product(small_complex_matrix(), Core::Transpose::ConjTrans, one, x, 0.0, y),
                      (std::vector<std::complex<double>>{{6.0, 0.0}, {1.0, -1.0}}));
        }
    } // namespace Test
} // namespace SushiBLAS
