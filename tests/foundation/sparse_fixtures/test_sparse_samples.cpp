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
 * @file test_sparse_samples.cpp
 * @brief Tests the sample sparse matrices, the index storage and the host product they give.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SparseSamples, TheSampleMatrixHoldsSevenEntriesInsideItsShape)
        {
            const HostTriplets<double> m = sample_matrix<double>();

            EXPECT_EQ(m.rows, int64_t{4});
            EXPECT_EQ(m.cols, int64_t{5});
            ASSERT_EQ(m.values.size(), std::size_t{7});
            ASSERT_EQ(m.row_indices.size(), std::size_t{7});
            ASSERT_EQ(m.column_indices.size(), std::size_t{7});
            for (std::size_t k = 0; k < 7; ++k)
            {
                EXPECT_LT(m.row_indices[k], m.rows);
                EXPECT_LT(m.column_indices[k], m.cols);
            }
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
            const std::vector<int64_t> offsets = sample_row_offsets();

            EXPECT_EQ(offsets.back(), static_cast<int64_t>(sample_columns().size()));
            EXPECT_EQ(sample_values<double>().size(), sample_columns().size());
        }

        TEST(Small_SparseSamples, StoredIndicesAddTheFirstIndexOfTheBase)
        {
            const std::vector<int64_t> zero_based{0, 2, 5};

            EXPECT_EQ(stored_indices<int32_t>(zero_based, Core::IndexBase::Zero),
                      (std::vector<int32_t>{0, 2, 5}));
            EXPECT_EQ(stored_indices<int64_t>(zero_based, Core::IndexBase::One),
                      (std::vector<int64_t>{1, 3, 6}));
        }

        TEST(Small_SparseSamples, CountingAndWideningRoundTripExactValues)
        {
            const std::vector<std::complex<double>> values = counting(3, 2.0);

            EXPECT_EQ(values[2], std::complex<double>(4.0, 0.0));
            EXPECT_EQ(narrowed<float>(values[1]), 3.0f);
            EXPECT_EQ(widened(narrowed<std::complex<float>>(std::complex<double>(1.5, -2.0))),
                      std::complex<double>(1.5, -2.0));
        }

        TEST(Small_SparseSamples, AnIntegerMatrixHasItsRowsAndIsRepeatableForASeed)
        {
            const HostTriplets<float> first = integer_valued_matrix<float>(6, 9, 3, 42);
            const HostTriplets<float> second = integer_valued_matrix<float>(6, 9, 3, 42);

            ASSERT_EQ(first.values.size(), std::size_t{18});
            EXPECT_EQ(first.column_indices, second.column_indices);
            EXPECT_EQ(first.values, second.values);
            for (float value : first.values)
            {
                EXPECT_GE(value, -3.0f);
                EXPECT_LE(value, 3.0f);
            }
        }

        TEST(Small_SparseSamples, TheHostProductAppliesEachTransposeAndTheScalars)
        {
            HostTriplets<std::complex<double>> a;
            a.rows = 2;
            a.cols = 2;
            a.row_indices = {0, 1};
            a.column_indices = {1, 0};
            a.values = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 0.0)};
            const std::vector<std::complex<double>> x{{1.0, 0.0}, {3.0, 0.0}};
            const std::vector<std::complex<double>> y{{10.0, 0.0}, {20.0, 0.0}};
            const std::complex<double> one(1.0, 0.0);

            EXPECT_EQ(host_product(a, Core::Transpose::NoTrans, one, x, one, y),
                      (std::vector<std::complex<double>>{{13.0, 3.0}, {22.0, 0.0}}));
            EXPECT_EQ(host_product(a, Core::Transpose::Trans, one, x, 0.0, y),
                      (std::vector<std::complex<double>>{{6.0, 0.0}, {1.0, 1.0}}));
            EXPECT_EQ(host_product(a, Core::Transpose::ConjTrans, one, x, 0.0, y),
                      (std::vector<std::complex<double>>{{6.0, 0.0}, {1.0, -1.0}}));
        }

        TEST(Small_SparseSamples, BothBasesAreListed)
        {
            EXPECT_EQ(K_BOTH_BASES[0], Core::IndexBase::Zero);
            EXPECT_EQ(K_BOTH_BASES[1], Core::IndexBase::One);
        }
    } // namespace Test
} // namespace SushiBLAS
