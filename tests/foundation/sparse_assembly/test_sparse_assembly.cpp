/**************************************************************************/
/* test_sparse_assembly.cpp                                               */
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
 * @file test_sparse_assembly.cpp
 * @brief Tests assemble_csr and require_distinct_positions on host triplets and positions.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_assembly/sparse_assembly.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::IndexBase;
            using Sparse::assemble_csr;
            using Sparse::HostCsr;
            using Sparse::require_distinct_positions;

            /// @brief Rows of the integer-valued matrix whose rows are checked for order.
            constexpr int64_t K_ORDERED_ROWS = 50;

            /// @brief Columns of that matrix; few enough that most rows draw a column twice.
            constexpr int64_t K_ORDERED_COLS = 13;

            /// @brief Triplets drawn for each row of that matrix.
            constexpr int64_t K_ORDERED_PER_ROW = 8;

            /// @brief Seed of that matrix.
            constexpr std::uint32_t K_ORDERED_SEED = 20261007u;

            /// @brief The bits of a quiet FLOAT32 NaN that carries a payload.
            constexpr std::uint32_t K_NAN_BITS = 0x7fc12345u;

            /// @brief A column count one INT32 index cannot address: 2^31 + 5.
            constexpr int64_t K_COLS_PAST_INT32 = (int64_t{1} << 31) + 5;

            /** @brief Returns the CSR arrays of host triplets, stored in Index with base. */
            template<typename Index, typename T>
            [[nodiscard]] HostCsr<Index, T> assemble(const HostTriplets<T>& host, IndexBase base)
            {
                return assemble_csr(host.rows, host.cols, Core::first_index(base),
                                    stored_indices<Index>(host.row_indices, base),
                                    stored_indices<Index>(host.column_indices, base),
                                    host.values);
            }

            /** @brief Assembles FLOAT32 triplets with INT32 indices and drops the arrays. */
            void assemble_and_drop(int64_t rows, int64_t cols, int64_t first_index,
                                   const std::vector<int32_t>& row_indices,
                                   const std::vector<int32_t>& column_indices,
                                   const std::vector<float>& values)
            {
                (void)assemble_csr(rows, cols, first_index, row_indices, column_indices, values);
            }

            /** @brief Returns the one value three triplets on the pair (0, 0) assemble to. */
            template<typename T>
            [[nodiscard]] T folded_triple(T first, T second, T third)
            {
                const HostCsr<int32_t, T> csr = assemble_csr<int32_t, T>(
                    1, 1, 0, {0, 0, 0}, {0, 0, 0}, {first, second, third});
                return csr.values.size() == 1 ? csr.values[0] : T(0);
            }

            /** @brief Expects csr to hold the sample matrix, its indices stored with base. */
            template<typename Index>
            void expect_sample(const HostCsr<Index, float>& csr, IndexBase base)
            {
                EXPECT_EQ(csr.row_offsets, stored_indices<Index>(sample_row_offsets(), base));
                EXPECT_EQ(csr.column_indices, stored_indices<Index>(sample_columns(), base));
                EXPECT_EQ(csr.values, (std::vector<float>{1, 2, 3, 4, 5, 6, 7}));
            }

            /** @brief Expects the sample triplets to assemble to the sample arrays in Index. */
            template<typename Index>
            void expect_sample_assembles(IndexBase base)
            {
                expect_sample(assemble<Index, float>(sample_matrix<float>(), base), base);
            }

            /** @brief Returns the sample with the entry 4 given as two triplets and 7 as four. */
            [[nodiscard]] HostTriplets<float> sample_in_parts()
            {
                HostTriplets<float> host;
                host.rows = 4;
                host.cols = 5;
                host.row_indices = {3, 0, 2, 3, 2, 2, 0, 3, 3, 3, 2};
                host.column_indices = {4, 3, 2, 4, 4, 2, 0, 0, 4, 4, 1};
                host.values = {0.5f, 2.0f, 1.5f, 1.5f, 5.0f, 2.5f,
                               1.0f, 6.0f, 2.0f, 3.0f, 3.0f};
                return host;
            }

            /** @brief Returns host with its triplets rotated left by shift places. */
            [[nodiscard]] HostTriplets<float> rotated(HostTriplets<float> host, std::size_t shift)
            {
                const std::ptrdiff_t places = static_cast<std::ptrdiff_t>(shift);
                std::rotate(host.row_indices.begin(), host.row_indices.begin() + places,
                            host.row_indices.end());
                std::rotate(host.column_indices.begin(), host.column_indices.begin() + places,
                            host.column_indices.end());
                std::rotate(host.values.begin(), host.values.begin() + places, host.values.end());
                return host;
            }
        } // namespace

        TEST(Medium_SparseAssembly, TheSampleTripletsGiveTheSampleArraysUnderBothBases)
        {
            for (const IndexBase base : K_BOTH_BASES)
            {
                expect_sample_assembles<int32_t>(base);
                expect_sample_assembles<int64_t>(base);
            }
        }

        TEST(Medium_SparseAssembly, AnEntryGivenInPartsIsAddedWhereverThePartsArrive)
        {
            const HostTriplets<float> parts = sample_in_parts();

            for (std::size_t shift = 0; shift < parts.values.size(); ++shift)
            {
                SCOPED_TRACE(shift);
                expect_sample(assemble<int32_t, float>(rotated(parts, shift), IndexBase::Zero),
                              IndexBase::Zero);
            }
        }

        TEST(Medium_SparseAssembly, ARepeatedPairInFloat32IsSummedInDouble)
        {
            // Left to right in FLOAT32, 16777216 + 1 rounds back to 16777216 and the sum is 0.
            EXPECT_EQ(folded_triple<float>(16777216.0f, 1.0f, -16777216.0f), 1.0f);
        }

        TEST(Medium_SparseAssembly, ARepeatedPairInFloat64IsSummedWithItsRoundingErrors)
        {
            EXPECT_EQ(folded_triple<double>(1e16, 1.0, -1e16), 1.0);
        }

        TEST(Medium_SparseAssembly, APairThatCancelsStaysAStoredZero)
        {
            const HostCsr<int32_t, float> csr =
                assemble_csr<int32_t, float>(2, 3, 0, {1, 1}, {2, 2}, {1.0f, -1.0f});

            EXPECT_EQ(csr.row_offsets, (std::vector<int32_t>{0, 0, 1}));
            EXPECT_EQ(csr.column_indices, (std::vector<int32_t>{2}));
            EXPECT_EQ(csr.values, (std::vector<float>{0.0f}));
        }

        TEST(Medium_SparseAssembly, APairGivenOnceKeepsANaNAndItsBits)
        {
            float nan = 0.0f;
            std::memcpy(&nan, &K_NAN_BITS, sizeof(nan));
            ASSERT_TRUE(std::isnan(nan));

            const HostCsr<int32_t, float> csr =
                assemble_csr<int32_t, float>(2, 2, 0, {1, 0}, {0, 1}, {nan, 2.0f});

            ASSERT_EQ(csr.values.size(), std::size_t{2});
            std::uint32_t bits = 0;
            std::memcpy(&bits, &csr.values[1], sizeof(bits));
            EXPECT_EQ(bits, K_NAN_BITS);
            EXPECT_EQ(csr.values[0], 2.0f);
        }

        TEST(Medium_SparseAssembly, AMatrixWithoutTripletsHasEqualOffsets)
        {
            for (const IndexBase base : K_BOTH_BASES)
            {
                const int64_t first = Core::first_index(base);
                const HostCsr<int64_t, double> csr =
                    assemble_csr<int64_t, double>(3, 4, first, {}, {}, {});

                EXPECT_EQ(csr.row_offsets, std::vector<int64_t>(4, first));
                EXPECT_TRUE(csr.column_indices.empty());
                EXPECT_TRUE(csr.values.empty());
            }
        }

        TEST(Medium_SparseAssembly, AMatrixWithoutRowsHasOneOffset)
        {
            for (const IndexBase base : K_BOTH_BASES)
            {
                const int64_t first = Core::first_index(base);
                const HostCsr<int64_t, double> csr =
                    assemble_csr<int64_t, double>(0, 0, first, {}, {}, {});

                EXPECT_EQ(csr.row_offsets, std::vector<int64_t>(1, first));
            }
        }

        TEST(Medium_SparseAssembly, EveryRowAscendsStrictlyAndHoldsTheSumOfItsTriplets)
        {
            const HostTriplets<float> host = integer_valued_matrix<float>(
                K_ORDERED_ROWS, K_ORDERED_COLS, K_ORDERED_PER_ROW, K_ORDERED_SEED);
            const HostCsr<int32_t, float> csr = assemble<int32_t, float>(host, IndexBase::Zero);

            // Sums of integers in [-3, 3] are exact, so the dense image has one value per pair.
            const std::size_t cols = static_cast<std::size_t>(host.cols);
            std::vector<float> dense(static_cast<std::size_t>(host.rows) * cols, 0.0f);
            std::vector<bool> named(dense.size(), false);
            for (std::size_t k = 0; k < host.values.size(); ++k)
            {
                const std::size_t slot = static_cast<std::size_t>(host.row_indices[k]) * cols +
                                         static_cast<std::size_t>(host.column_indices[k]);
                dense[slot] += host.values[k];
                named[slot] = true;
            }

            ASSERT_EQ(csr.row_offsets.size(), static_cast<std::size_t>(host.rows) + 1);
            ASSERT_EQ(csr.row_offsets.front(), 0);
            ASSERT_EQ(static_cast<std::size_t>(csr.row_offsets.back()), csr.values.size());

            std::size_t stored = 0;
            for (std::size_t row = 0; row < static_cast<std::size_t>(host.rows); ++row)
            {
                ASSERT_LE(csr.row_offsets[row], csr.row_offsets[row + 1]) << "row " << row;
                const std::size_t begin = static_cast<std::size_t>(csr.row_offsets[row]);
                const std::size_t end = static_cast<std::size_t>(csr.row_offsets[row + 1]);

                for (std::size_t entry = begin; entry < end; ++entry)
                {
                    const std::size_t column = static_cast<std::size_t>(csr.column_indices[entry]);
                    if (entry > begin)
                        EXPECT_LT(csr.column_indices[entry - 1], csr.column_indices[entry])
                            << "row " << row;
                    EXPECT_TRUE(named[row * cols + column]) << "row " << row;
                    EXPECT_EQ(csr.values[entry], dense[row * cols + column]) << "row " << row;
                    ++stored;
                }
            }

            const std::ptrdiff_t pairs = std::count(named.begin(), named.end(), true);
            EXPECT_EQ(stored, static_cast<std::size_t>(pairs));
        }

        TEST(Medium_SparseAssembly, ANegativeExtentThrows)
        {
            EXPECT_THROW(assemble_and_drop(-1, 5, 0, {}, {}, {}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop(4, -1, 0, {}, {}, {}),
                         std::runtime_error);
        }

        TEST(Medium_SparseAssembly, TripletArraysOfDifferentLengthsThrow)
        {
            EXPECT_THROW(assemble_and_drop(4, 5, 0, {0, 1}, {0}, {1.0f, 2.0f}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop(4, 5, 0, {0, 1}, {0, 1}, {1.0f}),
                         std::runtime_error);
        }

        TEST(Medium_SparseAssembly, ATripletOutsideTheMatrixThrows)
        {
            EXPECT_THROW(assemble_and_drop(4, 5, 0, {4}, {0}, {1.0f}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop(4, 5, 0, {0}, {5}, {1.0f}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop(4, 5, 1, {1}, {0}, {1.0f}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop(4, 5, 1, {5}, {1}, {1.0f}),
                         std::runtime_error);
        }

        TEST(Medium_SparseAssembly, AColumnCountPastTheIndexTypeThrows)
        {
            EXPECT_THROW(assemble_and_drop(1, K_COLS_PAST_INT32, 0, {}, {}, {}),
                         std::runtime_error);
        }

        TEST(Medium_SparseAssembly, DistinctPositionsInsideTheVectorAreAccepted)
        {
            EXPECT_NO_THROW(require_distinct_positions<int32_t>(8, 0, {6, 1, 4}));
            EXPECT_NO_THROW(require_distinct_positions<int64_t>(8, 1, {7, 2, 5}));
        }

        TEST(Medium_SparseAssembly, ARepeatedPositionThrows)
        {
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 0, {6, 1, 6}),
                         std::runtime_error);
        }

        TEST(Medium_SparseAssembly, APositionOutsideTheVectorThrows)
        {
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 0, {6, 8, 4}),
                         std::runtime_error);
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 1, {7, 0, 5}),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
