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
#include <string>
#include <utility>
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

            /// @brief Holds the number of triplets of the sample given in parts.
            constexpr std::size_t K_PART_COUNT = 11;

            /// @brief Names an index width and a base the sample is stored in.
            struct SampleRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the base of the stored indices.
                IndexBase base;
                /// @brief Holds the check of the sample in the row's index width.
                void (*expect_assembles)(IndexBase);
            };

            /// @brief Holds the arguments of one call to assemble_csr that must be refused.
            struct RefusedRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the row count.
                int64_t rows;
                /// @brief Holds the column count.
                int64_t cols;
                /// @brief Holds the first index of the base.
                int64_t first_index;
                /// @brief Holds the row index of each triplet.
                std::vector<int32_t> row_indices;
                /// @brief Holds the column index of each triplet.
                std::vector<int32_t> column_indices;
                /// @brief Holds the value of each triplet.
                std::vector<float> values;
            };

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
            void assemble_and_drop(const RefusedRow& row)
            {
                (void)assemble_csr(row.rows, row.cols, row.first_index, row.row_indices,
                                   row.column_indices, row.values);
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

            /** @brief Returns the dense image of the triplets: one summed value per pair. */
            [[nodiscard]] std::vector<float> dense_of(const HostTriplets<float>& host)
            {
                const std::size_t cols = static_cast<std::size_t>(host.cols);
                std::vector<float> dense(static_cast<std::size_t>(host.rows) * cols, 0.0f);
                for (std::size_t k = 0; k < host.values.size(); ++k)
                {
                    dense[static_cast<std::size_t>(host.row_indices[k]) * cols +
                          static_cast<std::size_t>(host.column_indices[k])] += host.values[k];
                }
                return dense;
            }

            /** @brief Returns the dense image of the CSR arrays of a zero-based matrix. */
            [[nodiscard]] std::vector<float> dense_of(const HostCsr<int32_t, float>& csr,
                                                      std::size_t rows, std::size_t cols)
            {
                std::vector<float> dense(rows * cols, 0.0f);
                for (std::size_t row = 0; row < rows; ++row)
                {
                    for (int32_t entry = csr.row_offsets[row]; entry < csr.row_offsets[row + 1];
                         ++entry)
                    {
                        dense[row * cols + static_cast<std::size_t>(csr.column_indices[entry])] =
                            csr.values[static_cast<std::size_t>(entry)];
                    }
                }
                return dense;
            }

            /** @brief Returns the number of distinct (row, column) pairs the triplets name. */
            [[nodiscard]] std::size_t distinct_pairs_of(const HostTriplets<float>& host)
            {
                std::vector<std::pair<int32_t, int32_t>> pairs;
                for (std::size_t k = 0; k < host.values.size(); ++k)
                    pairs.emplace_back(static_cast<int32_t>(host.row_indices[k]),
                                       static_cast<int32_t>(host.column_indices[k]));
                std::sort(pairs.begin(), pairs.end());
                return static_cast<std::size_t>(std::unique(pairs.begin(), pairs.end()) -
                                                pairs.begin());
            }

            /** @brief Returns whether the columns of every row of csr strictly ascend. */
            [[nodiscard]] bool rows_ascend_strictly(const HostCsr<int32_t, float>& csr)
            {
                for (std::size_t row = 0; row + 1 < csr.row_offsets.size(); ++row)
                {
                    for (int32_t entry = csr.row_offsets[row] + 1; entry < csr.row_offsets[row + 1];
                         ++entry)
                    {
                        if (csr.column_indices[entry - 1] >= csr.column_indices[entry])
                            return false;
                    }
                }
                return true;
            }

            /** @brief Returns the matrix whose rows are checked for order. */
            [[nodiscard]] HostTriplets<float> ordered_matrix()
            {
                return integer_valued_matrix<float>(K_ORDERED_ROWS, K_ORDERED_COLS,
                                                    K_ORDERED_PER_ROW, K_ORDERED_SEED);
            }

            /** @brief Names a test case after its row. */
            template<typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Runs the sample in one index width and one base.
        class Small_SparseAssemblySample : public ::testing::TestWithParam<SampleRow>
        {
        };

        /** @brief Tests that the sample triplets assemble to the sample arrays. */
        TEST_P(Small_SparseAssemblySample, TheSampleTripletsGiveTheSampleArrays)
        {
            GetParam().expect_assembles(GetParam().base);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_SparseAssemblySample,
            ::testing::Values(
                SampleRow{"Int32ZeroBased", IndexBase::Zero, &expect_sample_assembles<int32_t>},
                SampleRow{"Int32OneBased", IndexBase::One, &expect_sample_assembles<int32_t>},
                SampleRow{"Int64ZeroBased", IndexBase::Zero, &expect_sample_assembles<int64_t>},
                SampleRow{"Int64OneBased", IndexBase::One, &expect_sample_assembles<int64_t>}),
            row_name<SampleRow>);

        /// @brief Runs the sample given in parts at one rotation of its triplets.
        class Small_SparseAssemblyRotation : public ::testing::TestWithParam<std::size_t>
        {
        };

        /** @brief Tests that an entry given in parts is added wherever its parts arrive. */
        TEST_P(Small_SparseAssemblyRotation, AnEntryGivenInPartsIsAddedWhereverThePartsArrive)
        {
            const HostTriplets<float> parts = sample_in_parts();
            ASSERT_EQ(parts.values.size(), K_PART_COUNT);

            expect_sample(assemble<int32_t, float>(rotated(parts, GetParam()), IndexBase::Zero),
                          IndexBase::Zero);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseAssemblyRotation,
                                 ::testing::Range<std::size_t>(0, K_PART_COUNT));

        /** @brief Tests that a repeated FLOAT32 pair is summed in double. */
        TEST(Small_SparseAssembly, ARepeatedPairInFloat32IsSummedInDouble)
        {
            // Left to right in FLOAT32, 16777216 + 1 rounds back to 16777216 and the sum is 0.
            EXPECT_EQ(folded_triple<float>(16777216.0f, 1.0f, -16777216.0f), 1.0f);
        }

        /** @brief Tests that a repeated FLOAT64 pair is summed with compensation. */
        TEST(Small_SparseAssembly, ARepeatedPairInFloat64IsSummedWithItsRoundingErrors)
        {
            EXPECT_EQ(folded_triple<double>(1e16, 1.0, -1e16), 1.0);
        }

        /** @brief Tests that a pair that cancels stays a stored zero. */
        TEST(Small_SparseAssembly, APairThatCancelsStaysAStoredZero)
        {
            const HostCsr<int32_t, float> csr =
                assemble_csr<int32_t, float>(2, 3, 0, {1, 1}, {2, 2}, {1.0f, -1.0f});

            EXPECT_EQ(csr.row_offsets, (std::vector<int32_t>{0, 0, 1}));
            EXPECT_EQ(csr.column_indices, (std::vector<int32_t>{2}));
            EXPECT_EQ(csr.values, (std::vector<float>{0.0f}));
        }

        /** @brief Tests that a pair given once keeps a NaN and its payload bit for bit. */
        TEST(Small_SparseAssembly, APairGivenOnceKeepsANaNAndItsBits)
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

        /// @brief Runs a case in one index base.
        class Small_SparseAssemblyBase : public ::testing::TestWithParam<IndexBase>
        {
        };

        /** @brief Tests that a matrix without triplets has equal offsets. */
        TEST_P(Small_SparseAssemblyBase, AMatrixWithoutTripletsHasEqualOffsets)
        {
            const int64_t first = Core::first_index(GetParam());

            const HostCsr<int64_t, double> csr =
                assemble_csr<int64_t, double>(3, 4, first, {}, {}, {});

            EXPECT_EQ(csr.row_offsets, std::vector<int64_t>(4, first));
            EXPECT_TRUE(csr.column_indices.empty());
            EXPECT_TRUE(csr.values.empty());
        }

        /** @brief Tests that a matrix without rows has one offset. */
        TEST_P(Small_SparseAssemblyBase, AMatrixWithoutRowsHasOneOffset)
        {
            const int64_t first = Core::first_index(GetParam());

            const HostCsr<int64_t, double> csr =
                assemble_csr<int64_t, double>(0, 0, first, {}, {}, {});

            EXPECT_EQ(csr.row_offsets, std::vector<int64_t>(1, first));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseAssemblyBase,
                                 ::testing::Values(IndexBase::Zero, IndexBase::One));

        /** @brief Tests that the offsets of a matrix start at zero and end at the stored count. */
        TEST(Small_SparseAssembly, TheOffsetsFrameTheStoredValues)
        {
            const HostTriplets<float> host = ordered_matrix();

            const HostCsr<int32_t, float> csr = assemble<int32_t, float>(host, IndexBase::Zero);

            ASSERT_EQ(csr.row_offsets.size(), static_cast<std::size_t>(host.rows) + 1);
            EXPECT_EQ(csr.row_offsets.front(), 0);
            EXPECT_EQ(static_cast<std::size_t>(csr.row_offsets.back()), csr.values.size());
        }

        /** @brief Tests that the columns of every row ascend strictly. */
        TEST(Small_SparseAssembly, EveryRowAscendsStrictly)
        {
            const HostCsr<int32_t, float> csr =
                assemble<int32_t, float>(ordered_matrix(), IndexBase::Zero);

            EXPECT_TRUE(rows_ascend_strictly(csr));
        }

        /** @brief Tests that every stored entry holds the sum of the triplets on its pair. */
        TEST(Small_SparseAssembly, EveryEntryHoldsTheSumOfItsTriplets)
        {
            const HostTriplets<float> host = ordered_matrix();

            const HostCsr<int32_t, float> csr = assemble<int32_t, float>(host, IndexBase::Zero);

            // Sums of integers in [-3, 3] are exact, so the dense image has one value per pair.
            EXPECT_EQ(dense_of(csr, static_cast<std::size_t>(host.rows),
                               static_cast<std::size_t>(host.cols)),
                      dense_of(host));
        }

        /** @brief Tests that every pair the triplets name is stored once. */
        TEST(Small_SparseAssembly, EveryPairIsStoredOnce)
        {
            const HostTriplets<float> host = ordered_matrix();

            const HostCsr<int32_t, float> csr = assemble<int32_t, float>(host, IndexBase::Zero);

            EXPECT_EQ(csr.values.size(), distinct_pairs_of(host));
        }

        /** @brief Tests that a negative extent is refused. */
        TEST(Small_SparseAssembly, ANegativeExtentThrows)
        {
            EXPECT_THROW(assemble_and_drop({"Rows", -1, 5, 0, {}, {}, {}}), std::runtime_error);
            EXPECT_THROW(assemble_and_drop({"Columns", 4, -1, 0, {}, {}, {}}), std::runtime_error);
        }

        /** @brief Tests that triplet arrays of different lengths are refused. */
        TEST(Small_SparseAssembly, TripletArraysOfDifferentLengthsThrow)
        {
            EXPECT_THROW(assemble_and_drop({"Columns", 4, 5, 0, {0, 1}, {0}, {1.0f, 2.0f}}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop({"Values", 4, 5, 0, {0, 1}, {0, 1}, {1.0f}}),
                         std::runtime_error);
        }

        /** @brief Tests that a triplet outside the matrix is refused in both bases. */
        TEST(Small_SparseAssembly, ATripletOutsideTheMatrixThrows)
        {
            EXPECT_THROW(assemble_and_drop({"Row", 4, 5, 0, {4}, {0}, {1.0f}}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop({"Column", 4, 5, 0, {0}, {5}, {1.0f}}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop({"RowOneBased", 4, 5, 1, {5}, {1}, {1.0f}}),
                         std::runtime_error);
            EXPECT_THROW(assemble_and_drop({"ColumnOneBased", 4, 5, 1, {1}, {0}, {1.0f}}),
                         std::runtime_error);
        }

        /** @brief Tests that a column count past the index type is refused. */
        TEST(Small_SparseAssembly, AColumnCountPastTheIndexTypeThrows)
        {
            EXPECT_THROW(assemble_and_drop({"Columns", 1, K_COLS_PAST_INT32, 0, {}, {}, {}}),
                         std::runtime_error);
        }

        /** @brief Tests that distinct positions inside the vector are accepted. */
        TEST(Small_SparseAssembly, DistinctPositionsInsideTheVectorAreAccepted)
        {
            EXPECT_NO_THROW(require_distinct_positions<int32_t>(8, 0, {6, 1, 4}));
            EXPECT_NO_THROW(require_distinct_positions<int64_t>(8, 1, {7, 2, 5}));
        }

        /** @brief Tests that a repeated position is refused. */
        TEST(Small_SparseAssembly, ARepeatedPositionThrows)
        {
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 0, {6, 1, 6}),
                         std::runtime_error);
        }

        /** @brief Tests that a position outside the vector is refused in both bases. */
        TEST(Small_SparseAssembly, APositionOutsideTheVectorThrows)
        {
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 0, {6, 8, 4}),
                         std::runtime_error);
            EXPECT_THROW(require_distinct_positions<int32_t>(8, 1, {7, 0, 5}),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
