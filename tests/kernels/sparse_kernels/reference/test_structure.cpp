/**************************************************************************/
/* test_structure.cpp                                                     */
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
 * @file test_structure.cpp
 * @brief Tests find_defect on the arrays of a CSR matrix and on the positions of a sparse vector.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiBLAS/sparse_kernels/reference/structure.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::IndexBase;
            using Kernels::Reference::find_defect;
            using Kernels::Reference::StructureDefect;

            /// @brief Length of the dense vector the position lists of this file address.
            constexpr int64_t K_VECTOR_EXTENT = 8;

            /** @brief Returns the pattern of a 4 x 5 matrix over host arrays stored with base. */
            template<typename Index>
            [[nodiscard]] Kernels::CsrPattern<Index> sample_pattern(
                const std::vector<Index>& offsets, const std::vector<Index>& columns,
                IndexBase base)
            {
                return {4, 5, static_cast<int64_t>(columns.size()), Core::first_index(base),
                        offsets.data(), columns.data()};
            }

            /** @brief Returns the defect of a 4 x 5 matrix given by zero-based arrays. */
            [[nodiscard]] StructureDefect defect_of(const std::vector<int32_t>& offsets,
                                                    const std::vector<int32_t>& columns)
            {
                return find_defect(sample_pattern(offsets, columns, IndexBase::Zero));
            }

            /** @brief Returns the defect of stored positions in a vector of K_VECTOR_EXTENT. */
            [[nodiscard]] StructureDefect defect_of_positions(const std::vector<int32_t>& stored,
                                                              IndexBase base)
            {
                const Kernels::IndexList<int32_t> positions{static_cast<int64_t>(stored.size()),
                                                            K_VECTOR_EXTENT,
                                                            Core::first_index(base), stored.data()};
                return find_defect(positions);
            }

            /** @brief Returns the defect of the sample matrix stored in Index under base. */
            template<typename Index>
            StructureDefect defect_of_sample(IndexBase base)
            {
                const std::vector<Index> offsets =
                    stored_indices<Index>(sample_row_offsets(), base);
                const std::vector<Index> columns = stored_indices<Index>(sample_columns(), base);

                return find_defect(sample_pattern(offsets, columns, base));
            }

            /** @brief One base and index type of the sample matrix. */
            struct SampleRow
            {
                    const char* name;                 /**< The row's suffix. */
                    IndexBase base;                   /**< The index base the arrays are stored in. */
                    StructureDefect (*check)(IndexBase); /**< Returns the defect of the sample stored in the row's type. */
            };

            /** @brief Returns the four combinations of base and index type. */
            std::vector<SampleRow> sample_rows()
            {
                return {{"ZeroBasedInt32", IndexBase::Zero, &defect_of_sample<int32_t>},
                        {"ZeroBasedInt64", IndexBase::Zero, &defect_of_sample<int64_t>},
                        {"OneBasedInt32", IndexBase::One, &defect_of_sample<int32_t>},
                        {"OneBasedInt64", IndexBase::One, &defect_of_sample<int64_t>}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string sample_row_name(const ::testing::TestParamInfo<SampleRow>& info)
            {
                return info.param.name;
            }

            /** @brief A 4 x 5 matrix given by its stored arrays and base. */
            struct ColumnRow
            {
                    const char* name;               /**< The row's suffix. */
                    std::vector<int32_t> offsets;   /**< The row offsets, in the row's base. */
                    std::vector<int32_t> columns;   /**< The column indices, in the row's base. */
                    IndexBase base;                 /**< The index base the arrays are stored in. */
            };

            /** @brief Returns a column at the number of columns, and one below the one base. */
            std::vector<ColumnRow> out_of_range_rows()
            {
                return {{"ColumnPastTheLast", {0, 2, 2, 5, 7}, {0, 3, 1, 2, 5, 0, 4}, IndexBase::Zero},
                        {"ColumnBelowTheBase", {1, 3, 3, 6, 8}, {1, 4, 2, 3, 5, 0, 5}, IndexBase::One}};
            }

            /** @brief Returns columns out of order and a column repeated within a row. */
            std::vector<ColumnRow> tolerated_order_rows()
            {
                return {{"UnsortedColumns", {0, 2, 2, 5, 7}, {3, 0, 4, 1, 2, 4, 0}, IndexBase::Zero},
                        {"RepeatedPair", {0, 2, 2, 5, 7}, {3, 3, 1, 2, 2, 0, 4}, IndexBase::Zero}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string column_row_name(const ::testing::TestParamInfo<ColumnRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the sample check one base and index type at a time. */
        class Small_SparseStructureSample : public ::testing::TestWithParam<SampleRow>
        {

        };

        TEST_P(Small_SparseStructureSample, TheSampleKeepsEveryRule)
        {
            EXPECT_EQ(GetParam().check(GetParam().base), StructureDefect::None);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseStructureSample, ::testing::ValuesIn(sample_rows()),
                                 sample_row_name);

        TEST(Small_SparseStructure, AFirstOffsetOtherThanTheBaseIsReported)
        {
            EXPECT_EQ(defect_of({1, 2, 2, 5, 7}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::FirstOffset);
        }

        TEST(Small_SparseStructure, ARowThatEndsBeforeItStartsIsReported)
        {
            EXPECT_EQ(defect_of({0, 3, 2, 5, 7}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::DecreasingOffset);
        }

        TEST(Small_SparseStructure, ALastOffsetOtherThanTheEntryCountIsReported)
        {
            EXPECT_EQ(defect_of({0, 2, 2, 5, 6}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::LastOffset);
        }

        /** @brief Runs the column range check one stored matrix at a time. */
        class Small_SparseStructureColumnRange : public ::testing::TestWithParam<ColumnRow>
        {

        };

        TEST_P(Small_SparseStructureColumnRange, AColumnOutsideTheRangeIsReported)
        {
            const ColumnRow& row = GetParam();

            EXPECT_EQ(find_defect(sample_pattern(row.offsets, row.columns, row.base)),
                      StructureDefect::IndexOutOfRange);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_SparseStructureColumnRange, ::testing::ValuesIn(out_of_range_rows()),
            column_row_name);

        /** @brief Runs the column order check one stored matrix at a time. */
        class Small_SparseStructureColumnOrder : public ::testing::TestWithParam<ColumnRow>
        {

        };

        TEST_P(Small_SparseStructureColumnOrder, UnsortedOrRepeatedColumnsAreNotADefect)
        {
            const ColumnRow& row = GetParam();

            EXPECT_EQ(find_defect(sample_pattern(row.offsets, row.columns, row.base)),
                      StructureDefect::None);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseStructureColumnOrder,
                                 ::testing::ValuesIn(tolerated_order_rows()), column_row_name);

        TEST(Small_SparseStructure, DistinctPositionsInsideTheVectorKeepEveryRule)
        {
            EXPECT_EQ(defect_of_positions({6, 1, 4}, IndexBase::Zero), StructureDefect::None);
            EXPECT_EQ(defect_of_positions({7, 2, 5}, IndexBase::One), StructureDefect::None);
            EXPECT_EQ(defect_of_positions({}, IndexBase::Zero), StructureDefect::None);
        }

        TEST(Small_SparseStructure, APositionOutsideTheVectorIsReported)
        {
            EXPECT_EQ(defect_of_positions({6, 8, 4}, IndexBase::Zero),
                      StructureDefect::IndexOutOfRange);
            EXPECT_EQ(defect_of_positions({7, 0, 5}, IndexBase::One),
                      StructureDefect::IndexOutOfRange);
        }

        TEST(Small_SparseStructure, APositionStoredTwiceIsReported)
        {
            EXPECT_EQ(defect_of_positions({6, 1, 6}, IndexBase::Zero),
                      StructureDefect::RepeatedIndex);
        }
    } // namespace Test
} // namespace SushiBLAS
