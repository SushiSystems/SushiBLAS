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
 * @file test_structure.cpp
 * @brief Tests find_defect on the arrays of a CSR matrix and on the positions of a sparse vector.
 * @author Mustafa Garip
 */

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

            /** @brief Expects the sample matrix stored in Index to keep every rule under base. */
            template<typename Index>
            void expect_sample_has_no_defect(IndexBase base)
            {
                const std::vector<Index> offsets =
                    stored_indices<Index>(sample_row_offsets(), base);
                const std::vector<Index> columns = stored_indices<Index>(sample_columns(), base);

                EXPECT_EQ(find_defect(sample_pattern(offsets, columns, base)),
                          StructureDefect::None)
                    << "base " << Core::to_string(base);
            }
        } // namespace

        TEST(Unit_SparseStructure, TheSampleKeepsEveryRule)
        {
            for (const IndexBase base : K_BOTH_BASES)
            {
                expect_sample_has_no_defect<int32_t>(base);
                expect_sample_has_no_defect<int64_t>(base);
            }
        }

        TEST(Unit_SparseStructure, AFirstOffsetOtherThanTheBaseIsReported)
        {
            EXPECT_EQ(defect_of({1, 2, 2, 5, 7}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::FirstOffset);
        }

        TEST(Unit_SparseStructure, ARowThatEndsBeforeItStartsIsReported)
        {
            EXPECT_EQ(defect_of({0, 3, 2, 5, 7}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::DecreasingOffset);
        }

        TEST(Unit_SparseStructure, ALastOffsetOtherThanTheEntryCountIsReported)
        {
            EXPECT_EQ(defect_of({0, 2, 2, 5, 6}, {0, 3, 1, 2, 4, 0, 4}),
                      StructureDefect::LastOffset);
        }

        TEST(Unit_SparseStructure, AColumnPastTheLastIsReported)
        {
            EXPECT_EQ(defect_of({0, 2, 2, 5, 7}, {0, 3, 1, 2, 5, 0, 4}),
                      StructureDefect::IndexOutOfRange);
        }

        TEST(Unit_SparseStructure, AColumnBelowTheBaseIsReported)
        {
            const std::vector<int32_t> offsets = {1, 3, 3, 6, 8};
            const std::vector<int32_t> columns = {1, 4, 2, 3, 5, 0, 5};

            EXPECT_EQ(find_defect(sample_pattern(offsets, columns, IndexBase::One)),
                      StructureDefect::IndexOutOfRange);
        }

        TEST(Unit_SparseStructure, UnsortedColumnsAreNotADefect)
        {
            EXPECT_EQ(defect_of({0, 2, 2, 5, 7}, {3, 0, 4, 1, 2, 4, 0}), StructureDefect::None);
        }

        TEST(Unit_SparseStructure, ARepeatedPairIsNotADefect)
        {
            EXPECT_EQ(defect_of({0, 2, 2, 5, 7}, {3, 3, 1, 2, 2, 0, 4}), StructureDefect::None);
        }

        TEST(Unit_SparseStructure, DistinctPositionsInsideTheVectorKeepEveryRule)
        {
            EXPECT_EQ(defect_of_positions({6, 1, 4}, IndexBase::Zero), StructureDefect::None);
            EXPECT_EQ(defect_of_positions({7, 2, 5}, IndexBase::One), StructureDefect::None);
            EXPECT_EQ(defect_of_positions({}, IndexBase::Zero), StructureDefect::None);
        }

        TEST(Unit_SparseStructure, APositionOutsideTheVectorIsReported)
        {
            EXPECT_EQ(defect_of_positions({6, 8, 4}, IndexBase::Zero),
                      StructureDefect::IndexOutOfRange);
            EXPECT_EQ(defect_of_positions({7, 0, 5}, IndexBase::One),
                      StructureDefect::IndexOutOfRange);
        }

        TEST(Unit_SparseStructure, APositionStoredTwiceIsReported)
        {
            EXPECT_EQ(defect_of_positions({6, 1, 6}, IndexBase::Zero),
                      StructureDefect::RepeatedIndex);
        }
    } // namespace Test
} // namespace SushiBLAS
