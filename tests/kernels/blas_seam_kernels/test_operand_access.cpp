/**************************************************************************/
/* test_operand_access.cpp                                                */
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
 * @file test_operand_access.cpp
 * @brief Tests OperandAccess: the strides of each layout and transpose, and the read it gives.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Leading dimension of every access resolved in this file.
            constexpr int64_t K_LEADING_DIMENSION = 7;

            /// @brief Rows of the matrix the read tests store.
            constexpr int64_t K_STORED_ROWS = 2;

            /// @brief Columns of the matrix the read tests store.
            constexpr int64_t K_STORED_COLS = 3;

            /// @brief The matrix [[1, 2, 3], [4, 5, 6]] read row by row.
            constexpr std::array<float, 6> K_ROWS_OF_THE_MATRIX = {1.0f, 2.0f, 3.0f,
                                                                   4.0f, 5.0f, 6.0f};

            /// @brief The transpose of that matrix, [[1, 4], [2, 5], [3, 6]], read row by row.
            constexpr std::array<float, 6> K_ROWS_OF_THE_TRANSPOSE = {1.0f, 4.0f, 2.0f,
                                                                      5.0f, 3.0f, 6.0f};

            /** @brief Holds one layout and a transpose with the strides they must give. */
            struct StrideRow
            {
                const char* name;           /**< The suffix of the test name. */
                Core::Layout layout;        /**< The storage order of the matrix. */
                Core::Transpose transpose;  /**< The transpose applied to it. */
                int64_t row;                /**< The expected step of a row. */
                int64_t col;                /**< The expected step of a column. */
            };

            /** @brief Returns the four stride rows of the layouts and the two plain transposes. */
            [[nodiscard]] std::array<StrideRow, 4> stride_rows()
            {
                return {{{"RowMajorUntransposed", Core::Layout::RowMajor,
                          Core::Transpose::NoTrans, K_LEADING_DIMENSION, 1},
                         {"RowMajorTransposed", Core::Layout::RowMajor,
                          Core::Transpose::Trans, 1, K_LEADING_DIMENSION},
                         {"ColumnMajorUntransposed", Core::Layout::ColumnMajor,
                          Core::Transpose::NoTrans, 1, K_LEADING_DIMENSION},
                         {"ColumnMajorTransposed", Core::Layout::ColumnMajor,
                          Core::Transpose::Trans, K_LEADING_DIMENSION, 1}}};
            }

            /** @brief Names a stride row for the test name. */
            std::string stride_row_name(const ::testing::TestParamInfo<StrideRow>& info)
            {
                return info.param.name;
            }

            /** @brief Returns the matrix [[1, 2, 3], [4, 5, 6]] stored in the layout. */
            [[nodiscard]] std::array<float, 6> stored_matrix(Core::Layout layout)
            {
                return layout == Core::Layout::RowMajor ? K_ROWS_OF_THE_MATRIX
                                                        : K_ROWS_OF_THE_TRANSPOSE;
            }

            /** @brief Returns the leading dimension of the stored matrix in the layout. */
            [[nodiscard]] int64_t stored_leading_dimension(Core::Layout layout)
            {
                return layout == Core::Layout::RowMajor ? K_STORED_COLS : K_STORED_ROWS;
            }

            /** @brief Reads a rows by cols matrix through an access, row by row. */
            [[nodiscard]] std::array<float, 6> read_rows(const Kernels::OperandAccess& access,
                                                         const std::array<float, 6>& stored,
                                                         int64_t rows, int64_t cols)
            {
                std::array<float, 6> read{};
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t c = 0; c < cols; ++c)
                        read[static_cast<std::size_t>(r * cols + c)] =
                            access.at(stored.data(), r, c);
                return read;
            }

            /** @brief Names a layout for the test name. */
            std::string layout_name(const ::testing::TestParamInfo<Core::Layout>& info)
            {
                return info.param == Core::Layout::RowMajor ? "RowMajor" : "ColumnMajor";
            }
        } // namespace

        /** @brief Resolves the access of one layout and transpose at a time. */
        class Small_OperandAccessStrides : public ::testing::TestWithParam<StrideRow>
        {

        };

        TEST_P(Small_OperandAccessStrides, StepARowAndAColumnByTheirStrides)
        {
            const StrideRow row = GetParam();

            const Kernels::OperandAccess access =
                Kernels::operand_access(row.layout, row.transpose, K_LEADING_DIMENSION);

            EXPECT_EQ(access.row, row.row);
            EXPECT_EQ(access.col, row.col);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_OperandAccessStrides, ::testing::ValuesIn(stride_rows()),
                                 stride_row_name);

        /** @brief Resolves and reads an access of one layout at a time. */
        class Small_OperandAccessLayouts : public ::testing::TestWithParam<Core::Layout>
        {

        };

        TEST_P(Small_OperandAccessLayouts, ConjugateTransposeHasTheStridesOfTransposeAndConjugates)
        {
            const Kernels::OperandAccess transposed = Kernels::operand_access(
                GetParam(), Core::Transpose::Trans, K_LEADING_DIMENSION);
            const Kernels::OperandAccess conjugated = Kernels::operand_access(
                GetParam(), Core::Transpose::ConjTrans, K_LEADING_DIMENSION);

            EXPECT_EQ(conjugated.row, transposed.row);
            EXPECT_EQ(conjugated.col, transposed.col);
            EXPECT_TRUE(conjugated.conjugated);
            EXPECT_FALSE(transposed.conjugated);
        }

        TEST_P(Small_OperandAccessLayouts, UntransposedAccessDoesNotConjugate)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                GetParam(), Core::Transpose::NoTrans, K_LEADING_DIMENSION);

            EXPECT_FALSE(access.conjugated);
        }

        TEST_P(Small_OperandAccessLayouts, AtReadsTheSameElementFromBothLayouts)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                GetParam(), Core::Transpose::NoTrans, stored_leading_dimension(GetParam()));

            EXPECT_EQ(read_rows(access, stored_matrix(GetParam()), K_STORED_ROWS, K_STORED_COLS),
                      K_ROWS_OF_THE_MATRIX);
        }

        TEST_P(Small_OperandAccessLayouts, AtReadsTheTransposeFromBothLayouts)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                GetParam(), Core::Transpose::Trans, stored_leading_dimension(GetParam()));

            EXPECT_EQ(read_rows(access, stored_matrix(GetParam()), K_STORED_COLS, K_STORED_ROWS),
                      K_ROWS_OF_THE_TRANSPOSE);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_OperandAccessLayouts,
                                 ::testing::Values(Core::Layout::RowMajor,
                                                   Core::Layout::ColumnMajor),
                                 layout_name);

        TEST(Small_OperandAccess, IndexCombinesBothStrides)
        {
            const Kernels::OperandAccess access{K_LEADING_DIMENSION, 1, false};

            EXPECT_EQ(access.index(0, 0), 0);
            EXPECT_EQ(access.index(2, 3), 2 * K_LEADING_DIMENSION + 3);
            EXPECT_EQ(access.transposed().index(2, 3), 2 + 3 * K_LEADING_DIMENSION);
        }

        TEST(Small_OperandAccess, AtConjugatesAComplexElement)
        {
            const std::array<std::complex<double>, 2> elements = {
                std::complex<double>(1.0, 2.0), std::complex<double>(3.0, -4.0)};
            const Kernels::OperandAccess plain{1, 1, false};
            const Kernels::OperandAccess conjugating{1, 1, true};

            EXPECT_EQ(plain.at(elements.data(), 0, 1), std::complex<double>(3.0, -4.0));
            EXPECT_EQ(conjugating.at(elements.data(), 0, 1), std::complex<double>(3.0, 4.0));
            EXPECT_EQ(conjugating.at(elements.data(), 0, 0), std::complex<double>(1.0, -2.0));
        }

        TEST(Small_OperandAccess, AtLeavesARealElementUnderAConjugatingAccess)
        {
            const std::array<double, 2> elements = {1.5, -2.5};
            const Kernels::OperandAccess conjugating{1, 1, true};

            EXPECT_EQ(conjugating.at(elements.data(), 0, 0), 1.5);
            EXPECT_EQ(conjugating.at(elements.data(), 0, 1), -2.5);
        }

        TEST(Small_OperandAccess, TransposedSwapsTheStridesAndKeepsTheConjugation)
        {
            const Kernels::OperandAccess access{K_LEADING_DIMENSION, 1, true};
            const Kernels::OperandAccess swapped = access.transposed();

            EXPECT_EQ(swapped.row, 1);
            EXPECT_EQ(swapped.col, K_LEADING_DIMENSION);
            EXPECT_TRUE(swapped.conjugated);
            EXPECT_FALSE((Kernels::OperandAccess{K_LEADING_DIMENSION, 1, false})
                             .transposed().conjugated);
        }
    } // namespace Test
} // namespace SushiBLAS
