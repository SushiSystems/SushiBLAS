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
 * @file test_operand_access.cpp
 * @brief Tests OperandAccess: the strides of each layout and transpose, and the read it gives.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Leading dimension of every access resolved in this file.
        constexpr int64_t K_LEADING_DIMENSION = 7;

        /// @brief Rows of the matrix the read tests store.
        constexpr int64_t K_STORED_ROWS = 2;

        /// @brief Columns of the matrix the read tests store.
        constexpr int64_t K_STORED_COLS = 3;

        /// @brief The matrix [[1, 2, 3], [4, 5, 6]] in row-major order.
        constexpr std::array<float, 6> K_ROW_MAJOR_MATRIX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        /// @brief The same matrix in column-major order.
        constexpr std::array<float, 6> K_COLUMN_MAJOR_MATRIX = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

        TEST(Unit_OperandAccess, RowMajorUntransposedStepsARowByTheLeadingDimension)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                Core::Layout::RowMajor, Core::Transpose::NoTrans, K_LEADING_DIMENSION);

            EXPECT_EQ(access.row, K_LEADING_DIMENSION);
            EXPECT_EQ(access.col, 1);
        }

        TEST(Unit_OperandAccess, RowMajorTransposedStepsAColumnByTheLeadingDimension)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                Core::Layout::RowMajor, Core::Transpose::Trans, K_LEADING_DIMENSION);

            EXPECT_EQ(access.row, 1);
            EXPECT_EQ(access.col, K_LEADING_DIMENSION);
        }

        TEST(Unit_OperandAccess, ColumnMajorUntransposedStepsAColumnByTheLeadingDimension)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                Core::Layout::ColumnMajor, Core::Transpose::NoTrans, K_LEADING_DIMENSION);

            EXPECT_EQ(access.row, 1);
            EXPECT_EQ(access.col, K_LEADING_DIMENSION);
        }

        TEST(Unit_OperandAccess, ColumnMajorTransposedStepsARowByTheLeadingDimension)
        {
            const Kernels::OperandAccess access = Kernels::operand_access(
                Core::Layout::ColumnMajor, Core::Transpose::Trans, K_LEADING_DIMENSION);

            EXPECT_EQ(access.row, K_LEADING_DIMENSION);
            EXPECT_EQ(access.col, 1);
        }

        TEST(Unit_OperandAccess, ConjugateTransposeHasTheStridesOfTransposeAndConjugates)
        {
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                const Kernels::OperandAccess transposed = Kernels::operand_access(
                    layout, Core::Transpose::Trans, K_LEADING_DIMENSION);
                const Kernels::OperandAccess conjugated = Kernels::operand_access(
                    layout, Core::Transpose::ConjTrans, K_LEADING_DIMENSION);

                EXPECT_EQ(conjugated.row, transposed.row);
                EXPECT_EQ(conjugated.col, transposed.col);
                EXPECT_TRUE(conjugated.conjugated);
                EXPECT_FALSE(transposed.conjugated);
            }
        }

        TEST(Unit_OperandAccess, UntransposedAccessDoesNotConjugate)
        {
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                const Kernels::OperandAccess access = Kernels::operand_access(
                    layout, Core::Transpose::NoTrans, K_LEADING_DIMENSION);

                EXPECT_FALSE(access.conjugated);
            }
        }

        TEST(Unit_OperandAccess, IndexCombinesBothStrides)
        {
            const Kernels::OperandAccess access{K_LEADING_DIMENSION, 1, false};

            EXPECT_EQ(access.index(0, 0), 0);
            EXPECT_EQ(access.index(2, 3), 2 * K_LEADING_DIMENSION + 3);
            EXPECT_EQ(access.transposed().index(2, 3), 2 + 3 * K_LEADING_DIMENSION);
        }

        TEST(Unit_OperandAccess, AtReadsTheSameElementFromBothLayouts)
        {
            const Kernels::OperandAccess row_major = Kernels::operand_access(
                Core::Layout::RowMajor, Core::Transpose::NoTrans, K_STORED_COLS);
            const Kernels::OperandAccess column_major = Kernels::operand_access(
                Core::Layout::ColumnMajor, Core::Transpose::NoTrans, K_STORED_ROWS);

            for (int64_t r = 0; r < K_STORED_ROWS; ++r)
            {
                for (int64_t c = 0; c < K_STORED_COLS; ++c)
                {
                    const float expected = static_cast<float>(r * K_STORED_COLS + c + 1);

                    EXPECT_EQ(row_major.at(K_ROW_MAJOR_MATRIX.data(), r, c), expected);
                    EXPECT_EQ(column_major.at(K_COLUMN_MAJOR_MATRIX.data(), r, c), expected);
                }
            }
        }

        TEST(Unit_OperandAccess, AtReadsTheTransposeFromBothLayouts)
        {
            const Kernels::OperandAccess row_major = Kernels::operand_access(
                Core::Layout::RowMajor, Core::Transpose::Trans, K_STORED_COLS);
            const Kernels::OperandAccess column_major = Kernels::operand_access(
                Core::Layout::ColumnMajor, Core::Transpose::Trans, K_STORED_ROWS);

            for (int64_t r = 0; r < K_STORED_COLS; ++r)
            {
                for (int64_t c = 0; c < K_STORED_ROWS; ++c)
                {
                    const float expected = static_cast<float>(c * K_STORED_COLS + r + 1);

                    EXPECT_EQ(row_major.at(K_ROW_MAJOR_MATRIX.data(), r, c), expected);
                    EXPECT_EQ(column_major.at(K_COLUMN_MAJOR_MATRIX.data(), r, c), expected);
                }
            }
        }

        TEST(Unit_OperandAccess, AtConjugatesAComplexElement)
        {
            const std::array<std::complex<double>, 2> elements = {
                std::complex<double>(1.0, 2.0), std::complex<double>(3.0, -4.0)};
            const Kernels::OperandAccess plain{1, 1, false};
            const Kernels::OperandAccess conjugating{1, 1, true};

            EXPECT_EQ(plain.at(elements.data(), 0, 1), std::complex<double>(3.0, -4.0));
            EXPECT_EQ(conjugating.at(elements.data(), 0, 1), std::complex<double>(3.0, 4.0));
            EXPECT_EQ(conjugating.at(elements.data(), 0, 0), std::complex<double>(1.0, -2.0));
        }

        TEST(Unit_OperandAccess, AtLeavesARealElementUnderAConjugatingAccess)
        {
            const std::array<double, 2> elements = {1.5, -2.5};
            const Kernels::OperandAccess conjugating{1, 1, true};

            EXPECT_EQ(conjugating.at(elements.data(), 0, 0), 1.5);
            EXPECT_EQ(conjugating.at(elements.data(), 0, 1), -2.5);
        }

        TEST(Unit_OperandAccess, TransposedSwapsTheStridesAndKeepsTheConjugation)
        {
            const Kernels::OperandAccess access{K_LEADING_DIMENSION, 1, true};
            const Kernels::OperandAccess swapped = access.transposed();

            EXPECT_EQ(swapped.row, 1);
            EXPECT_EQ(swapped.col, K_LEADING_DIMENSION);
            EXPECT_TRUE(swapped.conjugated);
            EXPECT_FALSE((Kernels::OperandAccess{K_LEADING_DIMENSION, 1, false})
                             .transposed().conjugated);
        }

        TEST(Unit_OperandAccess, ResolvesAtCompileTime)
        {
            constexpr Kernels::OperandAccess access = Kernels::operand_access(
                Core::Layout::RowMajor, Core::Transpose::ConjTrans, K_LEADING_DIMENSION);

            static_assert(access.row == 1, "A row-major transpose steps a row by one.");
            static_assert(access.col == K_LEADING_DIMENSION, "It steps a column by the leading "
                                                             "dimension.");
            static_assert(access.conjugated, "A conjugate transpose conjugates.");
            static_assert(access.index(1, 1) == K_LEADING_DIMENSION + 1, "index is constexpr.");
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
