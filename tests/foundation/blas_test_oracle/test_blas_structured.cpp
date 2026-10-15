/**************************************************************************/
/* test_blas_structured.cpp                                               */
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
 * @file test_blas_structured.cpp
 * @brief Tests the layout, band, triangle-band and packed conversions against hand-built storage.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The value a conversion writes into a slot that holds no element.
        constexpr int K_POISON = -1;

        /// @brief A 3 x 3 matrix given row by row, element (i, j) as 10 * (i + 1) + j + 1.
        const std::vector<int> K_SQUARE{11, 12, 13, 21, 22, 23, 31, 32, 33};

        TEST(Unit_BlasStructured, ColumnMajorTransposesTheElementOrder)
        {
            const std::vector<int> dense{1, 2, 3, 4, 5, 6};

            EXPECT_EQ(in_layout(dense, 2, 3, Core::Layout::RowMajor), dense);
            EXPECT_EQ(in_layout(dense, 2, 3, Core::Layout::ColumnMajor),
                      (std::vector<int>{1, 4, 2, 5, 3, 6}));
        }

        TEST(Unit_BlasStructured, ATridiagonalBandStoresOneDiagonalPerSlot)
        {
            const std::vector<int> row_band =
                to_band(K_SQUARE, 3, 3, 1, 1, 3, Core::Layout::RowMajor, K_POISON);
            const std::vector<int> column_band =
                to_band(K_SQUARE, 3, 3, 1, 1, 3, Core::Layout::ColumnMajor, K_POISON);

            EXPECT_EQ(row_band, (std::vector<int>{-1, 11, 12, 21, 22, 23, 32, 33, -1}));
            EXPECT_EQ(column_band, (std::vector<int>{-1, 11, 21, 12, 22, 32, 23, 33, -1}));
        }

        TEST(Unit_BlasStructured, ATriangleBandKeepsOneSideOfTheDiagonal)
        {
            const std::vector<int> upper = to_triangle_band(K_SQUARE, 3, Core::Uplo::Upper, 1, 2,
                                                            Core::Layout::ColumnMajor, K_POISON);
            const std::vector<int> lower = to_triangle_band(K_SQUARE, 3, Core::Uplo::Lower, 1, 2,
                                                            Core::Layout::ColumnMajor, K_POISON);

            EXPECT_EQ(upper, (std::vector<int>{-1, 11, 12, 22, 23, 33}));
            EXPECT_EQ(lower, (std::vector<int>{11, 21, 22, 32, 33, -1}));
        }

        TEST(Unit_BlasStructured, PackedStorageWalksTheTriangleInLayoutOrder)
        {
            EXPECT_EQ(to_packed(K_SQUARE, 3, Core::Uplo::Upper, Core::Layout::ColumnMajor),
                      (std::vector<int>{11, 12, 22, 13, 23, 33}));
            EXPECT_EQ(to_packed(K_SQUARE, 3, Core::Uplo::Upper, Core::Layout::RowMajor),
                      (std::vector<int>{11, 12, 13, 22, 23, 33}));
            EXPECT_EQ(to_packed(K_SQUARE, 3, Core::Uplo::Lower, Core::Layout::ColumnMajor),
                      (std::vector<int>{11, 21, 31, 22, 32, 33}));
            EXPECT_EQ(to_packed(K_SQUARE, 3, Core::Uplo::Lower, Core::Layout::RowMajor),
                      (std::vector<int>{11, 21, 22, 31, 32, 33}));
        }

        TEST(Unit_BlasStructured, ASpareSlotPerLineStaysPoisoned)
        {
            const std::vector<int> band =
                to_band(K_SQUARE, 3, 3, 0, 0, 2, Core::Layout::RowMajor, K_POISON);

            EXPECT_EQ(band, (std::vector<int>{11, -1, 22, -1, 33, -1}));
        }
    } // namespace Test
} // namespace SushiBLAS
