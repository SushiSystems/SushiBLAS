/**************************************************************************/
/* test_dense_oracle.cpp                                                  */
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
 * @file test_dense_oracle.cpp
 * @brief Tests the dense oracle loops against hand-computed matrices and vectors.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief A 2 x 3 matrix given row by row.
        const std::vector<double> K_WIDE{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

        /// @brief A 3 x 3 matrix given row by row.
        const std::vector<double> K_SQUARE{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

        TEST(Unit_DenseOracle, MatvecMultipliesTheMatrixOrItsTranspose)
        {
            EXPECT_EQ(dense_matvec(K_WIDE, 2, 3, false, 2.0, {1.0, 0.0, -1.0}, 1.0, {10.0, 20.0}),
                      (std::vector<double>{6.0, 16.0}));
            EXPECT_EQ(dense_matvec(K_WIDE, 2, 3, true, 1.0, {1.0, 1.0}, 0.0, {9.0, 9.0, 9.0}),
                      (std::vector<double>{5.0, 7.0, 9.0}));
        }

        TEST(Unit_DenseOracle, SymmetricFromMirrorsTheChosenTriangle)
        {
            EXPECT_EQ(symmetric_from(K_SQUARE, 3, true),
                      (std::vector<double>{1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0}));
            EXPECT_EQ(symmetric_from(K_SQUARE, 3, false),
                      (std::vector<double>{1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0}));
        }

        TEST(Unit_DenseOracle, TriangleOfZeroesTheOtherSideAndCanSetAUnitDiagonal)
        {
            EXPECT_EQ(triangle_of(K_SQUARE, 3, false, true),
                      (std::vector<double>{1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 7.0, 8.0, 1.0}));
            EXPECT_EQ(triangle_of(K_SQUARE, 3, true, false),
                      (std::vector<double>{1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0}));
        }

        TEST(Unit_DenseOracle, BandOfKeepsTheDiagonalsInsideTheBand)
        {
            EXPECT_EQ(band_of(K_SQUARE, 3, 3, 1, 0),
                      (std::vector<double>{1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, 9.0}));
        }

        TEST(Unit_DenseOracle, TriangleUpdateAddsTheRankOneOrRankTwoTerm)
        {
            const std::vector<double> zero(4, 0.0);

            EXPECT_EQ(triangle_update(zero, 2, true, 1.0, {1.0, 2.0}, {}, false),
                      (std::vector<double>{1.0, 2.0, 0.0, 4.0}));
            EXPECT_EQ(triangle_update(zero, 2, false, 2.0, {1.0, 2.0}, {3.0, 1.0}, true),
                      (std::vector<double>{12.0, 0.0, 14.0, 8.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
