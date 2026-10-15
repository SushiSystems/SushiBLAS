/**************************************************************************/
/* test_blas_flags.cpp                                                    */
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
 * @file test_blas_flags.cpp
 * @brief Tests that each flag helper maps true and false onto the two BLAS enumerators.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_BlasFlags, EachHelperMapsBothTruthValues)
        {
            EXPECT_EQ(transpose_of(true), Core::Transpose::Trans);
            EXPECT_EQ(transpose_of(false), Core::Transpose::NoTrans);
            EXPECT_EQ(uplo_of(true), Core::Uplo::Upper);
            EXPECT_EQ(uplo_of(false), Core::Uplo::Lower);
            EXPECT_EQ(diag_of(true), Core::Diag::Unit);
            EXPECT_EQ(diag_of(false), Core::Diag::NonUnit);
            EXPECT_EQ(side_of(true), Core::Side::Left);
            EXPECT_EQ(side_of(false), Core::Side::Right);
        }

        TEST(Unit_BlasFlags, TheHelpersAreConstantExpressions)
        {
            static_assert(transpose_of(true) == Core::Transpose::Trans, "Usable in a constant.");
            static_assert(side_of(false) == Core::Side::Right, "Usable in a constant.");

            EXPECT_TRUE(Core::is_valid(uplo_of(true)));
        }
    } // namespace Test
} // namespace SushiBLAS
