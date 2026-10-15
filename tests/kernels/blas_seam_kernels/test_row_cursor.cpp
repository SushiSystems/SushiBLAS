/**************************************************************************/
/* test_row_cursor.cpp                                                    */
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
 * @file test_row_cursor.cpp
 * @brief Tests that a RowCursor steps by a growing stride and retreats along the same path.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_RowCursor, AFlatCursorStepsByAConstantStride)
        {
            Kernels::RowCursor cursor{5, 3, 0};

            cursor.advance();
            cursor.advance();

            EXPECT_EQ(cursor.offset, int64_t{11});
            EXPECT_EQ(cursor.step, int64_t{3});
        }

        TEST(Unit_RowCursor, ASlopeGrowsTheStepAtEachColumn)
        {
            Kernels::RowCursor cursor{0, 1, 1};

            cursor.advance();
            cursor.advance();
            cursor.advance();

            EXPECT_EQ(cursor.offset, int64_t{6});
            EXPECT_EQ(cursor.step, int64_t{4});
        }

        TEST(Unit_RowCursor, RetreatUndoesAdvance)
        {
            Kernels::RowCursor cursor{2, 4, -1};

            cursor.advance();
            cursor.advance();
            cursor.retreat();
            cursor.retreat();

            EXPECT_EQ(cursor.offset, int64_t{2});
            EXPECT_EQ(cursor.step, int64_t{4});
        }
    } // namespace Test
} // namespace SushiBLAS
