/**************************************************************************/
/* test_matrix_operand.cpp                                                */
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
 * @file test_matrix_operand.cpp
 * @brief Tests how a tensor is read as a matrix operand and the checks between two of them.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds matrices on the fixture's engine. */
        class Medium_MatrixOperand : public SushiBLASTest
        {

        };

        TEST_F(Medium_MatrixOperand, ARowMajorMatrixLeadsByItsColumns)
        {
            const Dispatch::MatrixOperand m =
                Dispatch::describe_matrix_operand(engine->create_tensor({3, 5}), "op", "a");

            EXPECT_EQ(m.rows, int64_t{3});
            EXPECT_EQ(m.cols, int64_t{5});
            EXPECT_EQ(m.leading_dimension, int64_t{5});
            EXPECT_EQ(m.batch_count, int64_t{1});
            EXPECT_EQ(m.matrix_elements, int64_t{15});
        }

        TEST_F(Medium_MatrixOperand, AColumnMajorBatchLeadsByItsRowsAndCountsItsMatrices)
        {
            reinit_engine(Core::Layout::ColumnMajor);
            const Dispatch::MatrixOperand m =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 3, 4, 5}), "op", "a");

            EXPECT_EQ(m.leading_dimension, int64_t{4});
            EXPECT_EQ(m.batch_count, int64_t{6});
            EXPECT_EQ(Dispatch::batch_stride(m, 6, "op", "a"), int64_t{20});
            EXPECT_THROW((void)Dispatch::batch_stride(m, 4, "op", "a"), std::invalid_argument);
        }

        TEST_F(Medium_MatrixOperand, AVectorOrAStridedViewIsNoMatrix)
        {
            const Tensor matrix = engine->create_tensor({4, 4});

            EXPECT_THROW((void)Dispatch::describe_matrix_operand(engine->create_tensor({4}), "op",
                                                                 "a"),
                         std::invalid_argument);
            EXPECT_THROW((void)Dispatch::describe_matrix_operand(matrix.slice(1, 0, 2), "op",
                                                                 "a"),
                         std::invalid_argument);
        }

        TEST_F(Medium_MatrixOperand, TwoOperandsMustShareLayoutAndExtents)
        {
            const Dispatch::MatrixOperand row =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 2}), "op", "a");
            reinit_engine(Core::Layout::ColumnMajor);
            const Dispatch::MatrixOperand column =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 2}), "op", "b");

            EXPECT_THROW(Dispatch::require_same_layout(row, column, "op", "a", "b"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_matrix_extent(3, 2, "rows", "op", "a"),
                         std::invalid_argument);
            EXPECT_NO_THROW(Dispatch::require_unbatched(row, "op", "a"));
        }
    } // namespace Test
} // namespace SushiBLAS
