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
 * @file test_matrix_operand.cpp
 * @brief Tests how a tensor is read as a matrix operand and the checks between two of them.
 * @author Mustafa Garip
 */

#include <string>
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
        }

        TEST_F(Medium_MatrixOperand, TheBatchStrideIsTheMatrixSizeAndRefusesAWrongCount)
        {
            reinit_engine(Core::Layout::ColumnMajor);
            const Dispatch::MatrixOperand m =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 3, 4, 5}), "op", "a");

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

        TEST_F(Medium_MatrixOperand, TwoOperandsOfDifferentLayoutsAreRefused)
        {
            const Dispatch::MatrixOperand row =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 2}), "op", "a");
            reinit_engine(Core::Layout::ColumnMajor);
            const Dispatch::MatrixOperand column =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 2}), "op", "b");

            EXPECT_THROW(Dispatch::require_same_layout(row, column, "op", "a", "b"),
                         std::invalid_argument);
        }

        TEST_F(Medium_MatrixOperand, AnExtentMismatchIsRefused)
        {
            EXPECT_THROW(Dispatch::require_matrix_extent(3, 2, "rows", "op", "a"),
                         std::invalid_argument);
        }

        TEST_F(Medium_MatrixOperand, ARankTwoOperandIsUnbatchedAndABatchIsNot)
        {
            const Dispatch::MatrixOperand single =
                Dispatch::describe_matrix_operand(engine->create_tensor({2, 2}), "op", "a");
            const Dispatch::MatrixOperand batch =
                Dispatch::describe_matrix_operand(engine->create_tensor({3, 2, 2}), "op", "b");

            EXPECT_NO_THROW(Dispatch::require_unbatched(single, "op", "a"));
            EXPECT_THROW(Dispatch::require_unbatched(batch, "op", "b"), std::invalid_argument);
        }

        namespace
        {
            /** @brief Returns the text of the std::invalid_argument a check throws, or "". */
            template<typename Check>
            std::string message_of(const Check& check)
            {
                try
                {
                    check();
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }

                return std::string();
            }
        } // namespace

        TEST(Small_MatrixOperand, ANonSquareMatrixIsRefusedAndNamesItsExtents)
        {
            Dispatch::MatrixOperand rectangle;
            rectangle.rows = 2;
            rectangle.cols = 3;

            EXPECT_EQ(message_of([&] { Dispatch::require_square(rectangle, "op", "a"); }),
                      "op: operand 'a' is 2 x 3; the operation needs a square matrix");
        }

        TEST(Small_MatrixOperand, ASquareMatrixIsAccepted)
        {
            Dispatch::MatrixOperand square;
            square.rows = 3;
            square.cols = 3;

            EXPECT_EQ(message_of([&] { Dispatch::require_square(square, "op", "a"); }), "");
        }

        TEST_F(Medium_MatrixOperand, BatchesWithDifferentAxesAreRefused)
        {
            const Tensor base = engine->create_tensor({2, 3, 4});
            const Tensor other_size = engine->create_tensor({3, 3, 4});
            const Tensor other_count = engine->create_tensor({1, 2, 3, 4});
            const Dispatch::MatrixOperand mb = Dispatch::describe_matrix_operand(base, "op", "a");
            const Dispatch::MatrixOperand ms = Dispatch::describe_matrix_operand(other_size, "op", "b");
            const Dispatch::MatrixOperand mc = Dispatch::describe_matrix_operand(other_count, "op", "b");

            EXPECT_THROW(Dispatch::require_same_batch_axes(base, mb, other_size, ms, "op", "a", "b"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_same_batch_axes(base, mb, other_count, mc, "op", "a", "b"),
                         std::invalid_argument);
        }

        TEST_F(Medium_MatrixOperand, BatchesWithTheSameAxesAreAccepted)
        {
            const Tensor a = engine->create_tensor({2, 3, 4});
            const Tensor b = engine->create_tensor({2, 4, 5});
            const Dispatch::MatrixOperand ma = Dispatch::describe_matrix_operand(a, "op", "a");
            const Dispatch::MatrixOperand mb = Dispatch::describe_matrix_operand(b, "op", "b");

            EXPECT_EQ(message_of([&] {
                          Dispatch::require_same_batch_axes(a, ma, b, mb, "op", "a", "b");
                      }),
                      "");
        }
    } // namespace Test
} // namespace SushiBLAS
