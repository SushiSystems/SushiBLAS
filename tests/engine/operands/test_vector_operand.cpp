/**************************************************************************/
/* test_vector_operand.cpp                                                */
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
 * @file test_vector_operand.cpp
 * @brief Tests how a tensor is read as a vector operand, strided ones included, and its checks.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds vectors on the fixture's engine. */
        class Medium_VectorOperand : public SushiBLASTest
        {

        };

        TEST_F(Medium_VectorOperand, AContiguousTensorIsAUnitStrideVectorOfAllItsElements)
        {
            const Dispatch::VectorOperand v =
                Dispatch::describe_vector_operand(engine->create_tensor({2, 3}), "op", "x");

            EXPECT_EQ(v.length, int64_t{6});
            EXPECT_EQ(v.increment, int64_t{1});
        }

        TEST_F(Medium_VectorOperand, AStridedViewAboveRankOneIsRefused)
        {
            const Tensor column = engine->create_tensor({4, 3}).slice(1, 1, 2);

            EXPECT_THROW((void)Dispatch::describe_vector_operand(column, "op", "x"),
                         std::invalid_argument);
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

        TEST(Small_VectorOperand, ALengthMismatchIsRefusedAndNamesItsOperand)
        {
            const Dispatch::VectorOperand v{5, 2};

            EXPECT_EQ(message_of([&] {
                          Dispatch::require_vector_length(v, 4, "A's columns", "op", "x");
                      }),
                      "op: operand 'x' holds 5 elements, A's columns need 4");
        }

        TEST(Small_VectorOperand, AMatchingLengthIsAccepted)
        {
            const Dispatch::VectorOperand v{5, 2};

            EXPECT_EQ(message_of([&] {
                          Dispatch::require_vector_length(v, 5, "A's columns", "op", "x");
                      }),
                      "");
        }

        TEST(Small_VectorOperand, AnIncrementOtherThanOneIsRefused)
        {
            const Dispatch::VectorOperand v{5, 2};

            EXPECT_THROW(Dispatch::require_unit_increment(v, "op", "x"), std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
