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
 * @file test_vector_operand.cpp
 * @brief Tests how a tensor is read as a vector operand, strided ones included, and its checks.
 * @author Mustafa Garip
 */

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

        TEST_F(Medium_VectorOperand, LengthAndIncrementChecksNameTheirOperand)
        {
            const Dispatch::VectorOperand v{5, 2};

            EXPECT_THROW(Dispatch::require_vector_length(v, 4, "A's columns", "op", "x"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_unit_increment(v, "op", "x"), std::invalid_argument);
            EXPECT_NO_THROW(Dispatch::require_vector_length(v, 5, "A's columns", "op", "x"));
        }
    } // namespace Test
} // namespace SushiBLAS
