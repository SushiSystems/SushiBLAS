/**************************************************************************/
/* test_gemm_bindings.cpp                                                 */
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
 * @file test_gemm_bindings.cpp
 * @brief Tests that GemmBindings names the slots of A, B and C and that bound_gemm_operands reads
 *        them, with their offsets, from the table the runtime bound.
 * @author Mustafa Garip
 */

#include <array>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/gemm_bindings.hpp>
#include <SushiBLAS/blas/gemm_split.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_GemmBindings, ABoundCallReadsItsThreeOperandsFromTheirSlotsAndOffsets)
        {
            std::array<float, 8> a{};
            std::array<float, 8> b{};
            std::array<float, 8> c{};
            void* table[3] = {a.data(), b.data(), c.data()};
            const Dispatch::GemmBindings bindings{OperandBinding{0, sizeof(float)},
                                                  OperandBinding{1, 2 * sizeof(float)},
                                                  OperandBinding{2, 0}};

            const Dispatch::GemmOperands<float> operands = Dispatch::bound_gemm_operands<float>(
                SushiRuntime::BoundOperands::of_bytes(table, 3), bindings, Dispatch::GemmOperands<float>{});

            EXPECT_EQ(operands.a, a.data() + 1);
            EXPECT_EQ(operands.b, b.data() + 2);
            EXPECT_EQ(operands.c, c.data());
        }

        TEST(Small_GemmBindings, ACallThatBoundNothingReadsTheRecordedPointers)
        {
            std::array<float, 2> a{};
            std::array<float, 2> b{};
            std::array<float, 2> c{};
            const Dispatch::GemmOperands<float> recorded{a.data(), b.data(), c.data()};

            const Dispatch::GemmOperands<float> operands = Dispatch::bound_gemm_operands<float>(
                SushiRuntime::BoundOperands(), std::nullopt, recorded);

            EXPECT_EQ(operands.a, a.data());
            EXPECT_EQ(operands.b, b.data());
            EXPECT_EQ(operands.c, c.data());
        }
    } // namespace Test
} // namespace SushiBLAS
