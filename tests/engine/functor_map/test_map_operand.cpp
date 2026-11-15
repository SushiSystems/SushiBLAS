/**************************************************************************/
/* test_map_operand.cpp                                                   */
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
 * @file test_map_operand.cpp
 * @brief Tests MapOperand::resolve(): a bound operand reads the table of the run, a recorded one
 *        its own address, and an operand with no storage reads as null.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiRuntime/handle/bound_operands.hpp>
#include <SushiBLAS/engine/operand_binding.hpp>
#include <SushiBLAS/functor_map/map_operand.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_MapOperand, ABoundOperandReadsTheSlotAndOffsetOfTheTableItIsGiven)
        {
            std::array<float, 8> first{};
            std::array<float, 4> second{};
            void* table[2] = {first.data(), second.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 2);

            const Dispatch::Detail::MapOperand operand(OperandBinding{0, 3 * sizeof(float)});

            EXPECT_EQ(operand.resolve(operands), first.data() + 3);
        }

        TEST(Small_MapOperand, ARecordedOperandReadsItsOwnAddressWhateverTheTable)
        {
            std::array<float, 4> own{};
            std::array<float, 4> other{};
            void* table[1] = {other.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 1);

            const Dispatch::Detail::MapOperand operand(static_cast<const void*>(own.data()));

            EXPECT_EQ(operand.resolve(operands), own.data());
            EXPECT_EQ(operand.resolve(SushiRuntime::BoundOperands()), own.data());
        }

        TEST(Small_MapOperand, AnOperandWithNoStorageReadsAsNull)
        {
            std::array<float, 4> only{};
            void* table[1] = {only.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 1);

            EXPECT_EQ(Dispatch::Detail::MapOperand().resolve(operands), nullptr);
        }
    } // namespace Test
} // namespace SushiBLAS
