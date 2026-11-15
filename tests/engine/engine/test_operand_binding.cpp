/**************************************************************************/
/* test_operand_binding.cpp                                               */
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
 * @file test_operand_binding.cpp
 * @brief Tests operand_address(): the address of a bound operand from its slot and offset, read
 *        through the runtime's view, and the refusals the view makes.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

#include <SushiRuntime/handle/bound_operands.hpp>

#include <SushiBLAS/engine/operand_binding.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_OperandBinding, AnAddressIsTheBaseOfItsSlotPlusItsByteOffset)
        {
            std::array<float, 8> first{};
            std::array<float, 4> second{};
            void* table[2] = {first.data(), second.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 2);

            EXPECT_EQ(operand_address<float>(operands, OperandBinding{0, 0}), first.data());
            EXPECT_EQ(operand_address<float>(operands, OperandBinding{0, 3 * sizeof(float)}), first.data() + 3);
            EXPECT_EQ(operand_address<const float>(operands, OperandBinding{1, sizeof(float)}), second.data() + 1);
        }

        TEST(Small_OperandBinding, AnAbsentOperandReadsAsNull)
        {
            std::array<float, 2> only{};
            void* table[1] = {only.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 1);

            EXPECT_EQ(operand_address<float>(operands, OperandBinding()), nullptr);
            EXPECT_EQ(OperandBinding().slot, OperandBinding::K_ABSENT);
        }

        TEST(Small_OperandBinding, ASlotPastTheBoundOperandsIsRefused)
        {
            const SushiRuntime::BoundOperands operands;

            EXPECT_THROW(static_cast<void>(operand_address<float>(operands, OperandBinding{0, 0})),
                         std::runtime_error);
        }

        TEST(Small_OperandBinding, AnOffsetMisalignedForTheElementTypeIsRefused)
        {
            std::array<float, 4> only{};
            void* table[1] = {only.data()};
            const SushiRuntime::BoundOperands operands = SushiRuntime::BoundOperands::of_bytes(table, 1);

            EXPECT_THROW(static_cast<void>(operand_address<float>(operands, OperandBinding{0, 2})),
                         std::runtime_error);
        }

        TEST(Small_OperandBinding, AnElementTypeThatIsNotTheBuffersOwnIsRefused)
        {
            std::array<float, 4> only{};
            void* table[1] = {only.data()};
            const SushiRuntime::ElementLayout layouts[1] = {SushiRuntime::ElementLayout{sizeof(float), alignof(float)}};
            const SushiRuntime::BoundOperands operands(table, layouts, 1);

            EXPECT_THROW(static_cast<void>(operand_address<double>(operands, OperandBinding{0, 0})),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
