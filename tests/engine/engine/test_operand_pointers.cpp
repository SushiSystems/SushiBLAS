/**************************************************************************/
/* test_operand_pointers.cpp                                              */
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
 * @file test_operand_pointers.cpp
 * @brief Tests that OperandPointers finds the address of a bound operand from its slot and offset,
 *        and reads an absent operand as null.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/operand_pointers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_OperandPointers, AViewOfNoPointerIsEmpty)
        {
            const OperandPointers pointers;

            EXPECT_TRUE(pointers.is_empty());
            EXPECT_EQ(pointers.size(), std::size_t{0});
        }

        TEST(Small_OperandPointers, AnAddressIsTheBaseOfItsSlotPlusItsByteOffset)
        {
            std::array<float, 8> first{};
            std::array<float, 4> second{};
            void* table[2] = {first.data(), second.data()};
            const OperandPointers pointers(table, 2);

            EXPECT_EQ(pointers.size(), std::size_t{2});
            EXPECT_EQ(pointers.address(OperandBinding{0, 0}), first.data());
            EXPECT_EQ(pointers.address(OperandBinding{0, 3 * sizeof(float)}), first.data() + 3);
            EXPECT_EQ(pointers.as<float>(OperandBinding{1, sizeof(float)}), second.data() + 1);
        }

        TEST(Small_OperandPointers, AnAbsentOperandReadsAsNull)
        {
            std::array<float, 2> only{};
            void* table[1] = {only.data()};
            const OperandPointers pointers(table, 1);

            EXPECT_EQ(pointers.address(OperandBinding()), nullptr);
            EXPECT_EQ(OperandBinding().slot, OperandBinding::K_ABSENT);
        }
    } // namespace Test
} // namespace SushiBLAS
