/**************************************************************************/
/* test_host_late_count_keeper.cpp                                        */
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
 * @file test_host_late_count_keeper.cpp
 * @brief Tests that HostLateCountKeeper lends an empty late count and counts the owners it keeps.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/host_late_count_keeper.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_HostLateCountKeeper, ABorrowedCountHoldsNoOwnerAndNoWords)
        {
            const HostLateCountKeeper keeper;

            const LateCount late = keeper.borrow(OperandSet{}, Tensor{});

            EXPECT_EQ(late.owner, nullptr);
        }

        TEST(Small_HostLateCountKeeper, EachKeptOwnerIsCounted)
        {
            HostLateCountKeeper keeper;

            keeper.keep(std::make_shared<SushiRuntime::Liveness>());
            keeper.keep(std::make_shared<SushiRuntime::Liveness>());

            EXPECT_EQ(keeper.kept_count(), std::size_t{2});
        }
    } // namespace Test
} // namespace SushiBLAS
