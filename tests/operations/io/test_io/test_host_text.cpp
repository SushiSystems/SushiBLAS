/**************************************************************************/
/* test_host_text.cpp                                                     */
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
 * @file test_host_text.cpp
 * @brief Tests the text of an empty tensor and the layout refusal of load_npy, on a host engine with no device.
 * @author Mustafa Garip
 */

#include <string>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives each test a host engine. */
        class Small_HostIO : public HostEngineTest
        {
        };

        /** @brief Tests that a column-major tensor of rank three is refused before any file is opened. */
        TEST_F(Small_HostIO, LoadNpyRejectsAColumnMajorRankThreeTensor)
        {
            reinit_engine(Core::Layout::ColumnMajor);
            auto t = engine->create_tensor({2, 2, 2});

            EXPECT_THROW((void)SushiBLAS::IO().load_npy(t, "column_major.npy"), std::invalid_argument);
        }

        /** @brief Tests that an empty tensor prints an empty list. */
        TEST_F(Small_HostIO, ToStringOfAnEmptyTensorPrintsAnEmptyList)
        {
            auto t = engine->create_tensor({0});

            EXPECT_EQ(SushiBLAS::IO().to_string(t), "Tensor(shape=[0], dtype=f32):\n[]");
        }
    } // namespace Test
} // namespace SushiBLAS
