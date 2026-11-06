/**************************************************************************/
/* test_errors.cpp                                                        */
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
 * @file test_errors.cpp
 * @brief Tests that UnsupportedExecution is an invalid_argument that keeps its message.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_UnsupportedExecution, IsAFinalInvalidArgument)
        {
            static_assert(std::is_base_of<std::invalid_argument, UnsupportedExecution>::value,
                          "UnsupportedExecution derives from std::invalid_argument.");
            static_assert(std::is_final<UnsupportedExecution>::value,
                          "UnsupportedExecution is final.");

            EXPECT_THROW(throw UnsupportedExecution("refused"), std::invalid_argument);
        }

        TEST(Medium_UnsupportedExecution, KeepsTheMessageItWasGiven)
        {
            const UnsupportedExecution error("ForceReference cannot reach device USM");

            EXPECT_EQ(std::string(error.what()), "ForceReference cannot reach device USM");
        }

        TEST(Medium_UnsupportedExecution, DoesNotCatchAPlainInvalidArgument)
        {
            bool caught_as_unsupported = false;
            bool caught_as_invalid = false;
            try
            {
                throw std::invalid_argument("bad operand");
            }
            catch (const UnsupportedExecution&)
            {
                caught_as_unsupported = true;
            }
            catch (const std::invalid_argument&)
            {
                caught_as_invalid = true;
            }

            EXPECT_FALSE(caught_as_unsupported);
            EXPECT_TRUE(caught_as_invalid);
        }
    } // namespace Test
} // namespace SushiBLAS
