/**************************************************************************/
/* test_default_type_names.cpp                                            */
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
 * @file test_default_type_names.cpp
 * @brief Tests that DefaultTypeNames names a type by its index and that the suite macro uses it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_DefaultTypeNames, NamesATypeByItsIndex)
        {
            EXPECT_EQ(DefaultTypeNames::GetName<int32_t>(0), "0");
            EXPECT_EQ(DefaultTypeNames::GetName<double>(1), "1");
            EXPECT_EQ(DefaultTypeNames::GetName<double>(12), "12");
        }

        /** @brief Runs one test per element type through SB_TYPED_TEST_SUITE. */
        template<typename T>
        class Medium_TypedSuiteNaming : public ::testing::Test
        {

        };

        /// @brief The types the typed suite below runs over.
        using NamedTypes = ::testing::Types<int32_t, double>;
        SB_TYPED_TEST_SUITE(Medium_TypedSuiteNaming, NamedTypes);

        TYPED_TEST(Medium_TypedSuiteNaming, EachTypeGetsTheSuiteOfItsIndex)
        {
            const std::string expected = std::is_same_v<TypeParam, int32_t>
                                             ? "Medium_TypedSuiteNaming/0"
                                             : "Medium_TypedSuiteNaming/1";

            EXPECT_EQ(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name(),
                      expected);
        }
    } // namespace Test
} // namespace SushiBLAS
