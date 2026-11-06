/**************************************************************************/
/* test_profile_keys.cpp                                                  */
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
 * @file test_profile_keys.cpp
 * @brief Tests that the names SushiBLAS writes into a record are listed once each.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile_keys.hpp>

#include <set>
#include <string_view>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ProfileKeys, EveryKeyIsListedOnce)
        {
            std::set<std::string_view> seen;
            for (const std::string_view key : Internal::Profile::K_ALL_KEYS)
            {
                EXPECT_FALSE(key.empty());
                EXPECT_TRUE(seen.insert(key).second) << key;
            }
        }

        TEST(Small_ProfileKeys, EveryFactOfTheLibraryStartsWithItsName)
        {
            for (const std::string_view key : Internal::Profile::K_ALL_KEYS)
            {
                if (key != Internal::Profile::K_GROUP_NAME)
                    EXPECT_EQ(key.substr(0, 10), "sushiblas.") << key;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
