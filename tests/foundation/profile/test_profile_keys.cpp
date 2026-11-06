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
        namespace
        {
            /** @brief Checks that no key of keys is empty or repeats. */
            template<typename Keys>
            void expect_distinct_keys(const Keys& keys)
            {
                std::set<std::string_view> seen;
                for (const std::string_view key : keys)
                {
                    EXPECT_FALSE(key.empty());
                    EXPECT_TRUE(seen.insert(key).second) << key;
                }
            }

            /** @brief Checks that every key but the group name starts with "sushiblas.". */
            template<typename Keys>
            void expect_library_prefix(const Keys& keys, std::string_view group_name)
            {
                for (const std::string_view key : keys)
                    EXPECT_TRUE(key == group_name || key.substr(0, 10) == "sushiblas.") << key;
            }
        } // namespace

        TEST(Small_ProfileKeys, EveryKeyIsListedOnce)
        {
            expect_distinct_keys(Internal::Profile::K_ALL_KEYS);
        }

        TEST(Small_ProfileKeys, EveryFactOfTheLibraryStartsWithItsName)
        {
            expect_library_prefix(Internal::Profile::K_ALL_KEYS, Internal::Profile::K_GROUP_NAME);
        }
    } // namespace Test
} // namespace SushiBLAS
