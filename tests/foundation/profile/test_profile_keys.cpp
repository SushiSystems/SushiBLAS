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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
