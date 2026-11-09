/**************************************************************************/
/* test_in_memory_ceilings_source.cpp                                     */
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
 * @file test_in_memory_ceilings_source.cpp
 * @brief Tests that InMemoryCeilingsSource hands back the text put under a name, replaces it and reads nothing for another name.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/profiling/in_memory_ceilings_source.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_InMemoryCeilingsSource, ReadsTheTextPutUnderAName)
        {
            Profiling::InMemoryCeilingsSource source;
            source.put("a.ceilings.json", "text a");
            source.put("b.ceilings.json", "text b");

            EXPECT_EQ(source.read("a.ceilings.json"), "text a");
            EXPECT_EQ(source.read("b.ceilings.json"), "text b");
        }

        TEST(Small_InMemoryCeilingsSource, ReadsNothingForAnUnknownName)
        {
            const Profiling::InMemoryCeilingsSource source;

            EXPECT_FALSE(source.read("a.ceilings.json").has_value());
        }

        TEST(Small_InMemoryCeilingsSource, APutReplacesTheTextOfTheSameName)
        {
            Profiling::InMemoryCeilingsSource source;
            source.put("a.ceilings.json", "first");
            source.put("a.ceilings.json", "second");

            EXPECT_EQ(source.read("a.ceilings.json"), "second");
        }

        TEST(Small_InMemoryCeilingsSource, LocatesAFileByItsName)
        {
            const Profiling::InMemoryCeilingsSource source;

            EXPECT_EQ(source.location("a.ceilings.json"), "memory:a.ceilings.json");
        }
    } // namespace Test
} // namespace SushiBLAS
