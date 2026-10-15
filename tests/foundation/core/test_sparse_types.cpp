/**************************************************************************/
/* test_sparse_types.cpp                                                  */
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
 * @file test_sparse_types.cpp
 * @brief Tests the numbers, the validity, the first index and the names of IndexBase.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/sparse_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The number after the last IndexBase enumerator.
        constexpr int K_PAST_LAST_BASE = 223;

        TEST(Unit_IndexBase, KeepsItsSparseBLASNumbersInOneByte)
        {
            static_assert(sizeof(Core::IndexBase) == 1, "IndexBase is one byte.");

            EXPECT_EQ(static_cast<int>(Core::IndexBase::Zero), 221);
            EXPECT_EQ(static_cast<int>(Core::IndexBase::One), 222);
        }

        TEST(Unit_IndexBase, IsValidAcceptsBothBasesAndNothingElse)
        {
            EXPECT_TRUE(Core::is_valid(Core::IndexBase::Zero));
            EXPECT_TRUE(Core::is_valid(Core::IndexBase::One));
            EXPECT_FALSE(Core::is_valid(Core::IndexBase{}));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::IndexBase>(K_PAST_LAST_BASE)));
        }

        TEST(Unit_IndexBase, FirstIndexIsZeroOrOne)
        {
            static_assert(Core::first_index(Core::IndexBase::Zero) == 0, "Zero starts at 0.");
            static_assert(Core::first_index(Core::IndexBase::One) == 1, "One starts at 1.");

            EXPECT_EQ(Core::first_index(Core::IndexBase::Zero), int64_t{0});
            EXPECT_EQ(Core::first_index(Core::IndexBase::One), int64_t{1});
        }

        TEST(Unit_IndexBase, ToStringNamesBothBasesAndAnUnknownValue)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase::Zero)), "ZERO");
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase::One)), "ONE");
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase{})), "UNKNOWN");
        }
    } // namespace Test
} // namespace SushiBLAS
