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
 * @file test_sparse_types.cpp
 * @brief Tests the numbers, the validity, the first index and the names of IndexBase.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/sparse_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The number after the last IndexBase enumerator.
        constexpr int K_PAST_LAST_BASE = 223;

        TEST(Small_IndexBase, KeepsItsSparseBLASNumbers)
        {
            EXPECT_EQ(static_cast<int>(Core::IndexBase::Zero), 221);
            EXPECT_EQ(static_cast<int>(Core::IndexBase::One), 222);
        }

        namespace
        {
            /** @brief Holds one value of IndexBase and whether is_valid takes it for an enumerator. */
            struct ValidityRow
            {
                const char* name;     /**< The row's suffix. */
                Core::IndexBase base; /**< The value handed to is_valid. */
                bool valid;           /**< Whether it is an enumerator. */
            };

            /** @brief Returns both enumerators and each kind of value outside them. */
            std::vector<ValidityRow> validity_rows()
            {
                return {{"Zero", Core::IndexBase::Zero, true},
                        {"One", Core::IndexBase::One, true},
                        {"ACastOfZero", static_cast<Core::IndexBase>(0), false},
                        {"PastTheLastEnumerator", static_cast<Core::IndexBase>(K_PAST_LAST_BASE),
                         false},
                        {"TheLargestByte", static_cast<Core::IndexBase>(255), false}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string validity_row_name(const ::testing::TestParamInfo<ValidityRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Hands is_valid one value at a time. */
        class Small_IndexBaseValidity : public ::testing::TestWithParam<ValidityRow>
        {

        };

        TEST_P(Small_IndexBaseValidity, IsValidAcceptsBothBasesAndNothingElse)
        {
            EXPECT_EQ(Core::is_valid(GetParam().base), GetParam().valid);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_IndexBaseValidity, ::testing::ValuesIn(validity_rows()),
                                 validity_row_name);

        TEST(Small_IndexBase, FirstIndexIsZeroOrOne)
        {
            EXPECT_EQ(Core::first_index(Core::IndexBase::Zero), int64_t{0});
            EXPECT_EQ(Core::first_index(Core::IndexBase::One), int64_t{1});
        }

        TEST(Small_IndexBase, ToStringNamesBothBases)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase::Zero)), "ZERO");
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase::One)), "ONE");
        }

        TEST(Small_IndexBase, ToStringNamesAValueOutsideTheEnumerationUnknown)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::IndexBase{})), "UNKNOWN");
        }
    } // namespace Test
} // namespace SushiBLAS
