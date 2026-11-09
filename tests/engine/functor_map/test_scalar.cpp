/**************************************************************************/
/* test_scalar.cpp                                                        */
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
 * @file test_scalar.cpp
 * @brief Tests that ElementScalar holds one parameter in every element type, integers saturated.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names one read of the scalar 1.5 in an element type, widened to double. */
            struct FloatingReadRow
            {
                const char* name;                                  /**< The row name a failure shows. */
                double (*read)(const Dispatch::ElementScalar&);    /**< Reads the scalar in the type. */
            };

            /** @brief Returns one row per floating type and one for the stored value. */
            std::vector<FloatingReadRow> floating_read_rows()
            {
                return {{"Value", [](const Dispatch::ElementScalar& s) { return s.value(); }},
                        {"Double", [](const Dispatch::ElementScalar& s) { return s.as<double>(); }},
                        {"Float",
                         [](const Dispatch::ElementScalar& s)
                         { return static_cast<double>(s.as<float>()); }},
                        {"Half",
                         [](const Dispatch::ElementScalar& s)
                         { return static_cast<double>(static_cast<float>(s.as<sycl::half>())); }}};
            }
        } // namespace

        /** @brief Runs one row of floating_read_rows(). */
        class Small_ElementScalarFloating : public ::testing::TestWithParam<FloatingReadRow>
        {
        };

        TEST_P(Small_ElementScalarFloating, HoldsTheValueInTheType)
        {
            EXPECT_EQ(GetParam().read(Dispatch::ElementScalar(1.5)), 1.5);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ElementScalarFloating, ::testing::ValuesIn(floating_read_rows()),
            [](const ::testing::TestParamInfo<FloatingReadRow>& info)
            { return std::string(info.param.name); });

        namespace
        {
            /** @brief Names one narrowing of a double to an integer type, widened to int64. */
            struct TruncationRow
            {
                const char* name;    /**< The row name a failure shows. */
                double input;        /**< The scalar narrowed. */
                std::int64_t (*read)(const Dispatch::ElementScalar&); /**< Reads the scalar in the type. */
                std::int64_t expected; /**< The integer the narrowing gives. */
            };

            /** @brief Returns one row per sign and integer type. */
            std::vector<TruncationRow> truncation_rows()
            {
                const auto as_int32 = [](const Dispatch::ElementScalar& s)
                { return static_cast<std::int64_t>(s.as<std::int32_t>()); };
                const auto as_int64 = [](const Dispatch::ElementScalar& s)
                { return s.as<std::int64_t>(); };
                return {{"PositiveToInt32", 2.9, as_int32, 2},
                        {"NegativeToInt32", -2.9, as_int32, -2},
                        {"PositiveToInt64", 2.9, as_int64, 2},
                        {"NegativeToInt64", -2.9, as_int64, -2}};
            }
        } // namespace

        /** @brief Runs one row of truncation_rows(). */
        class Small_ElementScalarTruncation : public ::testing::TestWithParam<TruncationRow>
        {
        };

        TEST_P(Small_ElementScalarTruncation, IntegersTruncateTowardZero)
        {
            EXPECT_EQ(GetParam().read(Dispatch::ElementScalar(GetParam().input)),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ElementScalarTruncation, ::testing::ValuesIn(truncation_rows()),
            [](const ::testing::TestParamInfo<TruncationRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_ElementScalar, IntegersSaturateAtBothEnds)
        {
            EXPECT_EQ(Dispatch::ElementScalar(1e20).as<std::int32_t>(),
                      std::numeric_limits<std::int32_t>::max());
            EXPECT_EQ(Dispatch::ElementScalar(-1e20).as<std::int64_t>(),
                      std::numeric_limits<std::int64_t>::min());
        }

        TEST(Small_ElementScalar, ANaNBecomesZero)
        {
            EXPECT_EQ(Dispatch::ElementScalar(std::nan("")).as<std::int32_t>(), 0);
        }
    } // namespace Test
} // namespace SushiBLAS
