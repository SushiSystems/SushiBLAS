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
 * @file test_default_type_names.cpp
 * @brief Tests that DefaultTypeNames names a type by its index and that the suite macro uses it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_DefaultTypeNames, NamesATypeByItsIndex)
        {
            EXPECT_EQ(DefaultTypeNames::GetName<float>(0), "0");
            EXPECT_EQ(DefaultTypeNames::GetName<double>(12), "12");
        }

        /** @brief Runs one test per element type through SB_TYPED_TEST_SUITE. */
        template<typename T>
        class Unit_TypedSuiteNaming : public ::testing::Test
        {

        };

        /// @brief The types the typed suite below runs over.
        using NamedTypes = ::testing::Types<int32_t, double>;
        SB_TYPED_TEST_SUITE(Unit_TypedSuiteNaming, NamedTypes);

        TYPED_TEST(Unit_TypedSuiteNaming, EachTypeGetsItsOwnSuite)
        {
            const std::string suite =
                ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();

            EXPECT_TRUE(suite == "Unit_TypedSuiteNaming/0" || suite == "Unit_TypedSuiteNaming/1")
                << suite;
        }
    } // namespace Test
} // namespace SushiBLAS
