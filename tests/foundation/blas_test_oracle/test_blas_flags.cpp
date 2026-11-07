/**************************************************************************/
/* test_blas_flags.cpp                                                    */
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
 * @file test_blas_flags.cpp
 * @brief Tests that each flag helper maps true and false onto the two BLAS enumerators.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief One helper, one truth value and the enumerator number it must give.
        struct FlagRow
        {
            const char* name;
            int (*helper)(bool);
            bool value;
            int expected;
        };

        /// @brief Returns the number of the enumerator a helper gives for a truth value.
        template<typename Flag, Flag (*Helper)(bool)>
        static int number_of(bool value)
        {
            return static_cast<int>(Helper(value));
        }

        /// @brief Returns the number of an enumerator.
        template<typename Flag>
        static int number(Flag flag)
        {
            return static_cast<int>(flag);
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string row_name(const ::testing::TestParamInfo<FlagRow>& info)
        {
            return info.param.name;
        }

        /// @brief Returns one row per helper and truth value.
        static std::vector<FlagRow> flag_rows()
        {
            using namespace Core;
            return {
                {"transpose_true", &number_of<Transpose, &transpose_of>, true, number(Transpose::Trans)},
                {"transpose_false", &number_of<Transpose, &transpose_of>, false, number(Transpose::NoTrans)},
                {"uplo_true", &number_of<Uplo, &uplo_of>, true, number(Uplo::Upper)},
                {"uplo_false", &number_of<Uplo, &uplo_of>, false, number(Uplo::Lower)},
                {"diag_true", &number_of<Diag, &diag_of>, true, number(Diag::Unit)},
                {"diag_false", &number_of<Diag, &diag_of>, false, number(Diag::NonUnit)},
                {"side_true", &number_of<Side, &side_of>, true, number(Side::Left)},
                {"side_false", &number_of<Side, &side_of>, false, number(Side::Right)},
            };
        }

        /// @brief Runs one helper on one truth value.
        class Small_BlasFlags : public ::testing::TestWithParam<FlagRow>
        {
        };

        TEST_P(Small_BlasFlags, TheHelperMapsTheTruthValueOntoItsEnumerator)
        {
            const FlagRow& row = GetParam();
            EXPECT_EQ(row.helper(row.value), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BlasFlags, ::testing::ValuesIn(flag_rows()), row_name);

        TEST(Small_BlasFlagsValidity, AHelperGivesAValidEnumerator)
        {
            EXPECT_TRUE(Core::is_valid(uplo_of(true)));
        }
    } // namespace Test
} // namespace SushiBLAS
