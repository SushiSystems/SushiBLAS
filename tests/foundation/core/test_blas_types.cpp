/**************************************************************************/
/* test_blas_types.cpp                                                    */
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
 * @file test_blas_types.cpp
 * @brief Tests the numbers and the size of the BLAS flags, and the flag queries.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The number a value-initialised flag has, below every enumerator.
        constexpr int K_BELOW_EVERY_FLAG = 0;

        /// @brief A value above every enumerator of every flag.
        constexpr int K_ABOVE_EVERY_FLAG = 255;

        /// @brief Returns whether is_valid accepts the flag of type Flag that has this number.
        template<typename Flag>
        static bool is_valid_number(int number)
        {
            return Core::is_valid(static_cast<Flag>(static_cast<std::uint8_t>(number)));
        }

        /// @brief One flag type and a number that is not one of its enumerators.
        struct RefusedRow
        {
            const char* name;
            bool (*accepts)(int);
            int number;
        };

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string refused_row_name(const ::testing::TestParamInfo<RefusedRow>& info)
        {
            return info.param.name;
        }

        /**
         * @brief Returns, per flag, the value-initialised number, a number above every
         *        enumerator and the number of another flag's enumerator.
         */
        static std::vector<RefusedRow> refused_rows()
        {
            return {
                {"Transpose_below", &is_valid_number<Core::Transpose>, K_BELOW_EVERY_FLAG},
                {"Transpose_above", &is_valid_number<Core::Transpose>, K_ABOVE_EVERY_FLAG},
                {"Transpose_UploNumber", &is_valid_number<Core::Transpose>, 121},
                {"Uplo_below", &is_valid_number<Core::Uplo>, K_BELOW_EVERY_FLAG},
                {"Uplo_above", &is_valid_number<Core::Uplo>, K_ABOVE_EVERY_FLAG},
                {"Uplo_TransposeNumber", &is_valid_number<Core::Uplo>, 111},
                {"Diag_below", &is_valid_number<Core::Diag>, K_BELOW_EVERY_FLAG},
                {"Diag_above", &is_valid_number<Core::Diag>, K_ABOVE_EVERY_FLAG},
                {"Diag_SideNumber", &is_valid_number<Core::Diag>, 141},
                {"Side_below", &is_valid_number<Core::Side>, K_BELOW_EVERY_FLAG},
                {"Side_above", &is_valid_number<Core::Side>, K_ABOVE_EVERY_FLAG},
                {"Side_DiagNumber", &is_valid_number<Core::Side>, 131},
            };
        }

        /// @brief Runs one refused number against its flag.
        class Small_BlasTypesRefused : public ::testing::TestWithParam<RefusedRow>
        {
        };

        TEST_P(Small_BlasTypesRefused, IsValidRefusesAValueInitialisedFlag)
        {
            const RefusedRow& row = GetParam();
            EXPECT_FALSE(row.accepts(row.number));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BlasTypesRefused, ::testing::ValuesIn(refused_rows()),
                                 refused_row_name);

        TEST(Small_BlasTypes, EveryEnumeratorCarriesItsCblasNumber)
        {
            EXPECT_EQ(static_cast<int>(Core::Transpose::NoTrans), 111);
            EXPECT_EQ(static_cast<int>(Core::Transpose::Trans), 112);
            EXPECT_EQ(static_cast<int>(Core::Transpose::ConjTrans), 113);
            EXPECT_EQ(static_cast<int>(Core::Uplo::Upper), 121);
            EXPECT_EQ(static_cast<int>(Core::Uplo::Lower), 122);
            EXPECT_EQ(static_cast<int>(Core::Diag::NonUnit), 131);
            EXPECT_EQ(static_cast<int>(Core::Diag::Unit), 132);
            EXPECT_EQ(static_cast<int>(Core::Side::Left), 141);
            EXPECT_EQ(static_cast<int>(Core::Side::Right), 142);
        }

        TEST(Small_BlasTypes, IsValidAcceptsEveryEnumerator)
        {
            EXPECT_TRUE(Core::is_valid(Core::Transpose::NoTrans));
            EXPECT_TRUE(Core::is_valid(Core::Transpose::Trans));
            EXPECT_TRUE(Core::is_valid(Core::Transpose::ConjTrans));
            EXPECT_TRUE(Core::is_valid(Core::Uplo::Upper));
            EXPECT_TRUE(Core::is_valid(Core::Uplo::Lower));
            EXPECT_TRUE(Core::is_valid(Core::Diag::NonUnit));
            EXPECT_TRUE(Core::is_valid(Core::Diag::Unit));
            EXPECT_TRUE(Core::is_valid(Core::Side::Left));
            EXPECT_TRUE(Core::is_valid(Core::Side::Right));
        }

        TEST(Small_BlasTypes, IsTransposedHoldsForBothTransposes)
        {
            EXPECT_FALSE(Core::is_transposed(Core::Transpose::NoTrans));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::Trans));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::ConjTrans));
        }

        TEST(Small_BlasTypes, IsConjugatedHoldsForTheConjugateTransposeOnly)
        {
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::NoTrans));
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::Trans));
            EXPECT_TRUE(Core::is_conjugated(Core::Transpose::ConjTrans));
        }

        TEST(Small_BlasTypes, ToStringNamesEveryEnumerator)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::NoTrans)), "NO_TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::Trans)), "TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::ConjTrans)), "CONJ_TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo::Upper)), "UPPER");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo::Lower)), "LOWER");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag::NonUnit)), "NON_UNIT");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag::Unit)), "UNIT");
            EXPECT_EQ(std::string(Core::to_string(Core::Side::Left)), "LEFT");
            EXPECT_EQ(std::string(Core::to_string(Core::Side::Right)), "RIGHT");
        }

        TEST(Small_BlasTypes, ToStringNamesAValueOutsideTheEnumerationUnknown)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Side{})), "UNKNOWN");
        }
    } // namespace Test
} // namespace SushiBLAS
