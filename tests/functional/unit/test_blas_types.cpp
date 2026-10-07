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
 * @file test_blas_types.cpp
 * @brief Tests the numbers and the size of DataType and of the BLAS flags, and the flag queries.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The value of a value-initialised flag, below every enumerator.
        constexpr std::uint8_t K_BELOW_EVERY_FLAG = 0;

        /// @brief A value above every enumerator of every flag.
        constexpr std::uint8_t K_ABOVE_EVERY_FLAG = 255;

        TEST(Unit_BlasTypes, EveryEnumeratorCarriesItsCblasNumber)
        {
            EXPECT_EQ(static_cast<int>(Core::Transpose::NO_TRANS), 111);
            EXPECT_EQ(static_cast<int>(Core::Transpose::TRANS), 112);
            EXPECT_EQ(static_cast<int>(Core::Transpose::CONJ_TRANS), 113);
            EXPECT_EQ(static_cast<int>(Core::Uplo::UPPER), 121);
            EXPECT_EQ(static_cast<int>(Core::Uplo::LOWER), 122);
            EXPECT_EQ(static_cast<int>(Core::Diag::NON_UNIT), 131);
            EXPECT_EQ(static_cast<int>(Core::Diag::UNIT), 132);
            EXPECT_EQ(static_cast<int>(Core::Side::LEFT), 141);
            EXPECT_EQ(static_cast<int>(Core::Side::RIGHT), 142);
        }

        TEST(Unit_BlasTypes, EveryFlagIsOneByte)
        {
            static_assert(sizeof(Core::Transpose) == 1, "Transpose is one byte.");
            static_assert(sizeof(Core::Uplo) == 1, "Uplo is one byte.");
            static_assert(sizeof(Core::Diag) == 1, "Diag is one byte.");
            static_assert(sizeof(Core::Side) == 1, "Side is one byte.");

            EXPECT_EQ(sizeof(Core::Transpose), 1u);
            EXPECT_EQ(sizeof(Core::Uplo), 1u);
            EXPECT_EQ(sizeof(Core::Diag), 1u);
            EXPECT_EQ(sizeof(Core::Side), 1u);
        }

        TEST(Unit_BlasTypes, IsValidAcceptsEveryEnumerator)
        {
            EXPECT_TRUE(Core::is_valid(Core::Transpose::NO_TRANS));
            EXPECT_TRUE(Core::is_valid(Core::Transpose::TRANS));
            EXPECT_TRUE(Core::is_valid(Core::Transpose::CONJ_TRANS));
            EXPECT_TRUE(Core::is_valid(Core::Uplo::UPPER));
            EXPECT_TRUE(Core::is_valid(Core::Uplo::LOWER));
            EXPECT_TRUE(Core::is_valid(Core::Diag::NON_UNIT));
            EXPECT_TRUE(Core::is_valid(Core::Diag::UNIT));
            EXPECT_TRUE(Core::is_valid(Core::Side::LEFT));
            EXPECT_TRUE(Core::is_valid(Core::Side::RIGHT));
        }

        TEST(Unit_BlasTypes, IsValidRefusesAValueInitialisedFlag)
        {
            EXPECT_FALSE(Core::is_valid(Core::Transpose{}));
            EXPECT_FALSE(Core::is_valid(Core::Uplo{}));
            EXPECT_FALSE(Core::is_valid(Core::Diag{}));
            EXPECT_FALSE(Core::is_valid(Core::Side{}));

            EXPECT_FALSE(Core::is_valid(static_cast<Core::Transpose>(K_BELOW_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Uplo>(K_BELOW_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Diag>(K_BELOW_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Side>(K_BELOW_EVERY_FLAG)));
        }

        TEST(Unit_BlasTypes, IsValidRefusesAValueAboveEveryEnumerator)
        {
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Transpose>(K_ABOVE_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Uplo>(K_ABOVE_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Diag>(K_ABOVE_EVERY_FLAG)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Side>(K_ABOVE_EVERY_FLAG)));
        }

        TEST(Unit_BlasTypes, IsValidRefusesTheNumberOfAnotherFlag)
        {
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Transpose>(121)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Uplo>(111)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Diag>(141)));
            EXPECT_FALSE(Core::is_valid(static_cast<Core::Side>(131)));
        }

        TEST(Unit_BlasTypes, IsTransposedHoldsForBothTransposes)
        {
            EXPECT_FALSE(Core::is_transposed(Core::Transpose::NO_TRANS));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::TRANS));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::CONJ_TRANS));
        }

        TEST(Unit_BlasTypes, IsConjugatedHoldsForTheConjugateTransposeOnly)
        {
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::NO_TRANS));
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::TRANS));
            EXPECT_TRUE(Core::is_conjugated(Core::Transpose::CONJ_TRANS));
        }

        TEST(Unit_BlasTypes, ToStringNamesEveryEnumerator)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::NO_TRANS)), "NO_TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::TRANS)), "TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose::CONJ_TRANS)), "CONJ_TRANS");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo::UPPER)), "UPPER");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo::LOWER)), "LOWER");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag::NON_UNIT)), "NON_UNIT");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag::UNIT)), "UNIT");
            EXPECT_EQ(std::string(Core::to_string(Core::Side::LEFT)), "LEFT");
            EXPECT_EQ(std::string(Core::to_string(Core::Side::RIGHT)), "RIGHT");
        }

        TEST(Unit_BlasTypes, ToStringNamesAValueOutsideTheEnumerationUnknown)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Side{})), "UNKNOWN");
        }

        TEST(Unit_DataType, EveryEnumeratorKeepsItsFileFormatNumber)
        {
            static_assert(static_cast<int>(Core::DataType::HALF) == 0, "HALF is 0.");
            static_assert(static_cast<int>(Core::DataType::FLOAT32) == 1, "FLOAT32 is 1.");
            static_assert(static_cast<int>(Core::DataType::FLOAT64) == 2, "FLOAT64 is 2.");
            static_assert(static_cast<int>(Core::DataType::COMPLEX32) == 3, "COMPLEX32 is 3.");
            static_assert(static_cast<int>(Core::DataType::COMPLEX64) == 4, "COMPLEX64 is 4.");
            static_assert(static_cast<int>(Core::DataType::INT32) == 5, "INT32 is 5.");
            static_assert(static_cast<int>(Core::DataType::INT64) == 6, "INT64 is 6.");

            EXPECT_EQ(static_cast<int>(Core::DataType::HALF), 0);
            EXPECT_EQ(static_cast<int>(Core::DataType::FLOAT32), 1);
            EXPECT_EQ(static_cast<int>(Core::DataType::FLOAT64), 2);
            EXPECT_EQ(static_cast<int>(Core::DataType::COMPLEX32), 3);
            EXPECT_EQ(static_cast<int>(Core::DataType::COMPLEX64), 4);
            EXPECT_EQ(static_cast<int>(Core::DataType::INT32), 5);
            EXPECT_EQ(static_cast<int>(Core::DataType::INT64), 6);
        }

        TEST(Unit_DataType, IsOneByte)
        {
            static_assert(sizeof(Core::DataType) == 1, "DataType is one byte.");

            EXPECT_EQ(sizeof(Core::DataType), 1u);
        }
    } // namespace Test
} // namespace SushiBLAS
