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
            EXPECT_FALSE(Core::is_transposed(Core::Transpose::NoTrans));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::Trans));
            EXPECT_TRUE(Core::is_transposed(Core::Transpose::ConjTrans));
        }

        TEST(Unit_BlasTypes, IsConjugatedHoldsForTheConjugateTransposeOnly)
        {
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::NoTrans));
            EXPECT_FALSE(Core::is_conjugated(Core::Transpose::Trans));
            EXPECT_TRUE(Core::is_conjugated(Core::Transpose::ConjTrans));
        }

        TEST(Unit_BlasTypes, ToStringNamesEveryEnumerator)
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

        TEST(Unit_BlasTypes, ToStringNamesAValueOutsideTheEnumerationUnknown)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::Transpose{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Uplo{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Diag{})), "UNKNOWN");
            EXPECT_EQ(std::string(Core::to_string(Core::Side{})), "UNKNOWN");
        }

        TEST(Unit_DataType, EveryEnumeratorKeepsItsFileFormatNumber)
        {
            static_assert(static_cast<int>(Core::DataType::Half) == 0, "HALF is 0.");
            static_assert(static_cast<int>(Core::DataType::Float32) == 1, "FLOAT32 is 1.");
            static_assert(static_cast<int>(Core::DataType::Float64) == 2, "FLOAT64 is 2.");
            static_assert(static_cast<int>(Core::DataType::Complex32) == 3, "COMPLEX32 is 3.");
            static_assert(static_cast<int>(Core::DataType::Complex64) == 4, "COMPLEX64 is 4.");
            static_assert(static_cast<int>(Core::DataType::Int32) == 5, "INT32 is 5.");
            static_assert(static_cast<int>(Core::DataType::Int64) == 6, "INT64 is 6.");

            EXPECT_EQ(static_cast<int>(Core::DataType::Half), 0);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float32), 1);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float64), 2);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex32), 3);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex64), 4);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int32), 5);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int64), 6);
        }

        TEST(Unit_DataType, IsOneByte)
        {
            static_assert(sizeof(Core::DataType) == 1, "DataType is one byte.");

            EXPECT_EQ(sizeof(Core::DataType), 1u);
        }
    } // namespace Test
} // namespace SushiBLAS
