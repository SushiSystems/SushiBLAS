/**************************************************************************/
/* test_version.cpp                                                       */
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
 * @file test_version.cpp
 * @brief Tests that the version macros agree with each other and with their encoding.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/version.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_Version, TheStringSpellsTheThreeNumbers)
        {
            const std::string expected = std::to_string(SUSHIBLAS_VERSION_MAJOR) + "." +
                                         std::to_string(SUSHIBLAS_VERSION_MINOR) + "." +
                                         std::to_string(SUSHIBLAS_VERSION_PATCH);

            EXPECT_EQ(std::string(SUSHIBLAS_VERSION_STRING), expected);
        }

        TEST(Unit_Version, TheEncodingGivesEachPartTwoDecimalDigits)
        {
            static_assert(SUSHIBLAS_VERSION_ENCODE(1, 2, 3) == 10203, "1.2.3 encodes as 10203.");
            static_assert(SUSHIBLAS_VERSION_ENCODE(0, 10, 0) == 1000, "0.10.0 encodes as 1000.");

            EXPECT_EQ(SUSHIBLAS_VERSION_ENCODE(2, 0, 99), 20099);
        }

        TEST(Unit_Version, TheVersionIsTheEncodingOfItsParts)
        {
            EXPECT_EQ(SUSHIBLAS_VERSION, SUSHIBLAS_VERSION_ENCODE(SUSHIBLAS_VERSION_MAJOR,
                                                                  SUSHIBLAS_VERSION_MINOR,
                                                                  SUSHIBLAS_VERSION_PATCH));
        }

        TEST(Unit_Version, LaterReleasesEncodeLarger)
        {
            EXPECT_LT(SUSHIBLAS_VERSION_ENCODE(0, 1, 9), SUSHIBLAS_VERSION_ENCODE(0, 2, 0));
            EXPECT_LT(SUSHIBLAS_VERSION_ENCODE(0, 99, 99), SUSHIBLAS_VERSION_ENCODE(1, 0, 0));
        }
    } // namespace Test
} // namespace SushiBLAS
