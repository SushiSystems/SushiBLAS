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
 * @file test_version.cpp
 * @brief Tests that the version macros agree with each other and with their encoding.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/version.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(SUSHIBLAS_VERSION_ENCODE(1, 2, 3) == 10203, "1.2.3 encodes as 10203.");
        static_assert(SUSHIBLAS_VERSION_ENCODE(0, 10, 0) == 1000, "0.10.0 encodes as 1000.");

        TEST(Small_Version, TheEncodingGivesEachPartTwoDecimalDigits)
        {
            EXPECT_EQ(SUSHIBLAS_VERSION_ENCODE(2, 0, 99), 20099);
        }

        TEST(Small_Version, TheVersionIsTheEncodingOfItsParts)
        {
            EXPECT_EQ(SUSHIBLAS_VERSION, SUSHIBLAS_VERSION_ENCODE(SUSHIBLAS_VERSION_MAJOR,
                                                                  SUSHIBLAS_VERSION_MINOR,
                                                                  SUSHIBLAS_VERSION_PATCH));
        }

        TEST(Small_Version, LaterReleasesEncodeLarger)
        {
            EXPECT_LT(SUSHIBLAS_VERSION_ENCODE(0, 1, 9), SUSHIBLAS_VERSION_ENCODE(0, 2, 0));
            EXPECT_LT(SUSHIBLAS_VERSION_ENCODE(0, 99, 99), SUSHIBLAS_VERSION_ENCODE(1, 0, 0));
        }
    } // namespace Test
} // namespace SushiBLAS
