/**************************************************************************/
/* test_traits.cpp                                                        */
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
 * @file test_traits.cpp
 * @brief Tests that is_complex holds for std::complex and for nothing else.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/traits.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(Core::is_complex_v<std::complex<float>>, "complex<float> is complex.");
        static_assert(Core::is_complex_v<std::complex<double>>, "complex<double> is complex.");
        static_assert(Core::is_complex<std::complex<float>>::value, "The trait agrees.");
        static_assert(!Core::is_complex_v<float>, "float is real.");
        static_assert(!Core::is_complex_v<double>, "double is real.");
        static_assert(!Core::is_complex_v<int32_t>, "int32_t is an integer.");
        static_assert(!Core::is_complex_v<int64_t>, "int64_t is an integer.");
        static_assert(!Core::is_complex_v<const std::complex<float>>,
                      "A qualified complex is not matched.");

        TEST(Small_Traits, IsComplexHoldsForEveryStdComplex)
        {
            EXPECT_TRUE(Core::is_complex_v<std::complex<float>>);
            EXPECT_TRUE(Core::is_complex_v<std::complex<double>>);
        }

        TEST(Small_Traits, IsComplexIsFalseForRealAndIntegerTypes)
        {
            EXPECT_FALSE(Core::is_complex_v<float>);
            EXPECT_FALSE(Core::is_complex_v<int64_t>);
        }

        TEST(Small_Traits, IsComplexReadsThePlainTypeOnly)
        {
            EXPECT_FALSE(Core::is_complex_v<std::complex<float>*>);
        }
    } // namespace Test
} // namespace SushiBLAS
