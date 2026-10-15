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
        TEST(Unit_Traits, IsComplexHoldsForEveryStdComplex)
        {
            static_assert(Core::is_complex_v<std::complex<float>>, "complex<float> is complex.");
            static_assert(Core::is_complex_v<std::complex<double>>, "complex<double> is complex.");
            static_assert(Core::is_complex<std::complex<float>>::value, "The trait agrees.");

            EXPECT_TRUE(Core::is_complex_v<std::complex<float>>);
            EXPECT_TRUE(Core::is_complex_v<std::complex<double>>);
        }

        TEST(Unit_Traits, IsComplexIsFalseForRealAndIntegerTypes)
        {
            static_assert(!Core::is_complex_v<float>, "float is real.");
            static_assert(!Core::is_complex_v<double>, "double is real.");
            static_assert(!Core::is_complex_v<int32_t>, "int32_t is an integer.");
            static_assert(!Core::is_complex_v<int64_t>, "int64_t is an integer.");

            EXPECT_FALSE(Core::is_complex_v<float>);
            EXPECT_FALSE(Core::is_complex_v<int64_t>);
        }

        TEST(Unit_Traits, IsComplexReadsThePlainTypeOnly)
        {
            static_assert(!Core::is_complex_v<const std::complex<float>>,
                          "A qualified complex is not matched.");

            EXPECT_FALSE(Core::is_complex_v<std::complex<float>*>);
        }
    } // namespace Test
} // namespace SushiBLAS
