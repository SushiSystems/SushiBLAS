/**************************************************************************/
/* test_accumulator.cpp                                                   */
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
 * @file test_accumulator.cpp
 * @brief Tests that Accumulator widens HALF to FLOAT32 and keeps every other type.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_Accumulator, HalfAccumulatesInFloat)
        {
            static_assert(std::is_same<Kernels::Accumulator<sycl::half>, float>::value,
                          "HALF accumulates in FLOAT32.");

            EXPECT_TRUE((std::is_same<Kernels::AccumulatorOf<sycl::half>::type, float>::value));
        }

        TEST(Unit_Accumulator, EveryOtherTypeAccumulatesInItself)
        {
            static_assert(std::is_same<Kernels::Accumulator<float>, float>::value, "float.");
            static_assert(std::is_same<Kernels::Accumulator<double>, double>::value, "double.");
            static_assert(std::is_same<Kernels::Accumulator<int32_t>, int32_t>::value, "int32.");
            static_assert(std::is_same<Kernels::Accumulator<int64_t>, int64_t>::value, "int64.");
            static_assert(std::is_same<Kernels::Accumulator<std::complex<float>>,
                                       std::complex<float>>::value,
                          "complex<float>.");

            EXPECT_TRUE((std::is_same<Kernels::Accumulator<double>, double>::value));
        }

        TEST(Unit_Accumulator, AFloatSumOfHalfTermsKeepsWhatAHalfSumLoses)
        {
            // 2048 is where HALF's step grows to 2, so adding 1 to it in HALF is lost.
            const sycl::half big(2048.0f);
            const sycl::half one(1.0f);

            const sycl::half narrow = static_cast<sycl::half>(big + one);
            const Kernels::Accumulator<sycl::half> wide =
                static_cast<float>(big) + static_cast<float>(one);

            EXPECT_EQ(static_cast<float>(narrow), 2048.0f);
            EXPECT_EQ(wide, 2049.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
