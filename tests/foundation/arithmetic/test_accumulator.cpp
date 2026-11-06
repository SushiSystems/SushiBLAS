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
        static_assert(std::is_same<Kernels::Accumulator<sycl::half>, float>::value,
                      "HALF accumulates in FLOAT32.");
        static_assert(std::is_same<Kernels::AccumulatorOf<sycl::half>::type, float>::value,
                      "AccumulatorOf names the same type as Accumulator.");
        static_assert(std::is_same<Kernels::Accumulator<float>, float>::value, "float.");
        static_assert(std::is_same<Kernels::Accumulator<double>, double>::value, "double.");
        static_assert(std::is_same<Kernels::Accumulator<int32_t>, int32_t>::value, "int32.");
        static_assert(std::is_same<Kernels::Accumulator<int64_t>, int64_t>::value, "int64.");
        static_assert(std::is_same<Kernels::Accumulator<std::complex<float>>,
                                   std::complex<float>>::value,
                      "complex<float>.");

        TEST(Small_Accumulator, AFloatSumOfHalfTermsKeepsWhatAHalfSumLoses)
        {
            // 2048 is where HALF's step grows to 2; the float accumulator keeps the 1.
            const sycl::half big(2048.0f);
            const sycl::half one(1.0f);

            const Kernels::Accumulator<sycl::half> wide =
                static_cast<float>(big) + static_cast<float>(one);

            EXPECT_EQ(wide, 2049.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
