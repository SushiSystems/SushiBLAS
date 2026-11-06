/**************************************************************************/
/* test_reference_accumulator.cpp                                         */
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
 * @file test_reference_accumulator.cpp
 * @brief Tests the running sum a reference kernel accumulates in.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The first integer binary32 cannot hold: 2^24 + 1.
        constexpr std::int64_t K_FIRST_INTEGER_FLOAT_CANNOT_HOLD = 16777217;

        /// @brief A double whose neighbours are 2 apart, so adding 1 to it is rounded away.
        constexpr double K_LARGE_TERM = 1e16;

        /// @brief A term the plain sum of K_LARGE_TERM, this and -K_LARGE_TERM loses.
        constexpr double K_SMALL_TERM = 1.0;

        /// @brief Returns a float reference sum of count terms of 1.
        static Kernels::ReferenceSum<float> sum_of_ones(std::int64_t count)
        {
            Kernels::ReferenceSum<float> sum;
            for (std::int64_t i = 0; i < count; ++i)
            {
                sum.add(1.0);
            }
            return sum;
        }

        TEST(Small_ReferenceAccumulator, AFloatSumHoldsAnIntegerFloatCannot)
        {
            EXPECT_EQ(sum_of_ones(K_FIRST_INTEGER_FLOAT_CANNOT_HOLD).value(),
                      static_cast<double>(K_FIRST_INTEGER_FLOAT_CANNOT_HOLD));
        }

        TEST(Small_ReferenceAccumulator, ADoubleSumKeepsWhatAnAdditionRoundsOff)
        {
            Kernels::ReferenceSum<double> sum;
            sum.add(K_LARGE_TERM);
            sum.add(K_SMALL_TERM);
            sum.add(-K_LARGE_TERM);

            EXPECT_EQ(sum.value(), K_SMALL_TERM);
        }

        TEST(Small_ReferenceAccumulator, AFloatSumOfDoublesIsNotCompensated)
        {
            Kernels::ReferenceSum<float> sum;
            sum.add(K_LARGE_TERM);
            sum.add(K_SMALL_TERM);
            sum.add(-K_LARGE_TERM);

            // The oracle is a plain left-to-right double sum of the same terms; the exact sum is K_SMALL_TERM.
            const double plain = (K_LARGE_TERM + K_SMALL_TERM) - K_LARGE_TERM;

            EXPECT_EQ(sum.value(), plain);
        }

        TEST(Small_ReferenceAccumulator, AnInfiniteTermGivesThatInfinityInBothForms)
        {
            const double infinity = std::numeric_limits<double>::infinity();

            for (const double term : {infinity, -infinity})
            {
                Kernels::ReferenceSum<float> plain;
                plain.add(1.0);
                plain.add(term);
                plain.add(2.0);

                Kernels::ReferenceSum<double> compensated;
                compensated.add(1.0);
                compensated.add(term);
                compensated.add(2.0);

                EXPECT_EQ(plain.value(), term);
                EXPECT_EQ(compensated.value(), term);
            }
        }

        TEST(Small_ReferenceAccumulator, ANanTermGivesNanInBothForms)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            Kernels::ReferenceSum<float> plain;
            plain.add(1.0);
            plain.add(nan);
            plain.add(2.0);

            Kernels::ReferenceSum<double> compensated;
            compensated.add(1.0);
            compensated.add(nan);
            compensated.add(2.0);

            EXPECT_TRUE(std::isnan(plain.value()));
            EXPECT_TRUE(std::isnan(compensated.value()));
        }

        TEST(Small_ReferenceAccumulator, OpposedInfinitiesGiveNanInBothForms)
        {
            const double infinity = std::numeric_limits<double>::infinity();

            Kernels::ReferenceSum<float> plain;
            plain.add(infinity);
            plain.add(-infinity);

            Kernels::ReferenceSum<double> compensated;
            compensated.add(infinity);
            compensated.add(-infinity);

            EXPECT_TRUE(std::isnan(plain.value()));
            EXPECT_TRUE(std::isnan(compensated.value()));
        }

        TEST(Small_ReferenceAccumulator, AComplexSumCompensatesEachPartOnItsOwn)
        {
            Kernels::ReferenceSum<std::complex<double>> sum;
            sum.add(std::complex<double>(K_LARGE_TERM, K_SMALL_TERM));
            sum.add(std::complex<double>(K_SMALL_TERM, K_LARGE_TERM));
            sum.add(std::complex<double>(-K_LARGE_TERM, -K_LARGE_TERM));

            EXPECT_EQ(sum.value(), std::complex<double>(K_SMALL_TERM, K_SMALL_TERM));
        }

        TEST(Small_ReferenceAccumulator, AComplexFloatSumAddsEachPartInDouble)
        {
            Kernels::ReferenceSum<std::complex<float>> sum;
            sum.add(std::complex<double>(K_LARGE_TERM, 0.25));
            sum.add(std::complex<double>(K_SMALL_TERM, 0.5));
            sum.add(std::complex<double>(-K_LARGE_TERM, 0.125));

            // K_LARGE_TERM + K_SMALL_TERM ties between doubles 2 apart and rounds back, so the real part is 0.
            EXPECT_EQ(sum.value(), std::complex<double>(0.0, 0.875));
        }

        TEST(Small_ReferenceAccumulator, AnInfinitePartOfAComplexSumLeavesTheOtherPartFinite)
        {
            const double infinity = std::numeric_limits<double>::infinity();

            Kernels::ReferenceSum<std::complex<double>> sum;
            sum.add(std::complex<double>(1.0, 2.0));
            sum.add(std::complex<double>(infinity, 3.0));

            EXPECT_EQ(sum.value().real(), infinity);
            EXPECT_EQ(sum.value().imag(), 5.0);
        }
    } // namespace Test
} // namespace SushiBLAS
