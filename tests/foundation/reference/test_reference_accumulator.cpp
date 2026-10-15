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
 * @file test_reference_accumulator.cpp
 * @brief Tests the type and the running sum a reference kernel accumulates in.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
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

        TEST(Medium_ReferenceAccumulator, RealTypesAccumulateInDouble)
        {
            static_assert(std::is_same_v<Kernels::ReferenceAccumulator<sycl::half>, double>,
                          "HALF is summed in double.");
            static_assert(std::is_same_v<Kernels::ReferenceAccumulator<float>, double>,
                          "FLOAT32 is summed in double.");
            static_assert(std::is_same_v<Kernels::ReferenceAccumulator<double>, double>,
                          "FLOAT64 is summed in double.");
            SUCCEED();
        }

        TEST(Medium_ReferenceAccumulator, ComplexTypesAccumulateInComplexDouble)
        {
            static_assert(std::is_same_v<Kernels::ReferenceAccumulator<std::complex<float>>,
                                         std::complex<double>>,
                          "COMPLEX32 is summed in complex double.");
            static_assert(std::is_same_v<Kernels::ReferenceAccumulator<std::complex<double>>,
                                         std::complex<double>>,
                          "COMPLEX64 is summed in complex double.");
            SUCCEED();
        }

        TEST(Medium_ReferenceAccumulator, OnlyTheDoubleTypesAreCompensated)
        {
            EXPECT_FALSE(Kernels::ReferenceSum<sycl::half>::K_IS_COMPENSATED);
            EXPECT_FALSE(Kernels::ReferenceSum<float>::K_IS_COMPENSATED);
            EXPECT_TRUE(Kernels::ReferenceSum<double>::K_IS_COMPENSATED);
            EXPECT_FALSE(Kernels::ReferenceSum<std::complex<float>>::K_IS_COMPENSATED);
            EXPECT_TRUE(Kernels::ReferenceSum<std::complex<double>>::K_IS_COMPENSATED);
        }

        TEST(Medium_ReferenceAccumulator, AnEmptySumIsZero)
        {
            EXPECT_EQ(Kernels::ReferenceSum<float>().value(), 0.0);
            EXPECT_EQ(Kernels::ReferenceSum<double>().value(), 0.0);
            EXPECT_EQ(Kernels::ReferenceSum<std::complex<float>>().value(),
                      std::complex<double>(0.0, 0.0));
            EXPECT_EQ(Kernels::ReferenceSum<std::complex<double>>().value(),
                      std::complex<double>(0.0, 0.0));
        }

        TEST(Medium_ReferenceAccumulator, AFloatSumHoldsAnIntegerFloatCannot)
        {
            Kernels::ReferenceSum<float> sum;
            float plain = 0.0f;
            for (std::int64_t i = 0; i < K_FIRST_INTEGER_FLOAT_CANNOT_HOLD; ++i)
            {
                sum.add(1.0);
                plain += 1.0f;
            }

            EXPECT_EQ(sum.value(), static_cast<double>(K_FIRST_INTEGER_FLOAT_CANNOT_HOLD));
            EXPECT_NE(static_cast<double>(plain),
                      static_cast<double>(K_FIRST_INTEGER_FLOAT_CANNOT_HOLD));
        }

        TEST(Medium_ReferenceAccumulator, ADoubleSumKeepsWhatAnAdditionRoundsOff)
        {
            Kernels::ReferenceSum<double> sum;
            sum.add(K_LARGE_TERM);
            sum.add(K_SMALL_TERM);
            sum.add(-K_LARGE_TERM);

            double plain = K_LARGE_TERM;
            plain += K_SMALL_TERM;
            plain -= K_LARGE_TERM;

            EXPECT_EQ(sum.value(), K_SMALL_TERM);
            EXPECT_EQ(plain, 0.0);
        }

        TEST(Medium_ReferenceAccumulator, AFloatSumOfDoublesIsNotCompensated)
        {
            Kernels::ReferenceSum<float> sum;
            sum.add(K_LARGE_TERM);
            sum.add(K_SMALL_TERM);
            sum.add(-K_LARGE_TERM);

            EXPECT_EQ(sum.value(), 0.0);
        }

        TEST(Medium_ReferenceAccumulator, AnInfiniteTermGivesThatInfinityInBothForms)
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

        TEST(Medium_ReferenceAccumulator, ANanTermGivesNanInBothForms)
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

        TEST(Medium_ReferenceAccumulator, OpposedInfinitiesGiveNanInBothForms)
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

        TEST(Medium_ReferenceAccumulator, AComplexSumCompensatesEachPartOnItsOwn)
        {
            Kernels::ReferenceSum<std::complex<double>> sum;
            sum.add(std::complex<double>(K_LARGE_TERM, K_SMALL_TERM));
            sum.add(std::complex<double>(K_SMALL_TERM, K_LARGE_TERM));
            sum.add(std::complex<double>(-K_LARGE_TERM, -K_LARGE_TERM));

            EXPECT_EQ(sum.value(), std::complex<double>(K_SMALL_TERM, K_SMALL_TERM));
        }

        TEST(Medium_ReferenceAccumulator, AComplexFloatSumAddsEachPartInDouble)
        {
            Kernels::ReferenceSum<std::complex<float>> sum;
            sum.add(std::complex<double>(K_LARGE_TERM, 0.25));
            sum.add(std::complex<double>(K_SMALL_TERM, 0.5));
            sum.add(std::complex<double>(-K_LARGE_TERM, 0.125));

            EXPECT_EQ(sum.value(), std::complex<double>(0.0, 0.875));
        }

        TEST(Medium_ReferenceAccumulator, AnInfinitePartOfAComplexSumLeavesTheOtherPartFinite)
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
