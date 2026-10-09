/**************************************************************************/
/* test_blas_scalar.cpp                                                   */
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
 * @file test_blas_scalar.cpp
 * @brief Tests BlasScalar: what converts to it, what it narrows to and what it reports.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/operands/scalar.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The largest finite HALF value, which binary32 holds exactly.
        constexpr double K_LARGEST_HALF = 65504.0;

        /// @brief A value HALF cannot hold exactly; its neighbours in HALF are 2^-14 apart.
        constexpr double K_INEXACT_IN_HALF = 0.1;

        /// @brief The distance between the HALF values around K_INEXACT_IN_HALF.
        constexpr double K_HALF_SPACING_AT_INEXACT = 1.0 / 16384.0;

        /// @brief A value below half of the smallest positive HALF, 2^-24.
        constexpr double K_BELOW_HALF_RANGE = 1e-8;

        /// @brief The smallest positive HALF value.
        constexpr double K_SMALLEST_HALF = 1.0 / 16777216.0;

        TEST(Unit_BlasScalar, RealAndComplexDoubleConvertImplicitly)
        {
            static_assert(std::is_convertible_v<float, BlasScalar>, "float converts.");
            static_assert(std::is_convertible_v<int, BlasScalar>, "int converts.");
            static_assert(std::is_convertible_v<double, BlasScalar>, "double converts.");
            static_assert(std::is_convertible_v<std::complex<double>, BlasScalar>,
                          "std::complex<double> converts.");

            EXPECT_TRUE((std::is_convertible_v<float, BlasScalar>));
            EXPECT_TRUE((std::is_convertible_v<int, BlasScalar>));
            EXPECT_TRUE((std::is_convertible_v<double, BlasScalar>));
            EXPECT_TRUE((std::is_convertible_v<std::complex<double>, BlasScalar>));
        }

        TEST(Unit_BlasScalar, ComplexFloatDoesNotConvertImplicitly)
        {
            EXPECT_FALSE((std::is_convertible_v<std::complex<float>, BlasScalar>));
        }

        TEST(Unit_BlasScalar, ARealValueKeepsItsValueAndAZeroImaginaryPart)
        {
            const BlasScalar from_float = 1.5f;
            const BlasScalar from_int = 3;
            const BlasScalar from_double = -2.25;

            EXPECT_EQ(from_float.real(), 1.5);
            EXPECT_EQ(from_float.imag(), 0.0);
            EXPECT_EQ(from_int.real(), 3.0);
            EXPECT_EQ(from_int.imag(), 0.0);
            EXPECT_EQ(from_double.real(), -2.25);
            EXPECT_EQ(from_double.imag(), 0.0);
        }

        TEST(Unit_BlasScalar, AComplexValueKeepsBothParts)
        {
            const BlasScalar scalar = std::complex<double>(1.5, -2.25);

            EXPECT_EQ(scalar.real(), 1.5);
            EXPECT_EQ(scalar.imag(), -2.25);
        }

        TEST(Unit_BlasScalar, AsReturnsTheValueInEachRealType)
        {
            const BlasScalar scalar = 1.5;

            EXPECT_EQ(scalar.as<float>(), 1.5f);
            EXPECT_EQ(scalar.as<double>(), 1.5);
            EXPECT_EQ(static_cast<float>(scalar.as<sycl::half>()), 1.5f);
        }

        TEST(Unit_BlasScalar, AsRoundsADoubleOnceToFloat)
        {
            const BlasScalar scalar = K_INEXACT_IN_HALF;

            EXPECT_EQ(scalar.as<float>(), static_cast<float>(K_INEXACT_IN_HALF));
            EXPECT_EQ(scalar.as<double>(), K_INEXACT_IN_HALF);
        }

        TEST(Unit_BlasScalar, AsReturnsBothPartsInEachComplexType)
        {
            const BlasScalar scalar = std::complex<double>(1.5, -2.25);

            EXPECT_EQ(scalar.as<std::complex<float>>(), std::complex<float>(1.5f, -2.25f));
            EXPECT_EQ(scalar.as<std::complex<double>>(), std::complex<double>(1.5, -2.25));
        }

        TEST(Unit_BlasScalar, AsGivesARealScalarAZeroImaginaryPartInAComplexType)
        {
            const BlasScalar scalar = 1.5;

            EXPECT_EQ(scalar.as<std::complex<float>>(), std::complex<float>(1.5f, 0.0f));
            EXPECT_EQ(scalar.as<std::complex<double>>(), std::complex<double>(1.5, 0.0));
        }

        TEST(Unit_BlasScalar, AsHalfRoundsThroughFloat)
        {
            for (const double value : {K_INEXACT_IN_HALF, K_LARGEST_HALF, K_BELOW_HALF_RANGE})
            {
                const BlasScalar scalar = value;
                const sycl::half through_float = sycl::half(static_cast<float>(value));

                EXPECT_EQ(static_cast<float>(scalar.as<sycl::half>()),
                          static_cast<float>(through_float))
                    << "value " << value;
            }
        }

        TEST(Unit_BlasScalar, AsHalfKeepsTheLargestHalfExactly)
        {
            const BlasScalar scalar = K_LARGEST_HALF;

            EXPECT_EQ(static_cast<double>(static_cast<float>(scalar.as<sycl::half>())),
                      K_LARGEST_HALF);
        }

        TEST(Unit_BlasScalar, AsHalfLandsOnANeighbourOfAnInexactValue)
        {
            const BlasScalar scalar = K_INEXACT_IN_HALF;
            const double rounded = static_cast<double>(static_cast<float>(scalar.as<sycl::half>()));

            EXPECT_LE(std::fabs(rounded - K_INEXACT_IN_HALF), K_HALF_SPACING_AT_INEXACT);
        }

        TEST(Unit_BlasScalar, AsHalfTakesAValueBelowTheRangeToZeroOrTheSmallestHalf)
        {
            const BlasScalar scalar = K_BELOW_HALF_RANGE;
            const double rounded = static_cast<double>(static_cast<float>(scalar.as<sycl::half>()));

            EXPECT_GE(rounded, 0.0);
            EXPECT_LE(rounded, K_SMALLEST_HALF);
        }

        TEST(Unit_BlasScalar, IsRealHoldsForAZeroImaginaryPart)
        {
            EXPECT_TRUE(BlasScalar(2.0).is_real());
            EXPECT_TRUE(BlasScalar(std::complex<double>(2.0, 0.0)).is_real());
            EXPECT_TRUE(BlasScalar(std::complex<double>(2.0, -0.0)).is_real());
            EXPECT_FALSE(BlasScalar(std::complex<double>(2.0, 1e-300)).is_real());
        }

        TEST(Unit_BlasScalar, IsRealFailsForANanImaginaryPart)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_FALSE(BlasScalar(std::complex<double>(1.0, nan)).is_real());
        }

        TEST(Unit_BlasScalar, IsZeroHoldsOnlyWhenBothPartsAreZero)
        {
            EXPECT_TRUE(BlasScalar(0.0).is_zero());
            EXPECT_TRUE(BlasScalar(-0.0).is_zero());
            EXPECT_TRUE(BlasScalar(std::complex<double>(0.0, 0.0)).is_zero());
            EXPECT_FALSE(BlasScalar(1e-300).is_zero());
            EXPECT_FALSE(BlasScalar(std::complex<double>(0.0, 1e-300)).is_zero());
        }

        TEST(Unit_BlasScalar, IsZeroFailsForANanPart)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_FALSE(BlasScalar(nan).is_zero());
            EXPECT_FALSE(BlasScalar(std::complex<double>(0.0, nan)).is_zero());
        }
    } // namespace Test
} // namespace SushiBLAS
