/**************************************************************************/
/* test_safe_math.cpp                                                     */
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
 * @file test_safe_math.cpp
 * @brief Tests the real and complex safe functions of Dispatch against the standard library.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/safe_math.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Relative tolerance of a double result against the standard library.
        constexpr double K_RELATIVE = 1e-13;

        /// @brief Absolute tolerance for a double component whose true value is near zero.
        constexpr double K_ABSOLUTE = 1e-15;

        /**
         * @brief Expects two complex values to agree in both parts within the file's tolerances.
         */
        void expect_close(const std::complex<double>& got, const std::complex<double>& want)
        {
            EXPECT_NEAR(got.real(), want.real(),
                        K_ABSOLUTE + K_RELATIVE * std::abs(want.real()));
            EXPECT_NEAR(got.imag(), want.imag(),
                        K_ABSOLUTE + K_RELATIVE * std::abs(want.imag()));
        }

        TEST(Unit_SafeMath, TheRealFunctionsAgreeWithTheStandardLibrary)
        {
            EXPECT_NEAR(Dispatch::safe_exp(1.25), std::exp(1.25), K_RELATIVE * 4.0);
            EXPECT_NEAR(Dispatch::safe_tanh(0.75), std::tanh(0.75), K_RELATIVE);
            EXPECT_NEAR(Dispatch::safe_erf(0.5), std::erf(0.5), K_RELATIVE);
            EXPECT_NEAR(Dispatch::safe_log(3.0), std::log(3.0), K_RELATIVE * 2.0);
            EXPECT_EQ(Dispatch::safe_sqrt(9.0), 3.0);
            EXPECT_EQ(Dispatch::safe_abs(-4.0), 4.0);
        }

        TEST(Unit_SafeMath, TheComplexFunctionsAgreeWithTheStandardLibrary)
        {
            const std::complex<double> z(0.75, -1.25);

            expect_close(Dispatch::safe_exp(z), std::exp(z));
            expect_close(Dispatch::safe_tanh(z), std::tanh(z));
            expect_close(Dispatch::safe_log(z), std::log(z));
            expect_close(Dispatch::safe_sqrt(z), std::sqrt(z));
            expect_close(Dispatch::safe_abs(z), std::complex<double>(std::abs(z), 0.0));
        }

        TEST(Unit_SafeMath, ComplexSqrtTakesThePrincipalRootOnTheNegativeRealSide)
        {
            const std::complex<double> z(-4.0, 3.0);

            expect_close(Dispatch::safe_sqrt(z), std::sqrt(z));
            expect_close(Dispatch::safe_sqrt(std::complex<double>(-4.0, -3.0)),
                         std::sqrt(std::complex<double>(-4.0, -3.0)));
        }

        TEST(Unit_SafeMath, ComplexSqrtOfZeroAndOfAnInfiniteImaginaryPart)
        {
            const double infinity = std::numeric_limits<double>::infinity();

            EXPECT_EQ(Dispatch::safe_sqrt(std::complex<double>(0.0, 0.0)),
                      std::complex<double>(0.0, 0.0));
            const std::complex<double> root =
                Dispatch::safe_sqrt(std::complex<double>(1.0, infinity));
            EXPECT_EQ(root.real(), infinity);
            EXPECT_EQ(root.imag(), infinity);
        }

        TEST(Unit_SafeMath, ComplexTanhSaturatesForALargeRealPart)
        {
            const std::complex<double> got = Dispatch::safe_tanh(std::complex<double>(40.0, 0.5));

            EXPECT_EQ(got.real(), 1.0);
            EXPECT_NEAR(got.imag(), 0.0, 1e-30);
        }

        TEST(Unit_SafeMath, ComplexLogNearTheUnitCircleKeepsItsSmallModulus)
        {
            // log1p keeps the 1e-10 that log(hypot) rounds to a few ulps of 1.
            const double re = 1.0 + 1e-10;
            const double want = 0.5 * std::log1p((re - 1.0) * (re + 1.0));

            EXPECT_NEAR(Dispatch::safe_log(std::complex<double>(re, 0.0)).real(), want,
                        K_RELATIVE * want);
        }

        TEST(Unit_SafeMath, ARealImaginaryPartIsKeptExact)
        {
            const std::complex<double> got = Dispatch::safe_exp(std::complex<double>(1.0, 0.0));

            EXPECT_EQ(got.imag(), 0.0);
            EXPECT_NEAR(got.real(), std::exp(1.0), K_RELATIVE * 3.0);
        }

        TEST(Unit_SafeMath, TheSignTestsReadTheRealPart)
        {
            EXPECT_TRUE(Dispatch::greater_than_zero(2.0));
            EXPECT_FALSE(Dispatch::greater_than_zero(0.0));
            EXPECT_TRUE(Dispatch::less_than_zero(-1.0f));
            EXPECT_TRUE(Dispatch::greater_than_zero(std::complex<double>(1.0, -9.0)));
            EXPECT_TRUE(Dispatch::less_than_zero(std::complex<double>(-1.0, 9.0)));
        }

        TEST(Unit_SafeMath, RealPartAndMakeComplexDropAndRestoreTheImaginaryPart)
        {
            EXPECT_EQ(Dispatch::real_part(std::complex<float>(2.0f, 3.0f)),
                      std::complex<float>(2.0f, 0.0f));
            EXPECT_EQ(Dispatch::real_part(5.0), 5.0);
            EXPECT_EQ(Dispatch::make_complex_safe(1.5, std::complex<double>()),
                      std::complex<double>(1.5, 0.0));
            EXPECT_EQ(Dispatch::make_complex_safe(1.5, 0.0), 1.5);
        }

        TEST(Unit_SafeMath, ConjMulAndDivFollowComplexArithmetic)
        {
            const std::complex<double> a(1.0, 2.0);
            const std::complex<double> b(3.0, -1.0);

            EXPECT_EQ(Dispatch::safe_conj(a), std::complex<double>(1.0, -2.0));
            EXPECT_EQ(Dispatch::safe_mul(a, b), std::complex<double>(5.0, 5.0));
            expect_close(Dispatch::safe_div(a, b), std::complex<double>(0.1, 0.7));
            EXPECT_EQ(Dispatch::safe_mul(2.0, 4.0), 8.0);
            EXPECT_EQ(Dispatch::safe_div(9.0, 3.0), 3.0);
        }
    } // namespace Test
} // namespace SushiBLAS
