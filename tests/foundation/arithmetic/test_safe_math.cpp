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
 * @file test_safe_math.cpp
 * @brief Tests the real and complex safe functions of Dispatch against the standard library.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Relative tolerance: 64 double roundings per unit of magnitude.
        constexpr double K_RELATIVE =
            TestSupport::K_DEFAULT_THRESHOLD * 64.0 * TestSupport::unit_roundoff<double>();

        /// @brief Absolute tolerance: one double rounding of a value of order one.
        constexpr double K_ABSOLUTE =
            TestSupport::K_DEFAULT_THRESHOLD * TestSupport::unit_roundoff<double>();

        /**
         * @brief Expects two complex values to agree in both parts within the file's tolerances.
         */
        static void expect_close(const std::complex<double>& got, const std::complex<double>& want)
        {
            EXPECT_NEAR(got.real(), want.real(),
                        K_ABSOLUTE + K_RELATIVE * std::abs(want.real()));
            EXPECT_NEAR(got.imag(), want.imag(),
                        K_ABSOLUTE + K_RELATIVE * std::abs(want.imag()));
        }

        /// @brief Names a row by its function, which gtest takes as the test's suffix.
        template<typename Row>
        static std::string row_name(const ::testing::TestParamInfo<Row>& info)
        {
            return info.param.name;
        }

        namespace
        {
            /// @brief One real safe function, one input and the correctly rounded value.
            struct RealRow
            {
                const char* name;
                double (*function)(double);
                double input;
                double expected;
                double roundings;  ///< Multiples of K_RELATIVE allowed; zero demands the exact value.
            };
        } // namespace

        /// @brief Returns the real rows; the literals come from the correctly rounded values.
        static std::vector<RealRow> real_rows()
        {
            return {
                {"exp", [](double x) { return Dispatch::safe_exp(x); }, 1.25, 3.4903429574618414, 4.0},
                {"tanh", [](double x) { return Dispatch::safe_tanh(x); }, 0.75, 0.6351489523872873, 1.0},
                {"erf", [](double x) { return Dispatch::safe_erf(x); }, 0.5, 0.5204998778130465, 1.0},
                {"log", [](double x) { return Dispatch::safe_log(x); }, 3.0, 1.0986122886681098, 2.0},
                {"sqrt", [](double x) { return Dispatch::safe_sqrt(x); }, 9.0, 3.0, 0.0},
                {"abs", [](double x) { return Dispatch::safe_abs(x); }, -4.0, 4.0, 0.0},
            };
        }

        /// @brief Runs one real row of the safe functions.
        class Small_SafeMathReal : public ::testing::TestWithParam<RealRow>
        {
        };

        TEST_P(Small_SafeMathReal, TheRealFunctionAgreesWithTheStandardLibrary)
        {
            const RealRow& row = GetParam();
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                row.function(row.input), row.expected, K_RELATIVE * row.roundings, 1.0)));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SafeMathReal, ::testing::ValuesIn(real_rows()),
                                 row_name<RealRow>);

        /// @brief One complex safe function, applied to 0.75 - 1.25i, with the value to expect.
        struct ComplexRow
        {
            const char* name;
            std::complex<double> (*function)(const std::complex<double>&);
            std::complex<double> expected;
        };

        /// @brief Input of every complex row.
        constexpr std::complex<double> K_COMPLEX_INPUT(0.75, -1.25);

        /// @brief Returns the complex rows; the literals come from the correctly rounded values.
        static std::vector<ComplexRow> complex_rows()
        {
            return {
                {"exp", [](const std::complex<double>& z) { return Dispatch::safe_exp(z); },
                 {0.6675374464291316, -2.009000454940949}},
                {"tanh", [](const std::complex<double>& z) { return Dispatch::safe_tanh(z); },
                 {1.3726075705337832, -0.38579595260975064}},
                {"log", [](const std::complex<double>& z) { return Dispatch::safe_log(z); },
                 {0.3768859011881901, -1.0303768265243125}},
                {"sqrt", [](const std::complex<double>& z) { return Dispatch::safe_sqrt(z); },
                 {1.050651696260784, -0.594868882070379}},
                {"abs", [](const std::complex<double>& z) { return Dispatch::safe_abs(z); },
                 {1.4577379737113252, 0.0}},
            };
        }

        /// @brief Runs one complex row of the safe functions.
        class Small_SafeMathComplex : public ::testing::TestWithParam<ComplexRow>
        {
        };

        TEST_P(Small_SafeMathComplex, TheComplexFunctionAgreesWithTheStandardLibrary)
        {
            const ComplexRow& row = GetParam();
            expect_close(row.function(K_COMPLEX_INPUT), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SafeMathComplex, ::testing::ValuesIn(complex_rows()),
                                 row_name<ComplexRow>);

        TEST(Small_SafeMath, ComplexSqrtTakesThePrincipalRootOnTheNegativeRealSide)
        {
            const std::complex<double> z(-4.0, 3.0);

            expect_close(Dispatch::safe_sqrt(z), std::sqrt(z));
            expect_close(Dispatch::safe_sqrt(std::complex<double>(-4.0, -3.0)),
                         std::sqrt(std::complex<double>(-4.0, -3.0)));
        }

        TEST(Small_SafeMath, ComplexSqrtOfZeroIsZero)
        {
            EXPECT_EQ(Dispatch::safe_sqrt(std::complex<double>(0.0, 0.0)),
                      std::complex<double>(0.0, 0.0));
        }

        TEST(Small_SafeMath, ComplexSqrtOfAnInfiniteImaginaryPartIsInfinite)
        {
            const double infinity = std::numeric_limits<double>::infinity();
            const std::complex<double> root =
                Dispatch::safe_sqrt(std::complex<double>(1.0, infinity));

            EXPECT_EQ(root.real(), infinity);
            EXPECT_EQ(root.imag(), infinity);
        }

        TEST(Small_SafeMath, ComplexTanhSaturatesForALargeRealPart)
        {
            const std::complex<double> got = Dispatch::safe_tanh(std::complex<double>(40.0, 0.5));

            EXPECT_EQ(got.real(), 1.0);
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(got.imag(), 0.0, K_ABSOLUTE, 1.0)));
        }

        TEST(Small_SafeMath, ComplexLogNearTheUnitCircleKeepsItsSmallModulus)
        {
            // log1p keeps the 1e-10 that log(hypot) rounds to a few ulps of 1.
            const double re = 1.0 + 1e-10;
            const double want = 0.5 * std::log1p((re - 1.0) * (re + 1.0));

            EXPECT_NEAR(Dispatch::safe_log(std::complex<double>(re, 0.0)).real(), want,
                        K_RELATIVE * want);
        }

        TEST(Small_SafeMath, ARealImaginaryPartIsKeptExact)
        {
            const std::complex<double> got = Dispatch::safe_exp(std::complex<double>(1.0, 0.0));

            EXPECT_EQ(got.imag(), 0.0);
            EXPECT_NEAR(got.real(), std::exp(1.0), K_RELATIVE * 3.0);
        }

        TEST(Small_SafeMath, TheSignTestsReadTheRealPart)
        {
            EXPECT_TRUE(Dispatch::greater_than_zero(2.0));
            EXPECT_FALSE(Dispatch::greater_than_zero(0.0));
            EXPECT_TRUE(Dispatch::less_than_zero(-1.0f));
            EXPECT_TRUE(Dispatch::greater_than_zero(std::complex<double>(1.0, -9.0)));
            EXPECT_TRUE(Dispatch::less_than_zero(std::complex<double>(-1.0, 9.0)));
        }

        TEST(Small_SafeMath, RealPartDropsTheImaginaryPart)
        {
            EXPECT_EQ(Dispatch::real_part(std::complex<float>(2.0f, 3.0f)),
                      std::complex<float>(2.0f, 0.0f));
            EXPECT_EQ(Dispatch::real_part(5.0), 5.0);
        }

        TEST(Small_SafeMath, MakeComplexSafeRestoresTheImaginaryPartOnlyForAComplexType)
        {
            EXPECT_EQ(Dispatch::make_complex_safe(1.5, std::complex<double>()),
                      std::complex<double>(1.5, 0.0));
            EXPECT_EQ(Dispatch::make_complex_safe(1.5, 0.0), 1.5);
        }

        TEST(Small_SafeMath, ConjNegatesTheImaginaryPart)
        {
            const std::complex<double> a(1.0, 2.0);

            EXPECT_EQ(Dispatch::safe_conj(a), std::complex<double>(1.0, -2.0));
        }

        TEST(Small_SafeMath, MulFollowsComplexArithmetic)
        {
            const std::complex<double> a(1.0, 2.0);
            const std::complex<double> b(3.0, -1.0);

            EXPECT_EQ(Dispatch::safe_mul(a, b), std::complex<double>(5.0, 5.0));
            EXPECT_EQ(Dispatch::safe_mul(2.0, 4.0), 8.0);
        }

        TEST(Small_SafeMath, DivFollowsComplexArithmetic)
        {
            const std::complex<double> a(1.0, 2.0);
            const std::complex<double> b(3.0, -1.0);

            expect_close(Dispatch::safe_div(a, b), std::complex<double>(0.1, 0.7));
            EXPECT_EQ(Dispatch::safe_div(9.0, 3.0), 3.0);
        }
    } // namespace Test
} // namespace SushiBLAS
