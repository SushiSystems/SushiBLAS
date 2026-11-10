/**************************************************************************/
/* test_complex_arithmetic.cpp                                            */
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
 * @file test_complex_arithmetic.cpp
 * @brief Tests the conjugation, product and quotient bricks on real and complex values.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Divides the largest finite value into a divisor whose squared parts overflow.
        constexpr double K_LARGE_DIVISOR_FRACTION = 1e8;

        /**
         * @brief Ratio limit of a quotient against std::complex division, in units of u times the
         *        quotient's modulus: 6 epsilons = 12 u, from 3.2 epsilons for the error of
         *        Smith's algorithm plus 2.5 for that of the textbook formula, rounded up.
         */
        constexpr double K_DIVIDE_THRESHOLD = 12.0;

        /// @brief Ratio limit of a quotient that rounds once: one epsilon is two u.
        constexpr double K_ONE_ROUNDING_THRESHOLD = 2.0;

        /// @brief Two complex operands, written as doubles and cast to the component type.
        struct OperandRow
        {
            double a_real;
            double a_imag;
            double b_real;
            double b_imag;
        };

        /// @brief Operand pairs for the comparisons; no b_real and no divisor is zero.
        constexpr std::array<OperandRow, 6> K_OPERANDS{{
            {1.5, -2.25, 0.5, 4.0},
            {-3.75, 0.125, -2.0, -1.5},
            {7.1, -0.3, 2.9, 5.3},
            {0.01, 8.0, -6.2, 0.7},
            {-4.4, -4.4, 3.3, -1.1},
            {2.0, 0.5, -1.5, -3.0},
        }};

        namespace
        {
            /// @brief Operands whose complex product is exact in float, with that product.
            struct ProductRow
            {
                OperandRow operands;
                double product_real;
                double product_imag;
            };
        } // namespace

        /// @brief Complex products worked out by hand; every step is exact in float.
        constexpr std::array<ProductRow, 5> K_PRODUCTS{{
            {{1.5, -2.25, 0.5, 4.0}, 9.75, 4.875},
            {{-3.75, 0.125, -2.0, -1.5}, 7.6875, 5.375},
            {{2.0, 0.0, 0.0, -3.0}, 0.0, -6.0},
            {{1.0, 1.0, 1.0, -1.0}, 2.0, 0.0},
            {{-0.5, 6.0, 8.0, -0.25}, -2.5, 48.125},
        }};

        /// @brief A real factor times a complex value, with the exact product.
        struct ScaleRow
        {
            double factor;
            double real;
            double imag;
            double product_real;
            double product_imag;
        };

        /// @brief Real-by-complex products worked out by hand.
        constexpr std::array<ScaleRow, 3> K_SCALES{{
            {0.5, 3.0, -4.0, 1.5, -2.0},
            {-2.0, 1.25, 0.75, -2.5, -1.5},
            {8.0, -0.125, 2.0, -1.0, 16.0},
        }};

        namespace
        {
            /// @brief Two reals with their exact product and quotient.
            struct RealRow
            {
                double a;
                double b;
                double product;
                double quotient;
            };
        } // namespace

        /// @brief Real products and quotients worked out by hand.
        constexpr std::array<RealRow, 4> K_REALS{{
            {6.0, 1.5, 9.0, 4.0},
            {-2.5, 0.5, -1.25, -5.0},
            {7.0, -0.25, -1.75, -28.0},
            {0.75, 3.0, 2.25, 0.25},
        }};

        /// @brief Returns a name that differs per overload, so a test can read a type from a value.
        [[nodiscard]] inline const char* type_tag(float)
        {
            return "float";
        }

        /// @brief Returns a name that differs per overload, so a test can read a type from a value.
        [[nodiscard]] inline const char* type_tag(double)
        {
            return "double";
        }

        /// @brief Refuses every other type, so a converted argument cannot pass for the exact one.
        template<typename T>
        const char* type_tag(T) = delete;

        /**
         * @brief Builds real and complex values of component type R from the rows above.
         */
        template<typename R>
        class Small_ComplexArithmetic : public ::testing::Test
        {
            protected:
                /** @brief Returns the first operand of a row. */
                [[nodiscard]] static std::complex<R> first(const OperandRow& row) noexcept
                {
                    return std::complex<R>(static_cast<R>(row.a_real), static_cast<R>(row.a_imag));
                }

                /** @brief Returns the second operand of a row. */
                [[nodiscard]] static std::complex<R> second(const OperandRow& row) noexcept
                {
                    return std::complex<R>(static_cast<R>(row.b_real), static_cast<R>(row.b_imag));
                }

                /** @brief Calls check(a, b) with the two operands of every row of K_OPERANDS. */
                template<typename Check>
                static void for_each_operand_pair(Check check)
                {
                    for (const OperandRow& row : K_OPERANDS)
                    {
                        SCOPED_TRACE(::testing::Message() << "operands (" << row.a_real << ", "
                                                          << row.a_imag << ") and (" << row.b_real
                                                          << ", " << row.b_imag << ")");
                        check(first(row), second(row));
                    }
                }

                /** @brief Expects each part of a quotient to be within threshold units of u times
                 *         the modulus of the expected quotient. */
                static void expect_quotient_near(const std::complex<R>& got,
                                                 const std::complex<R>& expected, double threshold)
                {
                    const double bound = TestSupport::unit_roundoff<R>() *
                                         static_cast<double>(std::abs(expected));
                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(static_cast<double>(got.real()),
                                                   static_cast<double>(expected.real()), bound,
                                                   threshold)));
                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(static_cast<double>(got.imag()),
                                                   static_cast<double>(expected.imag()), bound,
                                                   threshold)));
                }

                /** @brief Returns a value whose square overflows R and whose double does not. */
                [[nodiscard]] static R large_divisor() noexcept
                {
                    return std::numeric_limits<R>::max() / static_cast<R>(K_LARGE_DIVISOR_FRACTION);
                }
        };

        using ComplexArithmeticTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_ComplexArithmetic, ComplexArithmeticTypes);

        TYPED_TEST(Small_ComplexArithmetic, RealOfNamesTheComponentType)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1), R(2));
            const Kernels::RealType<R> of_real = R(1);
            const Kernels::RealType<std::complex<R>> of_complex = value.real();

            EXPECT_STREQ(type_tag(of_real), type_tag(R(0)));
            EXPECT_STREQ(type_tag(of_complex), type_tag(R(0)));
        }

        TYPED_TEST(Small_ComplexArithmetic, RealMultiplyAndDivideAreTheBareOperators)
        {
            for (const RealRow& row : K_REALS)
            {
                const TypeParam a = static_cast<TypeParam>(row.a);
                const TypeParam b = static_cast<TypeParam>(row.b);

                EXPECT_EQ(Kernels::multiply(a, b), static_cast<TypeParam>(row.product));
                EXPECT_EQ(Kernels::divide(a, b), static_cast<TypeParam>(row.quotient));
            }
        }

        TYPED_TEST(Small_ComplexArithmetic, RealConjugationLeavesTheValue)
        {
            const TypeParam value = static_cast<TypeParam>(-2.5);

            EXPECT_EQ(Kernels::conjugate(value), value);
            EXPECT_EQ(Kernels::conjugate_if(true, value), value);
            EXPECT_EQ(Kernels::conjugate_if(false, value), value);
            EXPECT_EQ(Kernels::real_part(value), value);
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexConjugateNegatesTheImaginaryPart)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::conjugate(value), std::complex<R>(R(1.5), R(2.25)));
        }

        TYPED_TEST(Small_ComplexArithmetic, ConjugateIfConjugatesOnlyWhenEnabled)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::conjugate_if(true, value), std::complex<R>(R(1.5), R(2.25)));
            EXPECT_EQ(Kernels::conjugate_if(false, value), value);
        }

        TYPED_TEST(Small_ComplexArithmetic, RealPartZeroesTheImaginaryPart)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::real_part(value), std::complex<R>(R(1.5), R(0)));
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexMultiplyIsTheFourProductFormulaToTheBit)
        {
            using R = TypeParam;

            for (const ProductRow& row : K_PRODUCTS)
            {
                const std::complex<R> product = Kernels::multiply(
                    TestFixture::first(row.operands), TestFixture::second(row.operands));

                EXPECT_EQ(product.real(), static_cast<R>(row.product_real));
                EXPECT_EQ(product.imag(), static_cast<R>(row.product_imag));
            }
        }

        TYPED_TEST(Small_ComplexArithmetic, RealTimesComplexScalesEachPart)
        {
            using R = TypeParam;

            for (const ScaleRow& row : K_SCALES)
            {
                const std::complex<R> value(static_cast<R>(row.real), static_cast<R>(row.imag));
                const std::complex<R> product = Kernels::multiply(static_cast<R>(row.factor), value);

                EXPECT_EQ(product.real(), static_cast<R>(row.product_real));
                EXPECT_EQ(product.imag(), static_cast<R>(row.product_imag));
            }
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexDivideAgreesWithStdDivision)
        {
            using R = TypeParam;

            TestFixture::for_each_operand_pair(
                [](const std::complex<R>& a, const std::complex<R>& b)
                {
                    TestFixture::expect_quotient_near(Kernels::divide(a, b), a / b,
                                                      K_DIVIDE_THRESHOLD);
                });
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexDivideIsExactForARealDivisor)
        {
            using R = TypeParam;

            TestFixture::for_each_operand_pair(
                [](const std::complex<R>& a, const std::complex<R>& b)
                {
                    const R divisor = b.real();
                    const std::complex<R> quotient =
                        Kernels::divide(a, std::complex<R>(divisor, R(0)));

                    EXPECT_EQ(quotient.real(), a.real() / divisor);
                    EXPECT_EQ(quotient.imag(), a.imag() / divisor);
                });
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexDivideStaysFiniteForALargeDivisor)
        {
            using R = TypeParam;
            const R large = TestFixture::large_divisor();
            ASSERT_TRUE(std::isinf(large * large));

            const std::complex<R> quotient =
                Kernels::divide(std::complex<R>(R(1), R(1)), std::complex<R>(large, large));
            const double expected = 1.0 / static_cast<double>(large);
            const double bound = TestSupport::unit_roundoff<R>() * expected;

            EXPECT_TRUE(std::isfinite(quotient.real()));
            EXPECT_TRUE(std::isfinite(quotient.imag()));
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                static_cast<double>(quotient.real()), expected, bound, K_ONE_ROUNDING_THRESHOLD)));
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                static_cast<double>(quotient.imag()), 0.0, bound, K_ONE_ROUNDING_THRESHOLD)));
        }

        TYPED_TEST(Small_ComplexArithmetic, ComplexDivideByZeroGivesNanParts)
        {
            using R = TypeParam;

            const std::complex<R> quotient =
                Kernels::divide(std::complex<R>(R(1), R(2)), std::complex<R>(R(0), R(0)));

            EXPECT_TRUE(std::isnan(quotient.real()));
            EXPECT_TRUE(std::isnan(quotient.imag()));
        }

        TYPED_TEST(Small_ComplexArithmetic, SafeConjugateReturnsWhatTheBrickReturns)
        {
            using R = TypeParam;

            TestFixture::for_each_operand_pair(
                [](const std::complex<R>& a, const std::complex<R>&)
                {
                    EXPECT_EQ(Dispatch::safe_conj(a), Kernels::conjugate(a));
                    EXPECT_EQ(Dispatch::safe_conj(a.real()), a.real());
                });
        }

        TYPED_TEST(Small_ComplexArithmetic, SafeMultiplyReturnsWhatTheBrickReturns)
        {
            using R = TypeParam;

            TestFixture::for_each_operand_pair(
                [](const std::complex<R>& a, const std::complex<R>& b)
                {
                    EXPECT_EQ(Dispatch::safe_mul(a, b), Kernels::multiply(a, b));
                    EXPECT_EQ(Dispatch::safe_mul(a.real(), b.real()), a.real() * b.real());
                });
        }

        TYPED_TEST(Small_ComplexArithmetic, SafeDivideReturnsWhatTheBrickReturns)
        {
            using R = TypeParam;

            TestFixture::for_each_operand_pair(
                [](const std::complex<R>& a, const std::complex<R>& b)
                {
                    EXPECT_EQ(Dispatch::safe_div(a, b), Kernels::divide(a, b));
                    EXPECT_EQ(Dispatch::safe_div(a.real(), b.real()), a.real() / b.real());
                });
        }
    } // namespace Test
} // namespace SushiBLAS
