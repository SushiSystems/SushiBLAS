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
 * @file test_complex_arithmetic.cpp
 * @brief Tests the conjugation, product and quotient bricks on real and complex values.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Seed of every random draw in this file.
        constexpr std::uint32_t K_ARITHMETIC_SEED = 20261007u;

        /// @brief Random pairs each comparison runs over.
        constexpr int K_PAIR_COUNT = 1000;

        /// @brief Largest magnitude of a drawn component.
        constexpr double K_DRAW_RANGE = 8.0;

        /**
         * @brief Allowed distance from std::complex division, in epsilons of the quotient: 3.2
         *        for the error of Smith's algorithm plus 2.5 for that of the textbook formula.
         */
        constexpr double K_DIVIDE_EPSILONS = 6.0;

        /// @brief Allowed distance of a quotient that rounds once, in epsilons of the quotient.
        constexpr double K_ONE_ROUNDING_EPSILONS = 1.0;

        /// @brief Divides the largest finite value into a divisor whose squared parts overflow.
        constexpr double K_LARGE_DIVISOR_FRACTION = 1e8;

        /**
         * @brief Draws real and complex values of component type R from a fixed seed.
         */
        template<typename R>
        class Unit_ComplexArithmetic : public ::testing::Test
        {
            protected:
                /** @brief Returns the next real value, in [-K_DRAW_RANGE, K_DRAW_RANGE]. */
                [[nodiscard]] R draw_real()
                {
                    return static_cast<R>(distribution_(generator_));
                }

                /** @brief Returns the next complex value, each part drawn as draw_real does. */
                [[nodiscard]] std::complex<R> draw_complex()
                {
                    const R real = draw_real();
                    const R imaginary = draw_real();
                    return std::complex<R>(real, imaginary);
                }

                /** @brief Returns a value whose square overflows R and whose double does not. */
                [[nodiscard]] static R large_divisor() noexcept
                {
                    return std::numeric_limits<R>::max() / static_cast<R>(K_LARGE_DIVISOR_FRACTION);
                }

            private:
                /// @brief The generator every draw advances.
                std::mt19937 generator_{K_ARITHMETIC_SEED};

                /// @brief The range a component is drawn from.
                std::uniform_real_distribution<double> distribution_{-K_DRAW_RANGE, K_DRAW_RANGE};
        };

        using ComplexArithmeticTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_ComplexArithmetic, ComplexArithmeticTypes);

        TYPED_TEST(Unit_ComplexArithmetic, RealOfNamesTheComponentType)
        {
            using R = TypeParam;

            static_assert(std::is_same_v<Kernels::RealType<R>, R>, "A real type is its own.");
            static_assert(std::is_same_v<Kernels::RealType<std::complex<R>>, R>,
                          "A complex type names its component.");
            SUCCEED();
        }

        TYPED_TEST(Unit_ComplexArithmetic, RealMultiplyAndDivideAreTheBareOperators)
        {
            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const TypeParam a = this->draw_real();
                const TypeParam b = this->draw_real();

                EXPECT_EQ(Kernels::multiply(a, b), a * b);
                EXPECT_EQ(Kernels::divide(a, b), a / b);
            }
        }

        TYPED_TEST(Unit_ComplexArithmetic, RealConjugationLeavesTheValue)
        {
            const TypeParam value = static_cast<TypeParam>(-2.5);

            EXPECT_EQ(Kernels::conjugate(value), value);
            EXPECT_EQ(Kernels::conjugate_if(true, value), value);
            EXPECT_EQ(Kernels::conjugate_if(false, value), value);
            EXPECT_EQ(Kernels::real_part(value), value);
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexConjugateNegatesTheImaginaryPart)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::conjugate(value), std::complex<R>(R(1.5), R(2.25)));
        }

        TYPED_TEST(Unit_ComplexArithmetic, ConjugateIfConjugatesOnlyWhenEnabled)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::conjugate_if(true, value), std::complex<R>(R(1.5), R(2.25)));
            EXPECT_EQ(Kernels::conjugate_if(false, value), value);
        }

        TYPED_TEST(Unit_ComplexArithmetic, RealPartZeroesTheImaginaryPart)
        {
            using R = TypeParam;
            const std::complex<R> value(R(1.5), R(-2.25));

            EXPECT_EQ(Kernels::real_part(value), std::complex<R>(R(1.5), R(0)));
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexMultiplyIsTheFourProductFormulaToTheBit)
        {
            using R = TypeParam;

            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const std::complex<R> a = this->draw_complex();
                const std::complex<R> b = this->draw_complex();
                const std::complex<R> product = Kernels::multiply(a, b);

                EXPECT_EQ(product.real(), a.real() * b.real() - a.imag() * b.imag());
                EXPECT_EQ(product.imag(), a.real() * b.imag() + a.imag() * b.real());
            }
        }

        TYPED_TEST(Unit_ComplexArithmetic, RealTimesComplexScalesEachPart)
        {
            using R = TypeParam;

            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const R a = this->draw_real();
                const std::complex<R> b = this->draw_complex();
                const std::complex<R> product = Kernels::multiply(a, b);

                EXPECT_EQ(product.real(), a * b.real());
                EXPECT_EQ(product.imag(), a * b.imag());
            }
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexDivideAgreesWithStdDivision)
        {
            using R = TypeParam;
            const double epsilon = static_cast<double>(std::numeric_limits<R>::epsilon());

            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const std::complex<R> a = this->draw_complex();
                const std::complex<R> b = this->draw_complex();
                if (b == std::complex<R>(R(0), R(0)))
                    continue;

                const std::complex<R> quotient = Kernels::divide(a, b);
                const std::complex<R> expected = a / b;
                const double bound =
                    K_DIVIDE_EPSILONS * epsilon * static_cast<double>(std::abs(expected));

                EXPECT_NEAR(static_cast<double>(quotient.real()),
                            static_cast<double>(expected.real()), bound) << "pair " << i;
                EXPECT_NEAR(static_cast<double>(quotient.imag()),
                            static_cast<double>(expected.imag()), bound) << "pair " << i;
            }
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexDivideIsExactForARealDivisor)
        {
            using R = TypeParam;

            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const std::complex<R> a = this->draw_complex();
                const R divisor = this->draw_real();
                if (divisor == R(0))
                    continue;

                const std::complex<R> quotient =
                    Kernels::divide(a, std::complex<R>(divisor, R(0)));

                EXPECT_EQ(quotient.real(), a.real() / divisor);
                EXPECT_EQ(quotient.imag(), a.imag() / divisor);
            }
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexDivideStaysFiniteForALargeDivisor)
        {
            using R = TypeParam;
            const R large = TestFixture::large_divisor();
            ASSERT_TRUE(std::isinf(large * large));

            const std::complex<R> quotient =
                Kernels::divide(std::complex<R>(R(1), R(1)), std::complex<R>(large, large));
            const double expected = 1.0 / static_cast<double>(large);
            const double bound = K_ONE_ROUNDING_EPSILONS *
                                 static_cast<double>(std::numeric_limits<R>::epsilon()) * expected;

            EXPECT_TRUE(std::isfinite(quotient.real()));
            EXPECT_TRUE(std::isfinite(quotient.imag()));
            EXPECT_NEAR(static_cast<double>(quotient.real()), expected, bound);
            EXPECT_NEAR(static_cast<double>(quotient.imag()), 0.0, bound);
        }

        TYPED_TEST(Unit_ComplexArithmetic, ComplexDivideByZeroGivesNanParts)
        {
            using R = TypeParam;

            const std::complex<R> quotient =
                Kernels::divide(std::complex<R>(R(1), R(2)), std::complex<R>(R(0), R(0)));

            EXPECT_TRUE(std::isnan(quotient.real()));
            EXPECT_TRUE(std::isnan(quotient.imag()));
        }

        TYPED_TEST(Unit_ComplexArithmetic, SafeWrappersReturnWhatTheBricksReturn)
        {
            using R = TypeParam;

            for (int i = 0; i < K_PAIR_COUNT; ++i)
            {
                const std::complex<R> a = this->draw_complex();
                const std::complex<R> b = this->draw_complex();
                if (b == std::complex<R>(R(0), R(0)))
                    continue;

                EXPECT_EQ(Dispatch::safe_conj(a), Kernels::conjugate(a));
                EXPECT_EQ(Dispatch::safe_mul(a, b), Kernels::multiply(a, b));
                EXPECT_EQ(Dispatch::safe_div(a, b), Kernels::divide(a, b));
                EXPECT_EQ(Dispatch::safe_conj(a.real()), a.real());
                EXPECT_EQ(Dispatch::safe_mul(a.real(), b.real()), a.real() * b.real());
                EXPECT_EQ(Dispatch::safe_div(a.real(), b.real()), a.real() / b.real());
            }
        }
    } // namespace Test
} // namespace SushiBLAS
