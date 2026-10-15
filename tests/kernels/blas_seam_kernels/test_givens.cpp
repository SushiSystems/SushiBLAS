/**************************************************************************/
/* test_givens.cpp                                                        */
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
 * @file test_givens.cpp
 * @brief Tests the Givens rotation brick on real and complex pairs.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <random>
#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/givens.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the seed of every random draw in this file.
            constexpr std::uint32_t K_GIVENS_SEED = 20261007u;

            /// @brief Holds the random complex pairs the identities run over.
            constexpr int K_RANDOM_PAIRS = 200;

            /// @brief Holds the largest magnitude of a drawn component.
            constexpr double K_DRAW_RANGE = 1.0;

            /**
             * @brief Holds the allowed residual of a rotation identity for operands of unit
             *        size: 8 units of the epsilon of R.
             */
            template<typename R>
            constexpr R K_IDENTITY_TOLERANCE = R(8) * std::numeric_limits<R>::epsilon();

            /** @brief Takes the square root of a rotation on the host. */
            struct StdRoot
            {
                /** @brief Returns the square root of value. */
                template<typename R>
                [[nodiscard]] static R apply(R value) noexcept
                {
                    return std::sqrt(value);
                }
            };

            /**
             * @brief Holds the scaling bounds of a real type and operands that leave them.
             * @tparam R float or double.
             */
            template<typename R>
            struct GivensLimits;

            /** @brief Holds the bounds and the out-of-range operands of float. */
            template<>
            struct GivensLimits<float>
            {
                static constexpr float K_SAFE_MINIMUM = 0x1p-126f;     /**< 2^-126. */
                static constexpr float K_SAFE_MAXIMUM = 0x1p127f;      /**< 2^127. */
                static constexpr float K_SAFE_ROOT_MINIMUM = 0x1p-63f; /**< 2^-63. */
                static constexpr float K_LARGE = 1e30f;  /**< Its square overflows float. */
                static constexpr float K_SMALL = 1e-30f; /**< Its square underflows float. */
            };

            /** @brief Holds the bounds and the out-of-range operands of double. */
            template<>
            struct GivensLimits<double>
            {
                static constexpr double K_SAFE_MINIMUM = 0x1p-1022;     /**< 2^-1022. */
                static constexpr double K_SAFE_MAXIMUM = 0x1p1023;      /**< 2^1023. */
                static constexpr double K_SAFE_ROOT_MINIMUM = 0x1p-511; /**< 2^-511. */
                static constexpr double K_LARGE = 1e200;  /**< Its square overflows double. */
                static constexpr double K_SMALL = 1e-200; /**< Its square underflows double. */
            };

            /** @brief Names how z of a real rotation is rebuilt from its other values. */
            enum class RebuildValue : std::uint8_t
            {
                Zero,          /**< z is 0. */
                One,           /**< z is 1. */
                Sine,          /**< z is s. */
                InverseCosine /**< z is 1 / c. */
            };

            /** @brief Holds one row of the real table of the design, section 10.2. */
            struct RealRotationRow
            {
                double a;             /**< The first operand. */
                double b;             /**< The second operand. */
                double r;             /**< The expected r, which is exact. */
                double c_numerator;   /**< The expected c is this over c_denominator. */
                double c_denominator; /**< The divisor of the expected c. */
                double s_numerator;   /**< The expected s is this over s_denominator. */
                double s_denominator; /**< The divisor of the expected s. */
                RebuildValue z;       /**< The expected z. */
            };

            /** @brief Holds one row of the complex table of the design, section 10.2. */
            struct ComplexRotationRow
            {
                std::complex<double> f; /**< The first operand. */
                std::complex<double> g; /**< The second operand. */
                std::complex<double> r; /**< The expected r. */
                double c;               /**< The expected cosine. */
                std::complex<double> s; /**< The expected sine. */
            };

            /** @brief Returns a complex value of component type R from one of double. */
            template<typename R>
            [[nodiscard]] std::complex<R> narrowed(const std::complex<double>& value)
            {
                return std::complex<R>(static_cast<R>(value.real()), static_cast<R>(value.imag()));
            }

            /** @brief Returns whether both parts of a complex value are finite. */
            template<typename R>
            [[nodiscard]] bool is_finite(const std::complex<R>& value)
            {
                return std::isfinite(value.real()) && std::isfinite(value.imag());
            }

            /**
             * @brief Expects that the rotation of two real scalars is finite, maps (a, b) to
             *        (r, 0) and has unit norm.
             */
            template<typename R>
            void expect_real_rotation_holds(R a, R b)
            {
                SCOPED_TRACE(::testing::Message() << "a " << a << ", b " << b);

                const Kernels::GivensRotation<R> rotation =
                    Kernels::givens_rotation<StdRoot>(a, b);
                ASSERT_TRUE(std::isfinite(rotation.r));
                ASSERT_TRUE(std::isfinite(rotation.z));
                ASSERT_TRUE(std::isfinite(rotation.c));
                ASSERT_TRUE(std::isfinite(rotation.s));

                const R tolerance = K_IDENTITY_TOLERANCE<R> * (std::abs(a) + std::abs(b));
                EXPECT_LE(std::abs(rotation.c * a + rotation.s * b - rotation.r), tolerance);
                EXPECT_LE(std::abs(rotation.c * b - rotation.s * a), tolerance);
                EXPECT_LE(std::abs(rotation.c * rotation.c + rotation.s * rotation.s - R(1)),
                          K_IDENTITY_TOLERANCE<R>);
            }

            /**
             * @brief Expects that the rotation of two complex scalars is finite, maps (f, g)
             *        to (r, 0), has unit norm and a cosine that is not negative.
             */
            template<typename R>
            void expect_complex_rotation_holds(const std::complex<R>& f, const std::complex<R>& g)
            {
                SCOPED_TRACE(::testing::Message() << "f " << f << ", g " << g);

                const Kernels::GivensRotation<std::complex<R>> rotation =
                    Kernels::givens_rotation<StdRoot>(f, g);
                ASSERT_TRUE(is_finite(rotation.r));
                ASSERT_TRUE(std::isfinite(rotation.c));
                ASSERT_TRUE(is_finite(rotation.s));
                EXPECT_GE(rotation.c, R(0));
                EXPECT_EQ(std::complex<R>(), rotation.z);

                const R tolerance = K_IDENTITY_TOLERANCE<R> * (std::abs(f) + std::abs(g));
                EXPECT_LE(std::abs(rotation.c * f + rotation.s * g - rotation.r), tolerance);
                EXPECT_LE(std::abs(rotation.c * g - std::conj(rotation.s) * f), tolerance);
                EXPECT_LE(std::abs(rotation.c * rotation.c + std::norm(rotation.s) - R(1)),
                          K_IDENTITY_TOLERANCE<R>);
            }
        } // namespace

        /**
         * @brief Draws complex values of component type R from a fixed seed.
         */
        template<typename R>
        class Small_Givens : public ::testing::Test
        {
            protected:
                /** @brief Returns the next complex value, each part in the draw range. */
                [[nodiscard]] std::complex<R> draw_complex()
                {
                    const R real = static_cast<R>(distribution_(generator_));
                    const R imaginary = static_cast<R>(distribution_(generator_));
                    return std::complex<R>(real, imaginary);
                }

            private:
                /// @brief The generator every draw advances.
                std::mt19937 generator_{K_GIVENS_SEED};

                /// @brief The range a component is drawn from.
                std::uniform_real_distribution<double> distribution_{-K_DRAW_RANGE, K_DRAW_RANGE};
        };

        /// @brief Names the real types the rotation suite runs over.
        using GivensTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_Givens, GivensTypes);

        /**
         * @brief Tests at compile time that the scaling bounds are the powers of two xROTG
         *        names.
         */
        TYPED_TEST(Small_Givens, TheSafeScalingBoundsArePowersOfTwo)
        {
            using R = TypeParam;
            using Scaling = Kernels::Detail::SafeScaling<R>;

            static_assert(Scaling::K_MINIMUM == GivensLimits<R>::K_SAFE_MINIMUM,
                          "K_MINIMUM is the smallest normal value.");
            static_assert(Scaling::K_MAXIMUM == GivensLimits<R>::K_SAFE_MAXIMUM,
                          "K_MAXIMUM is the largest power of two.");
            static_assert(Scaling::K_ROOT_MINIMUM == GivensLimits<R>::K_SAFE_ROOT_MINIMUM,
                          "K_ROOT_MINIMUM is the root of K_MINIMUM.");

            SUCCEED();
        }

        /**
         * @brief Tests the real rotation against the table of the design, r for equality.
         */
        TYPED_TEST(Small_Givens, RealPairsMatchTheTable)
        {
            using R = TypeParam;

            const RealRotationRow rows[] = {
                {3.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0, RebuildValue::InverseCosine},
                {4.0, 3.0, 5.0, 4.0, 5.0, 3.0, 5.0, RebuildValue::Sine},
                {-4.0, 3.0, -5.0, -4.0, -5.0, 3.0, -5.0, RebuildValue::Sine},
                {3.0, -4.0, -5.0, 3.0, -5.0, -4.0, -5.0, RebuildValue::InverseCosine},
                {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero},
                {0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, RebuildValue::One},
                {2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero}};

            for (const RealRotationRow& row : rows)
            {
                SCOPED_TRACE(::testing::Message() << "a " << row.a << ", b " << row.b);

                const Kernels::GivensRotation<R> rotation = Kernels::givens_rotation<StdRoot>(
                    static_cast<R>(row.a), static_cast<R>(row.b));
                const R c = static_cast<R>(row.c_numerator) / static_cast<R>(row.c_denominator);
                const R s = static_cast<R>(row.s_numerator) / static_cast<R>(row.s_denominator);

                R z = R(0);
                switch (row.z)
                {
                    case RebuildValue::Zero:
                        z = R(0);
                        break;
                    case RebuildValue::One:
                        z = R(1);
                        break;
                    case RebuildValue::Sine:
                        z = s;
                        break;
                    case RebuildValue::InverseCosine:
                        z = R(1) / c;
                        break;
                }

                EXPECT_EQ(static_cast<R>(row.r), rotation.r);
                EXPECT_EQ(c, rotation.c);
                EXPECT_EQ(s, rotation.s);
                EXPECT_EQ(z, rotation.z);
            }
        }

        /**
         * @brief Tests the complex rotation against the table of the design.
         */
        TYPED_TEST(Small_Givens, ComplexPairsMatchTheTable)
        {
            using R = TypeParam;
            using Complex = std::complex<double>;

            const ComplexRotationRow rows[] = {
                {Complex(3.0, 0.0), Complex(4.0, 0.0), Complex(5.0, 0.0), 0.6, Complex(0.8, 0.0)},
                {Complex(0.0, 3.0), Complex(4.0, 0.0), Complex(0.0, 5.0), 0.6, Complex(0.0, 0.8)},
                {Complex(0.0, 0.0), Complex(3.0, 4.0), Complex(5.0, 0.0), 0.0, Complex(0.6, -0.8)},
                {Complex(0.0, 0.0), Complex(0.0, 4.0), Complex(4.0, 0.0), 0.0, Complex(0.0, -1.0)},
                {Complex(1.0, 2.0), Complex(0.0, 0.0), Complex(1.0, 2.0), 1.0, Complex(0.0, 0.0)}};

            for (const ComplexRotationRow& row : rows)
            {
                SCOPED_TRACE(::testing::Message() << "f " << row.f << ", g " << row.g);

                const std::complex<R> f = narrowed<R>(row.f);
                const std::complex<R> g = narrowed<R>(row.g);
                const Kernels::GivensRotation<std::complex<R>> rotation =
                    Kernels::givens_rotation<StdRoot>(f, g);

                const R tolerance = K_IDENTITY_TOLERANCE<R> * (std::abs(f) + std::abs(g));
                EXPECT_LE(std::abs(rotation.r - narrowed<R>(row.r)), tolerance);
                EXPECT_LE(std::abs(rotation.c - static_cast<R>(row.c)), K_IDENTITY_TOLERANCE<R>);
                EXPECT_LE(std::abs(rotation.s - narrowed<R>(row.s)), K_IDENTITY_TOLERANCE<R>);
                EXPECT_EQ(std::complex<R>(), rotation.z);
            }
        }

        /**
         * @brief Tests the three rotation identities on random complex pairs.
         */
        TYPED_TEST(Small_Givens, RandomComplexPairsKeepTheIdentities)
        {
            for (int pair = 0; pair < K_RANDOM_PAIRS; ++pair)
            {
                const std::complex<TypeParam> f = this->draw_complex();
                const std::complex<TypeParam> g = this->draw_complex();
                expect_complex_rotation_holds(f, g);
            }
        }

        /**
         * @brief Tests that real operands whose squares leave the type give a finite rotation.
         */
        TYPED_TEST(Small_Givens, RealPairsOutsideTheSquareRangeStayFinite)
        {
            using R = TypeParam;

            expect_real_rotation_holds(GivensLimits<R>::K_LARGE, GivensLimits<R>::K_LARGE);
            expect_real_rotation_holds(GivensLimits<R>::K_SMALL, GivensLimits<R>::K_SMALL);
        }

        /**
         * @brief Tests that complex operands whose squares leave the type, or that differ by
         *        30 orders of magnitude, give a finite rotation.
         */
        TYPED_TEST(Small_Givens, ComplexPairsOutsideTheSquareRangeStayFinite)
        {
            using R = TypeParam;

            const std::complex<R> huge_pair(GivensLimits<R>::K_LARGE, GivensLimits<R>::K_LARGE);
            const std::complex<R> tiny_pair(GivensLimits<R>::K_SMALL, GivensLimits<R>::K_SMALL);
            expect_complex_rotation_holds(huge_pair, huge_pair);
            expect_complex_rotation_holds(tiny_pair, tiny_pair);

            // Both factors have magnitude 1, so abs(f) is 1e-30 of abs(g).
            const std::complex<R> g(R(0.6), R(0.8));
            const std::complex<R> f = R(1e-30) * std::complex<R>(R(0.8), R(-0.6));
            expect_complex_rotation_holds(f, g);
        }
    } // namespace Test
} // namespace SushiBLAS
