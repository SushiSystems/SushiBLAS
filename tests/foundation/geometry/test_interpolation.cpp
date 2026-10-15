/**************************************************************************/
/* test_interpolation.cpp                                                 */
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
 * @file test_interpolation.cpp
 * @brief Tests lerp, nlerp and slerp against hand values, with the short-arc and parallel cases.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/geometry/interpolation.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type.
        constexpr double K_INTERPOLATION_TOLERANCE_EPSILONS = 16.0;

        /// @brief The sine and the cosine of an eighth of a turn.
        constexpr double K_INTERPOLATION_ROOT_HALF = 0.70710678118654752440;

        /// @brief The sine of a sixteenth of a turn, 22.5 degrees.
        constexpr double K_INTERPOLATION_SINE_SIXTEENTH_TURN = 0.38268343236508977173;

        /// @brief The cosine of a sixteenth of a turn, 22.5 degrees.
        constexpr double K_INTERPOLATION_COSINE_SIXTEENTH_TURN = 0.92387953251128675613;

        /// @brief The sine of a twenty-fourth of a turn, 15 degrees.
        constexpr double K_INTERPOLATION_SINE_TWENTY_FOURTH_TURN = 0.25881904510252076235;

        /// @brief The cosine of a twenty-fourth of a turn, 15 degrees.
        constexpr double K_INTERPOLATION_COSINE_TWENTY_FOURTH_TURN = 0.96592582628906828675;

        /// @brief The z of a rotation about z by a small angle, the sine of half that angle.
        constexpr double K_INTERPOLATION_NEAR_PARALLEL_Z = 0.0006;

        /// @brief The w of that rotation, sqrt(1 - 0.0006^2), above slerp's threshold of 1 - 1e-6.
        constexpr double K_INTERPOLATION_NEAR_PARALLEL_W = 0.9999998199999838;

        /// @brief The z of the rotation halfway to it, the sine of half of asin(0.0006).
        constexpr double K_INTERPOLATION_NEAR_PARALLEL_HALFWAY_Z = 0.000300000013500003;

        /// @brief The w of the rotation halfway to it, the cosine of half of asin(0.0006).
        constexpr double K_INTERPOLATION_NEAR_PARALLEL_HALFWAY_W = 0.99999995499999494;

        /**
         * @brief Compares blends of element type T with hand-computed components.
         */
        template<typename T>
        class Unit_GeometryInterpolation : public ::testing::Test
        {
            protected:
                /** @brief Returns the absolute tolerance of a comparison in element type T. */
                [[nodiscard]] static double tolerance() noexcept
                {
                    return K_INTERPOLATION_TOLERANCE_EPSILONS *
                           static_cast<double>(std::numeric_limits<T>::epsilon());
                }

                /** @brief Expects each component of actual within tolerance() of its hand value. */
                static void expect_quaternion_near(const Geometry::QuaternionT<T>& actual, double x,
                                                   double y, double z, double w)
                {
                    EXPECT_NEAR(static_cast<double>(actual.x), x, tolerance());
                    EXPECT_NEAR(static_cast<double>(actual.y), y, tolerance());
                    EXPECT_NEAR(static_cast<double>(actual.z), z, tolerance());
                    EXPECT_NEAR(static_cast<double>(actual.w), w, tolerance());
                }

                /** @brief Returns the rotation of a quarter turn about z. */
                [[nodiscard]] static Geometry::QuaternionT<T> quarter_turn_about_z() noexcept
                {
                    return Geometry::QuaternionT<T>{T(0), T(0), T(K_INTERPOLATION_ROOT_HALF),
                                                    T(K_INTERPOLATION_ROOT_HALF)};
                }

                /** @brief Returns q with every component negated, the same rotation. */
                [[nodiscard]] static Geometry::QuaternionT<T>
                negated(const Geometry::QuaternionT<T>& q) noexcept
                {
                    return Geometry::QuaternionT<T>{-q.x, -q.y, -q.z, -q.w};
                }
        };

        using GeometryInterpolationTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_GeometryInterpolation, GeometryInterpolationTypes);

        TYPED_TEST(Unit_GeometryInterpolation, ScalarLerpIsLinearAndUnclamped)
        {
            using T = TypeParam;
            const double tolerance = TestFixture::tolerance();

            EXPECT_NEAR(static_cast<double>(Geometry::lerp(T(2), T(10), T(0))), 2.0, tolerance);
            EXPECT_NEAR(static_cast<double>(Geometry::lerp(T(2), T(10), T(0.25))), 4.0, tolerance);
            EXPECT_NEAR(static_cast<double>(Geometry::lerp(T(2), T(10), T(1))), 10.0, tolerance);
            // A parameter of 2 runs as far past b as b is from a.
            EXPECT_NEAR(static_cast<double>(Geometry::lerp(T(2), T(10), T(2))), 18.0, tolerance);
        }

        TYPED_TEST(Unit_GeometryInterpolation, VectorLerpBlendsEachComponent)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> a{T(0), T(2), T(4)};
            const Geometry::Vector3T<T> b{T(8), T(6), T(4)};

            const Geometry::Vector3T<T> blend = Geometry::lerp(a, b, T(0.25));

            EXPECT_NEAR(static_cast<double>(blend.x), 2.0, TestFixture::tolerance());
            EXPECT_NEAR(static_cast<double>(blend.y), 3.0, TestFixture::tolerance());
            EXPECT_NEAR(static_cast<double>(blend.z), 4.0, TestFixture::tolerance());
        }

        TYPED_TEST(Unit_GeometryInterpolation, NlerpReturnsTheEndsAndTheHalfwayRotation)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> quarter = TestFixture::quarter_turn_about_z();

            TestFixture::expect_quaternion_near(Geometry::nlerp(identity, quarter, T(0)), 0.0, 0.0,
                                                0.0, 1.0);
            TestFixture::expect_quaternion_near(Geometry::nlerp(identity, quarter, T(1)), 0.0, 0.0,
                                                K_INTERPOLATION_ROOT_HALF,
                                                K_INTERPOLATION_ROOT_HALF);
            // Halfway between no turn and a quarter turn is an eighth of a turn about z.
            TestFixture::expect_quaternion_near(
                Geometry::nlerp(identity, quarter, T(0.5)), 0.0, 0.0,
                K_INTERPOLATION_SINE_SIXTEENTH_TURN, K_INTERPOLATION_COSINE_SIXTEENTH_TURN);
        }

        TYPED_TEST(Unit_GeometryInterpolation, NlerpFlipsTheSecondRotationToTheShortArc)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> far_side =
                TestFixture::negated(TestFixture::quarter_turn_about_z());

            // Without the flip the blend would be (0, 0, -0.9239, 0.3827), the long way round.
            TestFixture::expect_quaternion_near(Geometry::nlerp(identity, far_side, T(0.5)), 0.0,
                                                0.0, K_INTERPOLATION_SINE_SIXTEENTH_TURN,
                                                K_INTERPOLATION_COSINE_SIXTEENTH_TURN);
        }

        TYPED_TEST(Unit_GeometryInterpolation, NlerpReturnsTheIdentityWhenTheBlendHasNoLength)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> zero{T(0), T(0), T(0), T(0)};

            const Geometry::QuaternionT<T> blend = Geometry::nlerp(zero, zero, T(0.5));

            EXPECT_EQ(blend.x, T(0));
            EXPECT_EQ(blend.y, T(0));
            EXPECT_EQ(blend.z, T(0));
            EXPECT_EQ(blend.w, T(1));
        }

        TYPED_TEST(Unit_GeometryInterpolation, SlerpTurnsAtAConstantRate)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> quarter = TestFixture::quarter_turn_about_z();

            // A third of a quarter turn is a twelfth of a turn, whose half angle is 15 degrees.
            TestFixture::expect_quaternion_near(
                Geometry::slerp(identity, quarter, T(1.0 / 3.0)), 0.0, 0.0,
                K_INTERPOLATION_SINE_TWENTY_FOURTH_TURN, K_INTERPOLATION_COSINE_TWENTY_FOURTH_TURN);
            TestFixture::expect_quaternion_near(
                Geometry::slerp(identity, quarter, T(0.5)), 0.0, 0.0,
                K_INTERPOLATION_SINE_SIXTEENTH_TURN, K_INTERPOLATION_COSINE_SIXTEENTH_TURN);
            TestFixture::expect_quaternion_near(Geometry::slerp(identity, quarter, T(0)), 0.0, 0.0,
                                                0.0, 1.0);
            TestFixture::expect_quaternion_near(Geometry::slerp(identity, quarter, T(1)), 0.0, 0.0,
                                                K_INTERPOLATION_ROOT_HALF,
                                                K_INTERPOLATION_ROOT_HALF);
        }

        TYPED_TEST(Unit_GeometryInterpolation, SlerpFlipsTheSecondRotationToTheShortArc)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> far_side =
                TestFixture::negated(TestFixture::quarter_turn_about_z());

            // Without the flip the arc would span 135 degrees and a third of it would be 45.
            TestFixture::expect_quaternion_near(
                Geometry::slerp(identity, far_side, T(1.0 / 3.0)), 0.0, 0.0,
                K_INTERPOLATION_SINE_TWENTY_FOURTH_TURN, K_INTERPOLATION_COSINE_TWENTY_FOURTH_TURN);
        }

        TYPED_TEST(Unit_GeometryInterpolation, SlerpOfEqualRotationsIsFiniteAndReturnsThem)
        {
            using T = TypeParam;
            // The angle between them is zero, where the sine quotient would be 0 / 0.
            const Geometry::QuaternionT<T> rotation{T(0.5), T(0.5), T(0.5), T(0.5)};

            const Geometry::QuaternionT<T> same = Geometry::slerp(rotation, rotation, T(0.25));
            const Geometry::QuaternionT<T> flipped =
                Geometry::slerp(rotation, TestFixture::negated(rotation), T(0.25));

            EXPECT_TRUE(std::isfinite(same.x) && std::isfinite(same.y) && std::isfinite(same.z) &&
                        std::isfinite(same.w));
            TestFixture::expect_quaternion_near(same, 0.5, 0.5, 0.5, 0.5);
            TestFixture::expect_quaternion_near(flipped, 0.5, 0.5, 0.5, 0.5);
        }

        TYPED_TEST(Unit_GeometryInterpolation, SlerpOfNearlyParallelRotationsIsTheHalfwayRotation)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> nearby{T(0), T(0), T(K_INTERPOLATION_NEAR_PARALLEL_Z),
                                                  T(K_INTERPOLATION_NEAR_PARALLEL_W)};

            TestFixture::expect_quaternion_near(Geometry::slerp(identity, nearby, T(0.5)), 0.0, 0.0,
                                                K_INTERPOLATION_NEAR_PARALLEL_HALFWAY_Z,
                                                K_INTERPOLATION_NEAR_PARALLEL_HALFWAY_W);
        }
    } // namespace Test
} // namespace SushiBLAS
