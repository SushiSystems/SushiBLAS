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
 * @file test_interpolation.cpp
 * @brief Tests lerp, nlerp and slerp against hand values, with the short-arc and parallel cases.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/geometry/interpolation.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rounded operations on the longest path to a hand value: a blend, a normalisation.
        constexpr double K_INTERPOLATION_ROUNDED_OPERATIONS = 2.0;

        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type:
        /// the default threshold times the rounded operations times the unit roundoff, which is half an epsilon.
        constexpr double K_INTERPOLATION_TOLERANCE_EPSILONS =
            TestSupport::K_DEFAULT_THRESHOLD * K_INTERPOLATION_ROUNDED_OPERATIONS *
            (TestSupport::unit_roundoff<double>() / std::numeric_limits<double>::epsilon());

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

        /** @brief Holds a blend parameter and the value a scalar lerp of 2 and 10 gives at it. */
        struct ScalarBlendRow
        {
            double t;         /**< The blend parameter. */
            double expected;  /**< The hand value. */
        };

        /** @brief Holds a blend parameter and the components of the rotation expected at it. */
        struct RotationBlendRow
        {
            double t;         /**< The blend parameter. */
            double x;         /**< Expected x. */
            double y;         /**< Expected y. */
            double z;         /**< Expected z. */
            double w;         /**< Expected w. */
        };

        /// @brief The lerp of 2 and 10; a parameter of 2 runs as far past b as b is from a.
        constexpr std::array<ScalarBlendRow, 4> K_SCALAR_LERP_ROWS{
            {{0.0, 2.0}, {0.25, 4.0}, {1.0, 10.0}, {2.0, 18.0}}};

        /// @brief The nlerp from no turn to a quarter turn about z; halfway is an eighth of a turn.
        constexpr std::array<RotationBlendRow, 3> K_NLERP_ROWS{
            {{0.0, 0.0, 0.0, 0.0, 1.0},
             {1.0, 0.0, 0.0, K_INTERPOLATION_ROOT_HALF, K_INTERPOLATION_ROOT_HALF},
             {0.5, 0.0, 0.0, K_INTERPOLATION_SINE_SIXTEENTH_TURN, K_INTERPOLATION_COSINE_SIXTEENTH_TURN}}};

        /// @brief The slerp from no turn to a quarter turn about z; a third of it is a twelfth of a turn.
        constexpr std::array<RotationBlendRow, 4> K_SLERP_ROWS{
            {{1.0 / 3.0, 0.0, 0.0, K_INTERPOLATION_SINE_TWENTY_FOURTH_TURN, K_INTERPOLATION_COSINE_TWENTY_FOURTH_TURN},
             {0.5, 0.0, 0.0, K_INTERPOLATION_SINE_SIXTEENTH_TURN, K_INTERPOLATION_COSINE_SIXTEENTH_TURN},
             {0.0, 0.0, 0.0, 0.0, 1.0},
             {1.0, 0.0, 0.0, K_INTERPOLATION_ROOT_HALF, K_INTERPOLATION_ROOT_HALF}}};

        /**
         * @brief Compares blends of element type T with hand-computed components.
         */
        template<typename T>
        class Small_GeometryInterpolation : public ::testing::Test
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

                /** @brief Expects each row's scalar lerp of 2 and 10 to match its hand value. */
                template<std::size_t N>
                static void expect_scalar_lerp_rows(const std::array<ScalarBlendRow, N>& rows)
                {
                    for (const ScalarBlendRow& row : rows)
                    {
                        EXPECT_NEAR(static_cast<double>(Geometry::lerp(T(2), T(10), T(row.t))), row.expected,
                                    tolerance())
                            << "t " << row.t;
                    }
                }

                /** @brief Expects blend(t) of every row to match the row's hand rotation. */
                template<std::size_t N, typename Blend>
                static void expect_rotation_rows(const std::array<RotationBlendRow, N>& rows, Blend blend)
                {
                    for (const RotationBlendRow& row : rows)
                    {
                        SCOPED_TRACE(row.t);
                        expect_quaternion_near(blend(T(row.t)), row.x, row.y, row.z, row.w);
                    }
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
        SB_TYPED_TEST_SUITE(Small_GeometryInterpolation, GeometryInterpolationTypes);

        TYPED_TEST(Small_GeometryInterpolation, ScalarLerpIsLinearAndUnclamped)
        {
            TestFixture::expect_scalar_lerp_rows(K_SCALAR_LERP_ROWS);
        }

        TYPED_TEST(Small_GeometryInterpolation, VectorLerpBlendsEachComponent)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> a{T(0), T(2), T(4)};
            const Geometry::Vector3T<T> b{T(8), T(6), T(4)};

            const Geometry::Vector3T<T> blend = Geometry::lerp(a, b, T(0.25));

            EXPECT_NEAR(static_cast<double>(blend.x), 2.0, TestFixture::tolerance());
            EXPECT_NEAR(static_cast<double>(blend.y), 3.0, TestFixture::tolerance());
            EXPECT_NEAR(static_cast<double>(blend.z), 4.0, TestFixture::tolerance());
        }

        TYPED_TEST(Small_GeometryInterpolation, NlerpReturnsTheEndsAndTheHalfwayRotation)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> quarter = TestFixture::quarter_turn_about_z();

            TestFixture::expect_rotation_rows(
                K_NLERP_ROWS, [&](T t) { return Geometry::nlerp(identity, quarter, t); });
        }

        TYPED_TEST(Small_GeometryInterpolation, NlerpFlipsTheSecondRotationToTheShortArc)
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

        TYPED_TEST(Small_GeometryInterpolation, NlerpReturnsTheIdentityWhenTheBlendHasNoLength)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> zero{T(0), T(0), T(0), T(0)};

            const Geometry::QuaternionT<T> blend = Geometry::nlerp(zero, zero, T(0.5));

            EXPECT_EQ(blend.x, T(0));
            EXPECT_EQ(blend.y, T(0));
            EXPECT_EQ(blend.z, T(0));
            EXPECT_EQ(blend.w, T(1));
        }

        TYPED_TEST(Small_GeometryInterpolation, SlerpTurnsAtAConstantRate)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> quarter = TestFixture::quarter_turn_about_z();

            TestFixture::expect_rotation_rows(
                K_SLERP_ROWS, [&](T t) { return Geometry::slerp(identity, quarter, t); });
        }

        TYPED_TEST(Small_GeometryInterpolation, SlerpFlipsTheSecondRotationToTheShortArc)
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

        TYPED_TEST(Small_GeometryInterpolation, SlerpOfEqualRotationsIsFiniteAndReturnsThem)
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

        TYPED_TEST(Small_GeometryInterpolation, SlerpOfNearlyParallelRotationsIsTheHalfwayRotation)
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
