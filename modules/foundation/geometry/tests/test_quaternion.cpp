/**************************************************************************/
/* test_quaternion.cpp                                                    */
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
 * @file test_quaternion.cpp
 * @brief Tests QuaternionT, its product, conjugate, norm and vector rotation against hand values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type.
        constexpr double K_QUATERNION_TOLERANCE_EPSILONS = 16.0;

        /// @brief Half a turn in radians.
        constexpr double K_QUATERNION_HALF_TURN = 3.14159265358979323846;

        /// @brief The sine and the cosine of an eighth of a turn.
        constexpr double K_QUATERNION_ROOT_HALF = 0.70710678118654752440;

        /**
         * @brief Compares quaternions and vectors of element type T with hand-computed components.
         */
        template<typename T>
        class Unit_GeometryQuaternion : public ::testing::Test
        {
            protected:
                /** @brief Returns the absolute tolerance of a comparison in element type T. */
                [[nodiscard]] static double tolerance() noexcept
                {
                    return K_QUATERNION_TOLERANCE_EPSILONS *
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

                /** @brief Expects each component of actual within tolerance() of its hand value. */
                static void expect_vector_near(const Geometry::Vector3T<T>& actual, double x,
                                               double y, double z)
                {
                    EXPECT_NEAR(static_cast<double>(actual.x), x, tolerance());
                    EXPECT_NEAR(static_cast<double>(actual.y), y, tolerance());
                    EXPECT_NEAR(static_cast<double>(actual.z), z, tolerance());
                }
        };

        using GeometryQuaternionTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_GeometryQuaternion, GeometryQuaternionTypes);

        TYPED_TEST(Unit_GeometryQuaternion, DefaultIsTheIdentityRotation)
        {
            const Geometry::QuaternionT<TypeParam> quaternion{};

            EXPECT_EQ(quaternion.x, TypeParam(0));
            EXPECT_EQ(quaternion.y, TypeParam(0));
            EXPECT_EQ(quaternion.z, TypeParam(0));
            EXPECT_EQ(quaternion.w, TypeParam(1));
        }

        TYPED_TEST(Unit_GeometryQuaternion, ComponentsLieInOrderWithoutPadding)
        {
            using Quaternion = Geometry::QuaternionT<TypeParam>;

            EXPECT_EQ(offsetof(Quaternion, x), std::size_t(0));
            EXPECT_EQ(offsetof(Quaternion, y), sizeof(TypeParam));
            EXPECT_EQ(offsetof(Quaternion, z), 2 * sizeof(TypeParam));
            EXPECT_EQ(offsetof(Quaternion, w), 3 * sizeof(TypeParam));
        }

        TYPED_TEST(Unit_GeometryQuaternion, AxisAngleHoldsTheHalfAngleSineAndCosine)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> y_axis{T(0), T(1), T(0)};
            const Geometry::Vector3T<T> z_axis{T(0), T(0), T(1)};

            // A quarter turn: sin and cos of an eighth of a turn.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_axis_angle(y_axis, T(K_QUATERNION_HALF_TURN * 0.5)), 0.0,
                K_QUATERNION_ROOT_HALF, 0.0, K_QUATERNION_ROOT_HALF);
            // A half turn: sin of a quarter turn is one and its cosine zero.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_axis_angle(z_axis, T(K_QUATERNION_HALF_TURN)), 0.0, 0.0, 1.0,
                0.0);
            TestFixture::expect_quaternion_near(Geometry::quaternion_axis_angle(z_axis, T(0)), 0.0,
                                                0.0, 0.0, 1.0);
        }

        TYPED_TEST(Unit_GeometryQuaternion, MulIsTheHamiltonProduct)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> i{T(1), T(0), T(0), T(0)};
            const Geometry::QuaternionT<T> j{T(0), T(1), T(0), T(0)};
            const Geometry::QuaternionT<T> a{T(1), T(2), T(3), T(4)};
            const Geometry::QuaternionT<T> b{T(5), T(6), T(7), T(8)};

            // i * j = k and j * i = -k.
            TestFixture::expect_quaternion_near(Geometry::mul(i, j), 0.0, 0.0, 1.0, 0.0);
            TestFixture::expect_quaternion_near(Geometry::mul(j, i), 0.0, 0.0, -1.0, 0.0);
            // (4 + i + 2j + 3k)(8 + 5i + 6j + 7k) = -6 + 24i + 48j + 48k.
            TestFixture::expect_quaternion_near(Geometry::mul(a, b), 24.0, 48.0, 48.0, -6.0);
        }

        TYPED_TEST(Unit_GeometryQuaternion, ConjugateNegatesTheVectorPart)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> q{T(1), T(-2), T(3), T(4)};

            TestFixture::expect_quaternion_near(Geometry::conjugate(q), -1.0, 2.0, -3.0, 4.0);
        }

        TYPED_TEST(Unit_GeometryQuaternion, RotateTurnsAVectorAboutTheAxis)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> x_axis{T(1), T(0), T(0)};
            const Geometry::Vector3T<T> v{T(1), T(2), T(3)};
            const Geometry::QuaternionT<T> identity{};
            const Geometry::QuaternionT<T> quarter_turn_about_z{
                T(0), T(0), T(K_QUATERNION_ROOT_HALF), T(K_QUATERNION_ROOT_HALF)};
            const Geometry::QuaternionT<T> third_turn{T(0.5), T(0.5), T(0.5), T(0.5)};

            TestFixture::expect_vector_near(Geometry::rotate(identity, v), 1.0, 2.0, 3.0);
            // A quarter turn about z carries x to y and (1, 2, 3) to (-2, 1, 3).
            TestFixture::expect_vector_near(Geometry::rotate(quarter_turn_about_z, x_axis), 0.0,
                                            1.0, 0.0);
            TestFixture::expect_vector_near(Geometry::rotate(quarter_turn_about_z, v), -2.0, 1.0,
                                            3.0);
            // A third of a turn about (1, 1, 1) cycles the axes: (x, y, z) goes to (z, x, y).
            TestFixture::expect_vector_near(Geometry::rotate(third_turn, v), 3.0, 1.0, 2.0);
        }

        TYPED_TEST(Unit_GeometryQuaternion, NormalizeScalesToUnitLength)
        {
            using T = TypeParam;

            TestFixture::expect_quaternion_near(
                Geometry::normalize(Geometry::QuaternionT<T>{T(0), T(0), T(3), T(4)}), 0.0, 0.0,
                0.6, 0.8);
            // 1 + 4 + 4 + 16 = 25, so the length is 5.
            TestFixture::expect_quaternion_near(
                Geometry::normalize(Geometry::QuaternionT<T>{T(1), T(-2), T(2), T(4)}), 0.2, -0.4,
                0.4, 0.8);
        }

        TYPED_TEST(Unit_GeometryQuaternion, NormalizeReturnsTheIdentityForAZeroLength)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> zero{T(0), T(0), T(0), T(0)};
            // The square of the smallest normal value underflows to zero, so the length is zero.
            const Geometry::QuaternionT<T> tiny{std::numeric_limits<T>::min(), T(0), T(0), T(0)};

            for (const Geometry::QuaternionT<T>& degenerate : {zero, tiny})
            {
                const Geometry::QuaternionT<T> result = Geometry::normalize(degenerate);

                EXPECT_EQ(result.x, T(0));
                EXPECT_EQ(result.y, T(0));
                EXPECT_EQ(result.z, T(0));
                EXPECT_EQ(result.w, T(1));
            }
        }

        TYPED_TEST(Unit_GeometryQuaternion, DotSumsTheFourComponentProducts)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> a{T(1), T(2), T(3), T(4)};
            const Geometry::QuaternionT<T> b{T(5), T(6), T(7), T(8)};

            // 5 + 12 + 21 + 32 = 70.
            EXPECT_NEAR(static_cast<double>(Geometry::dot(a, b)), 70.0, TestFixture::tolerance());
        }
    } // namespace Test
} // namespace SushiBLAS
