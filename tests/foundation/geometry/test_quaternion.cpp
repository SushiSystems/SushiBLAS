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
 * @file test_quaternion.cpp
 * @brief Tests QuaternionT, its product, conjugate, norm and vector rotation against hand values.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rounded operations on the longest path to a hand value: a product and a sum.
        constexpr double K_QUATERNION_ROUNDED_OPERATIONS = 2.0;

        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type:
        /// the default threshold times the rounded operations times the unit roundoff, which is half an epsilon.
        constexpr double K_QUATERNION_TOLERANCE_EPSILONS =
            TestSupport::K_DEFAULT_THRESHOLD * K_QUATERNION_ROUNDED_OPERATIONS *
            (TestSupport::unit_roundoff<double>() / std::numeric_limits<double>::epsilon());

        /// @brief Half a turn in radians.
        constexpr double K_QUATERNION_HALF_TURN = 3.14159265358979323846;

        /// @brief The sine and the cosine of an eighth of a turn.
        constexpr double K_QUATERNION_ROOT_HALF = 0.70710678118654752440;

        /** @brief Holds the components of a quaternion in the order x, y, z, w. */
        using QuaternionComponents = std::array<double, 4>;

        /** @brief Holds the components of a vector in the order x, y, z. */
        using VectorComponents = std::array<double, 3>;

        /** @brief Holds an axis, an angle and the quaternion axis-angle gives for them. */
        struct AxisAngleRow
        {
            VectorComponents axis;           /**< The axis of the rotation. */
            double angle;                    /**< The angle in radians. */
            QuaternionComponents expected;   /**< The hand quaternion. */
        };

        namespace
        {
            /** @brief Holds two factors and their Hamilton product. */
            struct ProductRow
            {
                QuaternionComponents left;       /**< The left factor. */
                QuaternionComponents right;      /**< The right factor. */
                QuaternionComponents expected;   /**< The hand product. */
            };
        } // namespace

        /** @brief Holds a rotation, a vector and the rotated vector. */
        struct RotateRow
        {
            QuaternionComponents rotation;   /**< The rotation. */
            VectorComponents vector;         /**< The vector turned. */
            VectorComponents expected;       /**< The hand result. */
        };

        /// @brief A quarter turn: sin and cos of an eighth of a turn; a half turn: sin of a quarter turn is one.
        constexpr std::array<AxisAngleRow, 3> K_AXIS_ANGLE_ROWS{{
            {{0.0, 1.0, 0.0}, K_QUATERNION_HALF_TURN * 0.5, {0.0, K_QUATERNION_ROOT_HALF, 0.0, K_QUATERNION_ROOT_HALF}},
            {{0.0, 0.0, 1.0}, K_QUATERNION_HALF_TURN, {0.0, 0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}, 0.0, {0.0, 0.0, 0.0, 1.0}}}};

        /// @brief i j = k, j i = -k and (4 + i + 2j + 3k)(8 + 5i + 6j + 7k) = -6 + 24i + 48j + 48k.
        constexpr std::array<ProductRow, 3> K_PRODUCT_ROWS{{
            {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}},
            {{0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}},
            {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {24.0, 48.0, 48.0, -6.0}}}};

        /// @brief The identity; a quarter turn about z, which carries x to y; the same on (1, 2, 3); a third of a
        /// turn about (1, 1, 1), which cycles (x, y, z) to (z, x, y).
        constexpr std::array<RotateRow, 4> K_ROTATE_ROWS{{
            {{0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
            {{0.0, 0.0, K_QUATERNION_ROOT_HALF, K_QUATERNION_ROOT_HALF}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            {{0.0, 0.0, K_QUATERNION_ROOT_HALF, K_QUATERNION_ROOT_HALF}, {1.0, 2.0, 3.0}, {-2.0, 1.0, 3.0}},
            {{0.5, 0.5, 0.5, 0.5}, {1.0, 2.0, 3.0}, {3.0, 1.0, 2.0}}}};

        /**
         * @brief Compares quaternions and vectors of element type T with hand-computed components.
         */
        template<typename T>
        class Small_GeometryQuaternion : public ::testing::Test
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

                /** @brief Expects quaternion_axis_angle of every row to give the row's quaternion. */
                static void expect_axis_angle_rows()
                {
                    for (const AxisAngleRow& row : K_AXIS_ANGLE_ROWS)
                    {
                        const QuaternionComponents& want = row.expected;
                        expect_quaternion_near(Geometry::quaternion_axis_angle(vector_of(row.axis), T(row.angle)),
                                               want[0], want[1], want[2], want[3]);
                    }
                }

                /** @brief Expects mul of the factors of every row to give the row's product. */
                static void expect_product_rows()
                {
                    for (const ProductRow& row : K_PRODUCT_ROWS)
                    {
                        const QuaternionComponents& want = row.expected;
                        expect_quaternion_near(Geometry::mul(quaternion_of(row.left), quaternion_of(row.right)),
                                               want[0], want[1], want[2], want[3]);
                    }
                }

                /** @brief Expects rotate of the vector of every row to give the row's result. */
                static void expect_rotate_rows()
                {
                    for (const RotateRow& row : K_ROTATE_ROWS)
                    {
                        const VectorComponents& want = row.expected;
                        expect_vector_near(Geometry::rotate(quaternion_of(row.rotation), vector_of(row.vector)),
                                           want[0], want[1], want[2]);
                    }
                }

                /** @brief Returns the quaternion of the given components. */
                [[nodiscard]] static Geometry::QuaternionT<T> quaternion_of(const QuaternionComponents& c)
                {
                    return Geometry::QuaternionT<T>{T(c[0]), T(c[1]), T(c[2]), T(c[3])};
                }

                /** @brief Returns the vector of the given components. */
                [[nodiscard]] static Geometry::Vector3T<T> vector_of(const VectorComponents& c)
                {
                    return Geometry::Vector3T<T>{T(c[0]), T(c[1]), T(c[2])};
                }

                /** @brief Expects actual to be exactly the identity rotation. */
                static void expect_identity(const Geometry::QuaternionT<T>& actual)
                {
                    EXPECT_EQ(actual.x, T(0));
                    EXPECT_EQ(actual.y, T(0));
                    EXPECT_EQ(actual.z, T(0));
                    EXPECT_EQ(actual.w, T(1));
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
        SB_TYPED_TEST_SUITE(Small_GeometryQuaternion, GeometryQuaternionTypes);

        TYPED_TEST(Small_GeometryQuaternion, ComponentsLieInOrderWithoutPadding)
        {
            using Quaternion = Geometry::QuaternionT<TypeParam>;

            EXPECT_EQ(offsetof(Quaternion, x), std::size_t(0));
            EXPECT_EQ(offsetof(Quaternion, y), sizeof(TypeParam));
            EXPECT_EQ(offsetof(Quaternion, z), 2 * sizeof(TypeParam));
            EXPECT_EQ(offsetof(Quaternion, w), 3 * sizeof(TypeParam));
        }

        TYPED_TEST(Small_GeometryQuaternion, AxisAngleHoldsTheHalfAngleSineAndCosine)
        {
            TestFixture::expect_axis_angle_rows();
        }

        TYPED_TEST(Small_GeometryQuaternion, MulIsTheHamiltonProduct)
        {
            TestFixture::expect_product_rows();
        }

        TYPED_TEST(Small_GeometryQuaternion, ConjugateNegatesTheVectorPart)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> q{T(1), T(-2), T(3), T(4)};

            TestFixture::expect_quaternion_near(Geometry::conjugate(q), -1.0, 2.0, -3.0, 4.0);
        }

        TYPED_TEST(Small_GeometryQuaternion, RotateTurnsAVectorAboutTheAxis)
        {
            TestFixture::expect_rotate_rows();
        }

        TYPED_TEST(Small_GeometryQuaternion, NormalizeScalesToUnitLength)
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

        TYPED_TEST(Small_GeometryQuaternion, NormalizeReturnsTheIdentityForAZeroLength)
        {
            using T = TypeParam;

            TestFixture::expect_identity(Geometry::normalize(Geometry::QuaternionT<T>{T(0), T(0), T(0), T(0)}));
        }

        TYPED_TEST(Small_GeometryQuaternion, NormalizeReturnsTheIdentityWhenTheSquaredLengthUnderflows)
        {
            using T = TypeParam;
            // The square of the smallest normal value underflows to zero, so the length is zero.
            const Geometry::QuaternionT<T> tiny{std::numeric_limits<T>::min(), T(0), T(0), T(0)};

            TestFixture::expect_identity(Geometry::normalize(tiny));
        }

        TYPED_TEST(Small_GeometryQuaternion, DotSumsTheFourComponentProducts)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> a{T(1), T(2), T(3), T(4)};
            const Geometry::QuaternionT<T> b{T(5), T(6), T(7), T(8)};

            // 5 + 12 + 21 + 32 = 70.
            EXPECT_NEAR(static_cast<double>(Geometry::dot(a, b)), 70.0, TestFixture::tolerance());
        }
    } // namespace Test
} // namespace SushiBLAS
