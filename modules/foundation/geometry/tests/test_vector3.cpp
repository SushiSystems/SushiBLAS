/**************************************************************************/
/* test_vector3.cpp                                                       */
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
 * @file test_vector3.cpp
 * @brief Tests Vector3T, its operators, dot, cross, length and normalize against hand values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type.
        constexpr double K_VECTOR3_TOLERANCE_EPSILONS = 16.0;

        /**
         * @brief Compares vectors of element type T with hand-computed components.
         */
        template<typename T>
        class Unit_GeometryVector3 : public ::testing::Test
        {
            protected:
                /** @brief Returns the absolute tolerance of a comparison in element type T. */
                [[nodiscard]] static double tolerance() noexcept
                {
                    return K_VECTOR3_TOLERANCE_EPSILONS *
                           static_cast<double>(std::numeric_limits<T>::epsilon());
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

        using GeometryVector3Types = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_GeometryVector3, GeometryVector3Types);

        TYPED_TEST(Unit_GeometryVector3, DefaultIsTheZeroVector)
        {
            const Geometry::Vector3T<TypeParam> vector{};

            EXPECT_EQ(vector.x, TypeParam(0));
            EXPECT_EQ(vector.y, TypeParam(0));
            EXPECT_EQ(vector.z, TypeParam(0));
        }

        TYPED_TEST(Unit_GeometryVector3, ComponentsLieInOrderWithoutPadding)
        {
            using Vector = Geometry::Vector3T<TypeParam>;

            EXPECT_EQ(offsetof(Vector, x), std::size_t(0));
            EXPECT_EQ(offsetof(Vector, y), sizeof(TypeParam));
            EXPECT_EQ(offsetof(Vector, z), 2 * sizeof(TypeParam));
        }

        TYPED_TEST(Unit_GeometryVector3, SumDifferenceAndScaleAreComponentwise)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> a{T(1), T(2), T(3)};
            const Geometry::Vector3T<T> b{T(4), T(-5), T(6)};

            TestFixture::expect_vector_near(a + b, 5.0, -3.0, 9.0);
            TestFixture::expect_vector_near(a - b, -3.0, 7.0, -3.0);
            TestFixture::expect_vector_near(a * T(2), 2.0, 4.0, 6.0);
        }

        TYPED_TEST(Unit_GeometryVector3, DotSumsTheComponentProducts)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> a{T(1), T(2), T(3)};
            const Geometry::Vector3T<T> b{T(4), T(-5), T(6)};

            // 1 * 4 + 2 * -5 + 3 * 6 = 12.
            EXPECT_NEAR(static_cast<double>(Geometry::dot(a, b)), 12.0, TestFixture::tolerance());
        }

        TYPED_TEST(Unit_GeometryVector3, CrossIsRightHanded)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> x_axis{T(1), T(0), T(0)};
            const Geometry::Vector3T<T> y_axis{T(0), T(1), T(0)};
            const Geometry::Vector3T<T> a{T(1), T(2), T(3)};
            const Geometry::Vector3T<T> b{T(4), T(-5), T(6)};

            TestFixture::expect_vector_near(Geometry::cross(x_axis, y_axis), 0.0, 0.0, 1.0);
            TestFixture::expect_vector_near(Geometry::cross(y_axis, x_axis), 0.0, 0.0, -1.0);
            // (2 * 6 - 3 * -5, 3 * 4 - 1 * 6, 1 * -5 - 2 * 4) = (27, 6, -13).
            TestFixture::expect_vector_near(Geometry::cross(a, b), 27.0, 6.0, -13.0);
        }

        TYPED_TEST(Unit_GeometryVector3, LengthIsTheEuclideanNorm)
        {
            using T = TypeParam;

            // 3, 4, 12 and 2, 3, 6 are Pythagorean quadruples with norms 13 and 7.
            EXPECT_NEAR(static_cast<double>(
                            Geometry::length(Geometry::Vector3T<T>{T(3), T(4), T(12)})),
                        13.0, TestFixture::tolerance());
            EXPECT_NEAR(static_cast<double>(
                            Geometry::length(Geometry::Vector3T<T>{T(2), T(-3), T(6)})),
                        7.0, TestFixture::tolerance());
        }

        TYPED_TEST(Unit_GeometryVector3, NormalizeScalesToUnitLength)
        {
            using T = TypeParam;

            TestFixture::expect_vector_near(
                Geometry::normalize(Geometry::Vector3T<T>{T(3), T(0), T(4)}), 0.6, 0.0, 0.8);
            TestFixture::expect_vector_near(
                Geometry::normalize(Geometry::Vector3T<T>{T(0), T(-5), T(0)}), 0.0, -1.0, 0.0);
        }

        TYPED_TEST(Unit_GeometryVector3, NormalizeReturnsAZeroLengthVectorUnchanged)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> zero{};
            // The square of the smallest normal value underflows to zero, so the length is zero.
            const Geometry::Vector3T<T> tiny{std::numeric_limits<T>::min(), T(0), T(0)};

            const Geometry::Vector3T<T> from_zero = Geometry::normalize(zero);
            const Geometry::Vector3T<T> from_tiny = Geometry::normalize(tiny);

            EXPECT_EQ(from_zero.x, T(0));
            EXPECT_EQ(from_zero.y, T(0));
            EXPECT_EQ(from_zero.z, T(0));
            EXPECT_EQ(from_tiny.x, std::numeric_limits<T>::min());
            EXPECT_EQ(from_tiny.y, T(0));
            EXPECT_EQ(from_tiny.z, T(0));
        }
    } // namespace Test
} // namespace SushiBLAS
