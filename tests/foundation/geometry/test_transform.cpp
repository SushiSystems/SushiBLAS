/**************************************************************************/
/* test_transform.cpp                                                     */
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
 * @file test_transform.cpp
 * @brief Tests the functions that build a 4x4 transform or take one apart against hand values.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/geometry/matrix4.hpp>
#include <SushiBLAS/geometry/transform.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type.
        constexpr double K_TRANSFORM_TOLERANCE_EPSILONS = 16.0;

        /// @brief Elements a 4x4 matrix holds.
        constexpr std::size_t K_TRANSFORM_ELEMENT_COUNT = 16;

        /// @brief A quarter turn in radians.
        constexpr double K_TRANSFORM_QUARTER_TURN = 1.57079632679489661923;

        /// @brief The sine and the cosine of an eighth of a turn.
        constexpr double K_TRANSFORM_ROOT_HALF = 0.70710678118654752440;

        /// @brief The elements of a 4x4 matrix in storage order, column by column.
        using TransformElements = std::array<double, K_TRANSFORM_ELEMENT_COUNT>;

        /// @brief The identity matrix.
        constexpr TransformElements K_TRANSFORM_IDENTITY = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

        /**
         * @brief Compares transforms of element type T with hand-computed elements.
         */
        template<typename T>
        class Unit_GeometryTransform : public ::testing::Test
        {
            protected:
                /** @brief Returns the absolute tolerance of a comparison in element type T. */
                [[nodiscard]] static double tolerance() noexcept
                {
                    return K_TRANSFORM_TOLERANCE_EPSILONS *
                           static_cast<double>(std::numeric_limits<T>::epsilon());
                }

                /** @brief Returns the matrix that holds elements, each rounded to T. */
                [[nodiscard]] static Geometry::Matrix4T<T>
                matrix_of(const TransformElements& elements) noexcept
                {
                    Geometry::Matrix4T<T> matrix{};
                    for (std::size_t index = 0; index < K_TRANSFORM_ELEMENT_COUNT; ++index)
                        matrix.m[index] = static_cast<T>(elements[index]);
                    return matrix;
                }

                /** @brief Expects each element of actual within tolerance() of its hand value. */
                static void expect_matrix_near(const Geometry::Matrix4T<T>& actual,
                                               const TransformElements& expected)
                {
                    for (std::size_t index = 0; index < K_TRANSFORM_ELEMENT_COUNT; ++index)
                        EXPECT_NEAR(static_cast<double>(actual.m[index]), expected[index],
                                    tolerance())
                            << "element " << index;
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

        using GeometryTransformTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_GeometryTransform, GeometryTransformTypes);

        TYPED_TEST(Unit_GeometryTransform, TranslationWritesTheFourthColumn)
        {
            using T = TypeParam;

            TestFixture::expect_matrix_near(
                Geometry::translation(Geometry::Vector3T<T>{T(1), T(2), T(3)}),
                {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryTransform, ScalingWritesTheDiagonal)
        {
            using T = TypeParam;

            TestFixture::expect_matrix_near(
                Geometry::scaling(Geometry::Vector3T<T>{T(2), T(3), T(4)}),
                {2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryTransform, PerspectiveIsReverseZAndIgnoresTheFarPlane)
        {
            using T = TypeParam;
            // A quarter-turn field of view has a focal length of 1; the aspect of 2 halves x.
            const TransformElements expected = {0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5, 0.0};

            TestFixture::expect_matrix_near(
                Geometry::perspective(T(K_TRANSFORM_QUARTER_TURN), T(2), T(0.5), T(100)), expected);
            TestFixture::expect_matrix_near(
                Geometry::perspective(T(K_TRANSFORM_QUARTER_TURN), T(2), T(0.5), T(1)), expected);
        }

        TYPED_TEST(Unit_GeometryTransform, LookAtPlacesTheTargetOnTheNegativeZAxis)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> origin{};
            const Geometry::Vector3T<T> up{T(0), T(1), T(0)};

            // From (0, 0, 5) the camera axes are the world axes; the origin lands on z = -5.
            TestFixture::expect_matrix_near(
                Geometry::look_at(Geometry::Vector3T<T>{T(0), T(0), T(5)}, origin, up),
                {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -5.0, 1.0});
            // From (3, 0, 0): forward (-1, 0, 0), side (0, 0, -1), up (0, 1, 0); origin on z = -3.
            TestFixture::expect_matrix_near(
                Geometry::look_at(Geometry::Vector3T<T>{T(3), T(0), T(0)}, origin, up),
                {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryTransform, Matrix4FromQuaternionHoldsTheRotatedAxesAsColumns)
        {
            using T = TypeParam;
            const Geometry::QuaternionT<T> third_turn{T(0.5), T(0.5), T(0.5), T(0.5)};
            const Geometry::QuaternionT<T> quarter_turn_about_z{
                T(0), T(0), T(K_TRANSFORM_ROOT_HALF), T(K_TRANSFORM_ROOT_HALF)};

            TestFixture::expect_matrix_near(
                Geometry::matrix4_from_quaternion(Geometry::QuaternionT<T>{}),
                K_TRANSFORM_IDENTITY);
            // A third of a turn about (1, 1, 1) carries x to y, y to z and z to x.
            TestFixture::expect_matrix_near(
                Geometry::matrix4_from_quaternion(third_turn),
                {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
            // A quarter turn about z carries x to y and y to -x.
            TestFixture::expect_matrix_near(
                Geometry::matrix4_from_quaternion(quarter_turn_about_z),
                {0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryTransform, ComposeTransformScalesThenRotatesThenTranslates)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> position{T(1), T(2), T(3)};
            const Geometry::QuaternionT<T> rotation{T(0.5), T(0.5), T(0.5), T(0.5)};
            const Geometry::Vector3T<T> scale{T(2), T(3), T(4)};

            // The rotated axes y, z, x scaled by 2, 3, 4, then the position.
            TestFixture::expect_matrix_near(
                Geometry::compose_transform(position, rotation, scale),
                {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryTransform, QuaternionFromMatrixTakesTheTraceBranchForAQuarterTurn)
        {
            const double root_half = K_TRANSFORM_ROOT_HALF;

            // Quarter turns about x, y and z; each has trace 1.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})),
                root_half, 0.0, 0.0, root_half);
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.0, root_half, 0.0, root_half);
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.0, 0.0, root_half, root_half);
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(K_TRANSFORM_IDENTITY)), 0.0,
                0.0, 0.0, 1.0);
        }

        TYPED_TEST(Unit_GeometryTransform, QuaternionFromMatrixTakesTheXBranchWhenXIsLargest)
        {
            // A half turn about (0.8, 0, 0.6): 2 n n^T - I, trace -1, element (0, 0) largest.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {0.28, 0.0, 0.96, 0.0, 0.0, -1.0, 0.0, 0.0,
                     0.96, 0.0, -0.28, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.8, 0.0, 0.6, 0.0);
        }

        TYPED_TEST(Unit_GeometryTransform, QuaternionFromMatrixTakesTheYBranchWhenYIsLargest)
        {
            // A half turn about (0.6, 0.8, 0): 2 n n^T - I, trace -1, element (1, 1) largest.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {-0.28, 0.96, 0.0, 0.0, 0.96, 0.28, 0.0, 0.0,
                     0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.6, 0.8, 0.0, 0.0);
        }

        TYPED_TEST(Unit_GeometryTransform, QuaternionFromMatrixTakesTheZBranchOtherwise)
        {
            // A half turn about (0.6, 0, 0.8): 2 n n^T - I, trace -1, element (2, 2) largest.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {-0.28, 0.0, 0.96, 0.0, 0.0, -1.0, 0.0, 0.0,
                     0.96, 0.0, 0.28, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.6, 0.0, 0.8, 0.0);
            // A third of a turn about (1, 1, 1) has trace 0 and three equal diagonal elements.
            TestFixture::expect_quaternion_near(
                Geometry::quaternion_from_matrix(TestFixture::matrix_of(
                    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})),
                0.5, 0.5, 0.5, 0.5);
        }

        TYPED_TEST(Unit_GeometryTransform, AffineInverseInvertsAShearedMatrix)
        {
            // Rows of the 3x3 part: (2, 1, 0), (0, 3, 1), (1, 0, 4); translation (1, -2, 3).
            const Geometry::Matrix4T<TypeParam> matrix = TestFixture::matrix_of(
                {2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 1.0, -2.0, 3.0, 1.0});

            const Geometry::Matrix4T<TypeParam> inverse = Geometry::affine_inverse(matrix);

            // The determinant is 25 and the adjugate rows (12, -4, 1), (1, 8, -2), (-3, 1, 6).
            TestFixture::expect_matrix_near(inverse,
                                            {0.48, 0.04, -0.12, 0.0, -0.16, 0.32, 0.04, 0.0, 0.04,
                                             -0.08, 0.24, 0.0, -0.92, 0.84, -0.52, 1.0});
            TestFixture::expect_matrix_near(Geometry::mul(matrix, inverse), K_TRANSFORM_IDENTITY);
            TestFixture::expect_matrix_near(Geometry::mul(inverse, matrix), K_TRANSFORM_IDENTITY);
        }

        TYPED_TEST(Unit_GeometryTransform, AffineInverseReturnsTheIdentityForASingularMatrix)
        {
            // The y scale is zero, so the determinant is zero; the translation is (1, 2, 3).
            const Geometry::Matrix4T<TypeParam> inverse = Geometry::affine_inverse(
                TestFixture::matrix_of(
                    {2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0}));

            for (std::size_t index = 0; index < K_TRANSFORM_ELEMENT_COUNT; ++index)
                EXPECT_EQ(static_cast<double>(inverse.m[index]), K_TRANSFORM_IDENTITY[index])
                    << "element " << index;
        }

        TYPED_TEST(Unit_GeometryTransform, DecomposeTransformRecoversWhatComposeTransformTook)
        {
            using T = TypeParam;
            const Geometry::Vector3T<T> scale{T(2), T(3), T(4)};
            Geometry::Vector3T<T> out_position{};
            Geometry::QuaternionT<T> out_rotation{};
            Geometry::Vector3T<T> out_scale{};

            Geometry::decompose_transform(
                Geometry::compose_transform(
                    Geometry::Vector3T<T>{T(1), T(-2), T(3)},
                    Geometry::QuaternionT<T>{T(0.5), T(0.5), T(0.5), T(0.5)}, scale),
                out_position, out_rotation, out_scale);

            TestFixture::expect_vector_near(out_position, 1.0, -2.0, 3.0);
            TestFixture::expect_quaternion_near(out_rotation, 0.5, 0.5, 0.5, 0.5);
            TestFixture::expect_vector_near(out_scale, 2.0, 3.0, 4.0);

            Geometry::decompose_transform(
                Geometry::compose_transform(
                    Geometry::Vector3T<T>{T(-4), T(0), T(0.5)},
                    Geometry::QuaternionT<T>{T(0), T(0), T(K_TRANSFORM_ROOT_HALF),
                                             T(K_TRANSFORM_ROOT_HALF)},
                    scale),
                out_position, out_rotation, out_scale);

            TestFixture::expect_vector_near(out_position, -4.0, 0.0, 0.5);
            TestFixture::expect_quaternion_near(out_rotation, 0.0, 0.0, K_TRANSFORM_ROOT_HALF,
                                                K_TRANSFORM_ROOT_HALF);
            TestFixture::expect_vector_near(out_scale, 2.0, 3.0, 4.0);
        }

        TYPED_TEST(Unit_GeometryTransform, DecomposeTransformPutsAReflectionOnTheXScale)
        {
            using T = TypeParam;
            Geometry::Vector3T<T> out_position{};
            Geometry::QuaternionT<T> out_rotation{};
            Geometry::Vector3T<T> out_scale{};

            // Scales by (-2, 3, 4), determinant -24, then translates by (1, 2, 3).
            Geometry::decompose_transform(
                TestFixture::matrix_of(
                    {-2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                     0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0}),
                out_position, out_rotation, out_scale);

            TestFixture::expect_vector_near(out_position, 1.0, 2.0, 3.0);
            TestFixture::expect_quaternion_near(out_rotation, 0.0, 0.0, 0.0, 1.0);
            TestFixture::expect_vector_near(out_scale, -2.0, 3.0, 4.0);
        }

        TYPED_TEST(Unit_GeometryTransform, DecomposeTransformKeepsTheIdentityAxisOfAZeroScale)
        {
            using T = TypeParam;
            Geometry::Vector3T<T> out_position{};
            Geometry::QuaternionT<T> out_rotation{};
            Geometry::Vector3T<T> out_scale{};

            // The x column is zero, so there is no x axis to divide by its length.
            Geometry::decompose_transform(
                TestFixture::matrix_of(
                    {0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                     0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0}),
                out_position, out_rotation, out_scale);

            TestFixture::expect_vector_near(out_position, 0.0, 0.0, 0.0);
            TestFixture::expect_quaternion_near(out_rotation, 0.0, 0.0, 0.0, 1.0);
            TestFixture::expect_vector_near(out_scale, 0.0, 3.0, 4.0);
        }
    } // namespace Test
} // namespace SushiBLAS
