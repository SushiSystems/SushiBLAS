/**************************************************************************/
/* test_matrix4.cpp                                                       */
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
 * @file test_matrix4.cpp
 * @brief Tests Matrix4T's default, its column-major layout and its product against hand values.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/matrix4.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type.
        constexpr double K_MATRIX4_TOLERANCE_EPSILONS = 16.0;

        /// @brief Elements a 4x4 matrix holds.
        constexpr std::size_t K_MATRIX4_ELEMENT_COUNT = 16;

        /**
         * @brief Compares matrices of element type T with hand-computed elements.
         */
        template<typename T>
        class Unit_GeometryMatrix4 : public ::testing::Test
        {
            protected:
                /** @brief Returns the absolute tolerance of a comparison in element type T. */
                [[nodiscard]] static double tolerance() noexcept
                {
                    return K_MATRIX4_TOLERANCE_EPSILONS *
                           static_cast<double>(std::numeric_limits<T>::epsilon());
                }

                /**
                 * @brief Expects each element of actual within tolerance() of its hand value.
                 * @param expected The elements in storage order, column by column.
                 */
                static void expect_matrix_near(
                    const Geometry::Matrix4T<T>& actual,
                    const std::array<double, K_MATRIX4_ELEMENT_COUNT>& expected)
                {
                    for (std::size_t index = 0; index < K_MATRIX4_ELEMENT_COUNT; ++index)
                        EXPECT_NEAR(static_cast<double>(actual.m[index]), expected[index],
                                    tolerance())
                            << "element " << index;
                }
        };

        using GeometryMatrix4Types = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_GeometryMatrix4, GeometryMatrix4Types);

        TYPED_TEST(Unit_GeometryMatrix4, DefaultIsTheIdentity)
        {
            const Geometry::Matrix4T<TypeParam> matrix{};

            for (std::size_t column = 0; column < 4; ++column)
                for (std::size_t row = 0; row < 4; ++row)
                    EXPECT_EQ(matrix.m[column * 4 + row], TypeParam(column == row ? 1 : 0))
                        << "row " << row << ", column " << column;
        }

        TYPED_TEST(Unit_GeometryMatrix4, HoldsSixteenElementsAndNothingElse)
        {
            EXPECT_EQ(sizeof(Geometry::Matrix4T<TypeParam>),
                      K_MATRIX4_ELEMENT_COUNT * sizeof(TypeParam));
        }

        TYPED_TEST(Unit_GeometryMatrix4, MulReadsRowsOfTheLeftAndColumnsOfTheRight)
        {
            using T = TypeParam;
            // Element (0, 1) is stored at m[4] and element (1, 0) at m[1].
            Geometry::Matrix4T<T> upper{};
            upper.m[4] = T(1);
            Geometry::Matrix4T<T> lower{};
            lower.m[1] = T(1);

            // [[1, 1], [0, 1]] * [[1, 0], [1, 1]] = [[2, 1], [1, 1]] in the upper-left block.
            TestFixture::expect_matrix_near(
                Geometry::mul(upper, lower),
                {2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
            // [[1, 0], [1, 1]] * [[1, 1], [0, 1]] = [[1, 1], [1, 2]].
            TestFixture::expect_matrix_near(
                Geometry::mul(lower, upper),
                {1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        }

        TYPED_TEST(Unit_GeometryMatrix4, MulAppliesTheRightOperandFirst)
        {
            using T = TypeParam;
            // Scales by (2, 3, 4), then translates by (5, 6, 7).
            Geometry::Matrix4T<T> scale_then_move{};
            scale_then_move.m[0] = T(2);
            scale_then_move.m[5] = T(3);
            scale_then_move.m[10] = T(4);
            scale_then_move.m[12] = T(5);
            scale_then_move.m[13] = T(6);
            scale_then_move.m[14] = T(7);
            // Translates by (1, 2, 3).
            Geometry::Matrix4T<T> move{};
            move.m[12] = T(1);
            move.m[13] = T(2);
            move.m[14] = T(3);

            // The point (1, 2, 3) scales to (2, 6, 12) and lands on (7, 12, 19).
            TestFixture::expect_matrix_near(
                Geometry::mul(scale_then_move, move),
                {2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 7.0, 12.0, 19.0, 1.0});
            // The other order adds (1, 2, 3) to the translation (5, 6, 7).
            TestFixture::expect_matrix_near(
                Geometry::mul(move, scale_then_move),
                {2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 6.0, 8.0, 10.0, 1.0});
        }
    } // namespace Test
} // namespace SushiBLAS
