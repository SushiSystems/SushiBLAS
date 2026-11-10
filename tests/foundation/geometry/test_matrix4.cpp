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
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rounded operations on the longest path to a hand value: a product and a sum.
        constexpr double K_MATRIX4_ROUNDED_OPERATIONS = 2.0;

        /// @brief Allowed absolute distance from a hand value, in epsilons of the element type:
        /// the default threshold times the rounded operations times the unit roundoff, which is half an epsilon.
        constexpr double K_MATRIX4_TOLERANCE_EPSILONS =
            TestSupport::K_DEFAULT_THRESHOLD * K_MATRIX4_ROUNDED_OPERATIONS *
            (TestSupport::unit_roundoff<double>() / std::numeric_limits<double>::epsilon());

        /// @brief Elements a 4x4 matrix holds.
        constexpr std::size_t K_MATRIX4_ELEMENT_COUNT = 16;

        namespace
        {
            /** @brief Holds two operands in storage order and the elements of their product. */
            struct ProductRow
            {
                std::array<double, K_MATRIX4_ELEMENT_COUNT> left;      /**< The left operand, column by column. */
                std::array<double, K_MATRIX4_ELEMENT_COUNT> right;     /**< The right operand, column by column. */
                std::array<double, K_MATRIX4_ELEMENT_COUNT> expected;  /**< The product, column by column. */
            };
        } // namespace

        /**
         * @brief The products of the order promise of mul.
         * @details Rows one and two: with the element (0, 1) at m[4] and (1, 0) at m[1], [[1, 1], [0, 1]] times
         * [[1, 0], [1, 1]] is [[2, 1], [1, 1]] and the reverse is [[1, 1], [1, 2]]. Rows three and four: a scale
         * by (2, 3, 4) then a move by (5, 6, 7), and a move by (1, 2, 3); the point (1, 2, 3) scales to
         * (2, 6, 12) and lands on (7, 12, 19), and the other order adds (1, 2, 3) to (5, 6, 7).
         */
        constexpr std::array<ProductRow, 4> K_PRODUCT_ROWS{{
            {{1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
             {1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
             {2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
            {{1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
             {1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
             {1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
            {{2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 6, 7, 1},
             {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1},
             {2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 7, 12, 19, 1}},
            {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1},
             {2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 6, 7, 1},
             {2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 6, 8, 10, 1}}}};

        /**
         * @brief Compares matrices of element type T with hand-computed elements.
         */
        template<typename T>
        class Small_GeometryMatrix4 : public ::testing::Test
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

                /** @brief Returns a matrix holding the given elements in storage order. */
                [[nodiscard]] static Geometry::Matrix4T<T>
                matrix_of(const std::array<double, K_MATRIX4_ELEMENT_COUNT>& elements)
                {
                    Geometry::Matrix4T<T> matrix{};
                    for (std::size_t index = 0; index < K_MATRIX4_ELEMENT_COUNT; ++index)
                        matrix.m[index] = T(elements[index]);
                    return matrix;
                }

                /** @brief Expects mul of the operands of every row to give the row's product. */
                static void expect_product_rows()
                {
                    for (const ProductRow& row : K_PRODUCT_ROWS)
                        expect_matrix_near(Geometry::mul(matrix_of(row.left), matrix_of(row.right)), row.expected);
                }
        };

        using GeometryMatrix4Types = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_GeometryMatrix4, GeometryMatrix4Types);

        TYPED_TEST(Small_GeometryMatrix4, MulReadsRowsOfTheLeftAndColumnsOfTheRight)
        {
            TestFixture::expect_product_rows();
        }
    } // namespace Test
} // namespace SushiBLAS
