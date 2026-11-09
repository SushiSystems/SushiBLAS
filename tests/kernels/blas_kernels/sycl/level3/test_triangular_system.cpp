/**************************************************************************/
/* test_triangular_system.cpp                                             */
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
 * @file test_triangular_system.cpp
 * @brief Tests the decoding of a TRSM call into the steps and vectors of one substitution.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_system.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the order of A in the decoding cases.
            constexpr int64_t K_ORDER = 4;

            /// @brief Holds the right-hand-side vectors of B in the decoding cases.
            constexpr int64_t K_VECTORS = 3;

            /// @brief Holds the leading dimension of A in the decoding cases.
            constexpr int64_t K_LDA = 5;

            /// @brief Holds the leading dimension of B in the decoding cases.
            constexpr int64_t K_LDB = 6;

            /// @brief Holds the elements of A, each different, so a misaddressed read shows.
            using Matrix = std::array<std::complex<float>, K_LDA * K_LDA>;

            /** @brief Returns the rows of B: the order on the left, else the vectors. */
            [[nodiscard]] constexpr int64_t rows_of(Core::Side side) noexcept
            {
                return side == Core::Side::Left ? K_ORDER : K_VECTORS;
            }

            /** @brief Returns the columns of B: the vectors on the left, else the order. */
            [[nodiscard]] constexpr int64_t cols_of(Core::Side side) noexcept
            {
                return side == Core::Side::Left ? K_VECTORS : K_ORDER;
            }

            /** @brief Returns the side of a flag: left when true. */
            [[nodiscard]] constexpr Core::Side side_of_flag(bool left) noexcept
            {
                return left ? Core::Side::Left : Core::Side::Right;
            }

            /** @brief Returns the layout of a flag: row major when true. */
            [[nodiscard]] constexpr Core::Layout layout_of_flag(bool row_major) noexcept
            {
                return row_major ? Core::Layout::RowMajor : Core::Layout::ColumnMajor;
            }

            /** @brief Returns the system one call decodes to, with a non-unit lower triangle by default. */
            [[nodiscard]] Kernels::SYCL::TriangularSystem decode(
                Core::Layout layout, Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                Core::Diag diag)
            {
                return Kernels::SYCL::triangular_system(layout, side, uplo, trans, diag,
                                                        rows_of(side), cols_of(side), K_LDA, K_LDB);
            }

            /** @brief Returns A with a different complex value in every element. */
            [[nodiscard]] Matrix distinct_matrix()
            {
                Matrix a;
                for (std::size_t i = 0; i < a.size(); ++i)
                    a[i] = std::complex<float>(static_cast<float>(i + 1),
                                               -static_cast<float>(i + 1) - 0.5f);
                return a;
            }

            /** @brief Returns the coefficient of every (step, unknown) pair, read through an access. */
            [[nodiscard]] std::vector<std::complex<float>> coefficients_of(
                const Kernels::OperandAccess& access, const Matrix& a, bool transposed_pair)
            {
                std::vector<std::complex<float>> values;
                for (int64_t s = 0; s < K_ORDER; ++s)
                    for (int64_t t = 0; t < K_ORDER; ++t)
                        values.push_back(transposed_pair ? access.at(a.data(), t, s)
                                                         : access.at(a.data(), s, t));
                return values;
            }

            /** @brief Returns the element index of every (step, vector) pair, read through an access. */
            [[nodiscard]] std::vector<int64_t> indices_of(const Kernels::OperandAccess& access,
                                                          bool transposed_pair)
            {
                std::vector<int64_t> values;
                for (int64_t s = 0; s < K_ORDER; ++s)
                    for (int64_t v = 0; v < K_VECTORS; ++v)
                        values.push_back(transposed_pair ? access.index(v, s) : access.index(s, v));
                return values;
            }

            /** @brief Returns the coefficient of every (step, unknown) pair of a decoded system. */
            [[nodiscard]] std::vector<std::complex<float>> coefficients_of(
                const Kernels::SYCL::TriangularSystem& system, const Matrix& a)
            {
                std::vector<std::complex<float>> values;
                for (int64_t s = 0; s < K_ORDER; ++s)
                    for (int64_t t = 0; t < K_ORDER; ++t)
                        values.push_back(system.coefficient.at(a.data(), s, t));
                return values;
            }

            /** @brief Returns the element index of every (step, vector) pair of a decoded system. */
            [[nodiscard]] std::vector<int64_t> indices_of(const Kernels::SYCL::TriangularSystem& system)
            {
                std::vector<int64_t> values;
                for (int64_t s = 0; s < K_ORDER; ++s)
                    for (int64_t v = 0; v < K_VECTORS; ++v)
                        values.push_back(system.unknown.index(s, v));
                return values;
            }

            /** @brief Holds one decoding of the direction of a substitution and the direction it must take. */
            struct DirectionRow
            {
                bool left;                /**< Whether op(A) stands on the left of X. */
                Core::Uplo uplo;          /**< The triangle of A that is read. */
                Core::Transpose trans;    /**< The op applied to A. */
                bool forward;             /**< Whether the substitution runs from the first step. */
            };

            /// @brief Names a row after its side and its flags.
            std::string direction_row_name(const ::testing::TestParamInfo<DirectionRow>& info)
            {
                return std::string(info.param.left ? "Left" : "Right") +
                       Core::to_string(info.param.uplo) + Core::to_string(info.param.trans);
            }

            /// @brief Names a row after its layout and its side.
            std::string layout_side_name(const ::testing::TestParamInfo<std::tuple<bool, bool>>& info)
            {
                return std::string(std::get<0>(info.param) ? "RowMajor" : "ColumnMajor") +
                       (std::get<1>(info.param) ? "Left" : "Right");
            }

            /// @brief Names a row after its layout, its side and its op.
            std::string layout_side_op_name(
                const ::testing::TestParamInfo<std::tuple<bool, bool, Core::Transpose>>& info)
            {
                return std::string(std::get<0>(info.param) ? "RowMajor" : "ColumnMajor") +
                       (std::get<1>(info.param) ? "Left" : "Right") +
                       Core::to_string(std::get<2>(info.param));
            }

            /// @brief Names a row after its side and its diagonal.
            std::string side_diag_name(const ::testing::TestParamInfo<std::tuple<bool, bool>>& info)
            {
                return std::string(std::get<0>(info.param) ? "Left" : "Right") +
                       (std::get<1>(info.param) ? "Unit" : "NonUnit");
            }
        } // namespace

        /// @brief Decodes a call by side and diagonal, as a row of (left, unit).
        class Small_TriangularSystemShape : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

        /// @brief Tests that the order, the vectors and the unit flag come from the side and the diagonal.
        TEST_P(Small_TriangularSystemShape, TheSideAndTheDiagonalFixTheShape)
        {
            const Core::Side side = side_of_flag(std::get<0>(GetParam()));
            const bool unit = std::get<1>(GetParam());

            const Kernels::SYCL::TriangularSystem system =
                decode(Core::Layout::RowMajor, side, Core::Uplo::Lower, Core::Transpose::NoTrans,
                       unit ? Core::Diag::Unit : Core::Diag::NonUnit);

            EXPECT_EQ(K_ORDER, system.order);
            EXPECT_EQ(K_VECTORS, system.vectors);
            EXPECT_EQ(unit, system.unit_diag);
        }

        INSTANTIATE_TEST_SUITE_P(SidesAndDiagonals, Small_TriangularSystemShape,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()),
                                 side_diag_name);

        /// @brief Decodes a call by side, triangle and op; on the left an upper op(A) runs from the last step, on the right from the first.
        class Small_TriangularSystemDirection : public ::testing::TestWithParam<DirectionRow> {};

        /// @brief Tests that the substitution runs forward exactly when the side and op(A) say so.
        TEST_P(Small_TriangularSystemDirection, TheSideTheTriangleAndTheOpFixTheDirection)
        {
            const DirectionRow& row = GetParam();

            const Kernels::SYCL::TriangularSystem system =
                decode(Core::Layout::RowMajor, side_of_flag(row.left), row.uplo, row.trans,
                       Core::Diag::NonUnit);

            EXPECT_EQ(row.forward, system.forward);
        }

        // A transposing op turns an upper A into a lower op(A).
        INSTANTIATE_TEST_SUITE_P(
            SidesTrianglesAndOps, Small_TriangularSystemDirection,
            ::testing::Values(
                DirectionRow{true, Core::Uplo::Upper, Core::Transpose::NoTrans, false},
                DirectionRow{true, Core::Uplo::Lower, Core::Transpose::NoTrans, true},
                DirectionRow{true, Core::Uplo::Upper, Core::Transpose::Trans, true},
                DirectionRow{true, Core::Uplo::Lower, Core::Transpose::Trans, false},
                DirectionRow{true, Core::Uplo::Upper, Core::Transpose::ConjTrans, true},
                DirectionRow{true, Core::Uplo::Lower, Core::Transpose::ConjTrans, false},
                DirectionRow{false, Core::Uplo::Upper, Core::Transpose::NoTrans, true},
                DirectionRow{false, Core::Uplo::Lower, Core::Transpose::NoTrans, false},
                DirectionRow{false, Core::Uplo::Upper, Core::Transpose::Trans, false},
                DirectionRow{false, Core::Uplo::Lower, Core::Transpose::Trans, true},
                DirectionRow{false, Core::Uplo::Upper, Core::Transpose::ConjTrans, false},
                DirectionRow{false, Core::Uplo::Lower, Core::Transpose::ConjTrans, true}),
            direction_row_name);

        /// @brief Decodes a call by layout, side and op, as a row of (row major, left, op).
        class Small_TriangularSystemCoefficient
            : public ::testing::TestWithParam<std::tuple<bool, bool, Core::Transpose>>
        {

        };

        /// @brief Tests that the coefficient of step s and unknown t addresses op(A) as the side says.
        TEST_P(Small_TriangularSystemCoefficient, TheCoefficientAddressesOpAOnTheSide)
        {
            const Core::Layout layout = layout_of_flag(std::get<0>(GetParam()));
            const bool left = std::get<1>(GetParam());
            const Core::Transpose trans = std::get<2>(GetParam());
            const Matrix a = distinct_matrix();
            const Kernels::SYCL::TriangularSystem system =
                decode(layout, side_of_flag(left), Core::Uplo::Lower, trans, Core::Diag::NonUnit);

            EXPECT_EQ(coefficients_of(Kernels::operand_access(layout, trans, K_LDA), a, !left),
                      coefficients_of(system, a));
        }

        INSTANTIATE_TEST_SUITE_P(
            LayoutsSidesAndOps, Small_TriangularSystemCoefficient,
            ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                               ::testing::Values(Core::Transpose::NoTrans, Core::Transpose::Trans,
                                                 Core::Transpose::ConjTrans)),
            layout_side_op_name);

        /// @brief Decodes a call by layout and side, as a row of (row major, left).
        class Small_TriangularSystemUnknown : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

        /// @brief Tests that the unknown of step s and vector v addresses B as the side says.
        TEST_P(Small_TriangularSystemUnknown, TheUnknownAddressesBOnTheSide)
        {
            const Core::Layout layout = layout_of_flag(std::get<0>(GetParam()));
            const bool left = std::get<1>(GetParam());
            const Kernels::SYCL::TriangularSystem system =
                decode(layout, side_of_flag(left), Core::Uplo::Lower, Core::Transpose::NoTrans,
                       Core::Diag::NonUnit);

            EXPECT_EQ(indices_of(Kernels::operand_access(layout, Core::Transpose::NoTrans, K_LDB), !left),
                      indices_of(system));
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndSides, Small_TriangularSystemUnknown,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()),
                                 layout_side_name);

        /// @brief Decodes a call by layout and side, as a row of (row major, left).
        class Small_TriangularSystemConjugation
            : public ::testing::TestWithParam<std::tuple<bool, bool>>
        {

        };

        /// @brief Tests that CONJ_TRANS conjugates the factors read from A and no element of B.
        TEST_P(Small_TriangularSystemConjugation, ConjTransConjugatesTheCoefficientAndNotTheUnknown)
        {
            const Core::Layout layout = layout_of_flag(std::get<0>(GetParam()));
            const Core::Side side = side_of_flag(std::get<1>(GetParam()));

            const Kernels::SYCL::TriangularSystem conjugating =
                decode(layout, side, Core::Uplo::Lower, Core::Transpose::ConjTrans,
                       Core::Diag::NonUnit);
            const Kernels::SYCL::TriangularSystem transposing =
                decode(layout, side, Core::Uplo::Lower, Core::Transpose::Trans,
                       Core::Diag::NonUnit);

            EXPECT_TRUE(conjugating.coefficient.conjugated);
            EXPECT_FALSE(conjugating.unknown.conjugated);
            EXPECT_FALSE(transposing.coefficient.conjugated);
            EXPECT_FALSE(transposing.unknown.conjugated);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndSides, Small_TriangularSystemConjugation,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()),
                                 layout_side_name);
    } // namespace Test
} // namespace SushiBLAS
