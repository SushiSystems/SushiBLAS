/**************************************************************************/
/* test_band_operand.cpp                                                  */
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
 * @file test_band_operand.cpp
 * @brief Tests the record-time checks on a band operand and on a triangle band operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/band_operand.hpp>
#include "host_operands.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Uplo;

            /** @brief Returns the text of the invalid_argument a check throws, or "" when it throws none. */
            std::string refusal_message(const std::function<void()>& check)
            {
                try
                {
                    check();
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }
                return std::string();
            }

            /** @brief One band tensor the geometry check describes, with the geometry it must give. */
            struct GeometryRow
            {
                const char* name;   /**< Names the row. */
                int64_t rows;       /**< Rows of the tensor. */
                int64_t cols;       /**< Columns of the tensor. */
                Layout layout;      /**< Layout of the tensor. */
                int64_t kl;         /**< Sub-diagonals asked. */
                int64_t ku;         /**< Super-diagonals asked. */
                int64_t lines;      /**< Lines the operand must give. */
                int64_t leading;    /**< Leading dimension the operand must give. */
            };

            /** @brief Returns the tensors a band operand describes without a refusal. */
            std::vector<GeometryRow> geometry_rows()
            {
                return {
                    {"ColumnMajorStoresOneLinePerColumn", 4, 5, Layout::ColumnMajor, 1, 2, 5, 4},
                    {"RowMajorStoresOneLinePerRow", 4, 4, Layout::RowMajor, 1, 2, 4, 4},
                    {"SpareSlotsPerLineAreAccepted", 6, 5, Layout::ColumnMajor, 1, 2, 5, 6},
                    {"AWidthBeyondTheMatrixIsAccepted", 4, 2, Layout::ColumnMajor, 3, 0, 2, 4},
                };
            }

            /** @brief Names the suite of the band geometry rows. */
            class Small_BandOperandGeometry : public ::testing::TestWithParam<GeometryRow>
            {
            };

            /** @brief One width the line-capacity check meets, on a square tensor of side four. */
            struct CapacityRow
            {
                std::string name; /**< Names the row. */
                Layout layout;    /**< Layout of the tensor. */
                int64_t kl;       /**< Sub-diagonals asked. */
                int64_t ku;       /**< Super-diagonals asked. */
            };

            /** @brief Returns the widths a line of four slots holds (@p accepted) or does not, under each layout. */
            std::vector<CapacityRow> capacity_rows(bool accepted)
            {
                struct Width
                {
                    const char* label;
                    int64_t kl;
                    int64_t ku;
                    bool accepted;
                };
                const Width widths[] = {
                    {"Kl1Ku2", 1, 2, true},
                    {"Kl3Ku0", 3, 0, true},
                    {"Kl0Ku3", 0, 3, true},
                    {"Kl2Ku2", 2, 2, false},
                    {"Kl4Ku0", 4, 0, false},
                    {"Kl0Ku4", 0, 4, false},
                    {"BothMax", INT64_MAX, INT64_MAX, false},
                    {"KlMax", INT64_MAX, 0, false},
                    {"KuMax", 0, INT64_MAX, false},
                };
                std::vector<CapacityRow> rows;
                for (const Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
                {
                    const std::string prefix = layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor";
                    for (const Width& width : widths)
                        if (width.accepted == accepted)
                            rows.push_back({prefix + width.label, layout, width.kl, width.ku});
                }
                return rows;
            }

            /** @brief Names the suite of the widths a line holds. */
            class Small_BandOperandFits : public ::testing::TestWithParam<CapacityRow>
            {
            };

            /** @brief Names the suite of the widths a line does not hold. */
            class Small_BandOperandOverflows : public ::testing::TestWithParam<CapacityRow>
            {
            };

            /** @brief One tensor that is no dense rank-2 matrix, built on demand. */
            struct ShapeRow
            {
                const char* name;               /**< Names the row. */
                std::function<Tensor()> build;  /**< Builds the tensor the check must refuse. */
            };

            /** @brief Returns the tensors a band operand refuses for their shape. */
            std::vector<ShapeRow> shape_rows()
            {
                return {
                    {"ARankOneVector", [] { return host_matrix({20}, Layout::ColumnMajor); }},
                    {"ARankThreeBatch", [] { return host_matrix({2, 4, 5}, Layout::ColumnMajor); }},
                    {"ATransposedView",
                     [] { return host_matrix({4, 5}, Layout::ColumnMajor).transpose(0, 1); }},
                };
            }

            /** @brief Names the suite of the shape rows. */
            class Small_BandOperandShape : public ::testing::TestWithParam<ShapeRow>
            {
            };

            /** @brief One band width the check refuses as negative, and the name its message gives. */
            struct NegativeWidthRow
            {
                const char* name; /**< Names the row. */
                int64_t kl;       /**< Sub-diagonals asked. */
                int64_t ku;       /**< Super-diagonals asked. */
                const char* named; /**< The parameter the message must name, quoted. */
            };

            /** @brief Names the suite of the negative band width rows. */
            class Small_BandOperandNegativeWidth : public ::testing::TestWithParam<NegativeWidthRow>
            {
            };

            /** @brief One side of the triangle band and the widths it must give. */
            struct TriangleRow
            {
                const char* name; /**< Names the row. */
                Uplo uplo;        /**< Which triangle the band lies in. */
                int64_t lower;    /**< Sub-diagonals the operand must give for a width of three. */
                int64_t upper;    /**< Super-diagonals the operand must give for a width of three. */
            };

            /** @brief Names the suite of the triangle band rows, one per side. */
            class Small_TriangleBandOperand : public ::testing::TestWithParam<TriangleRow>
            {
            };
        } // namespace

        TEST_P(Small_BandOperandGeometry, TheOperandGivesTheLinesTheLeadingDimensionAndTheWidths)
        {
            const GeometryRow row = GetParam();
            const Tensor t = host_matrix({row.rows, row.cols}, row.layout);

            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, row.kl, row.ku, "op", "A");

            EXPECT_EQ(row.lines, band.lines);
            EXPECT_EQ(row.leading, band.leading_dimension);
            EXPECT_EQ(row.kl, band.lower);
            EXPECT_EQ(row.ku, band.upper);
            EXPECT_EQ(row.layout, band.layout);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BandOperandGeometry, ::testing::ValuesIn(geometry_rows()),
            [](const ::testing::TestParamInfo<GeometryRow>& info) { return std::string(info.param.name); });

        TEST(Small_BandOperand, AnEmptyTensorHasNoLine)
        {
            const Tensor t = detached_matrix({3, 0}, Layout::ColumnMajor);

            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 1, "op", "A");

            EXPECT_EQ(0, band.lines);
            EXPECT_EQ(3, band.leading_dimension);
        }

        TEST_P(Small_TriangleBandOperand, TheWidthLandsOnTheSideUploNames)
        {
            const TriangleRow row = GetParam();
            const Tensor t = host_matrix({4, 5}, Layout::ColumnMajor);

            const Dispatch::BandOperand band =
                Dispatch::describe_triangle_band_operand(t, row.uplo, 3, "op", "A");

            EXPECT_EQ(row.lower, band.lower);
            EXPECT_EQ(row.upper, band.upper);
            EXPECT_EQ(5, band.lines);
            EXPECT_EQ(4, band.leading_dimension);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_TriangleBandOperand,
            ::testing::Values(TriangleRow{"Upper", Uplo::Upper, 0, 3}, TriangleRow{"Lower", Uplo::Lower, 3, 0}),
            [](const ::testing::TestParamInfo<TriangleRow>& info) { return std::string(info.param.name); });

        TEST_P(Small_BandOperandNegativeWidth, ANegativeBandWidthIsRefusedUnderItsName)
        {
            const NegativeWidthRow row = GetParam();
            const Tensor t = host_matrix({4, 5}, Layout::ColumnMajor);

            const std::string message = refusal_message(
                [&] { (void)Dispatch::describe_band_operand(t, row.kl, row.ku, "op", "A"); });

            EXPECT_NE(std::string::npos, message.find(row.named));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BandOperandNegativeWidth,
            ::testing::Values(NegativeWidthRow{"SubDiagonals", -1, 2, "'kl'"},
                              NegativeWidthRow{"SuperDiagonals", 1, -1, "'ku'"}),
            [](const ::testing::TestParamInfo<NegativeWidthRow>& info) { return std::string(info.param.name); });

        TEST_P(Small_TriangleBandOperand, ANegativeTriangleWidthIsRefusedUnderK)
        {
            const TriangleRow row = GetParam();
            const Tensor t = host_matrix({4, 5}, Layout::ColumnMajor);

            const std::string message = refusal_message(
                [&] { (void)Dispatch::describe_triangle_band_operand(t, row.uplo, -1, "op", "A"); });

            EXPECT_NE(std::string::npos, message.find("'k'"));
        }

        TEST_P(Small_BandOperandShape, ATensorThatIsNoDenseMatrixIsRefused)
        {
            const Tensor t = GetParam().build();

            EXPECT_THROW((void)Dispatch::describe_band_operand(t, 0, 0, "op", "A"), std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BandOperandShape, ::testing::ValuesIn(shape_rows()),
            [](const ::testing::TestParamInfo<ShapeRow>& info) { return std::string(info.param.name); });

        TEST_P(Small_BandOperandFits, ABandAWholeLineHoldsIsAccepted)
        {
            const CapacityRow row = GetParam();
            const Tensor t = host_matrix({4, 4}, row.layout);

            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, row.kl, row.ku, "op", "A");

            EXPECT_EQ(row.kl, band.lower);
            EXPECT_EQ(row.ku, band.upper);
            EXPECT_EQ(4, band.lines);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BandOperandFits, ::testing::ValuesIn(capacity_rows(true)),
            [](const ::testing::TestParamInfo<CapacityRow>& info) { return info.param.name; });

        TEST_P(Small_BandOperandOverflows, ABandWiderThanALineIsRefusedWithoutOverflow)
        {
            const CapacityRow row = GetParam();
            const Tensor t = host_matrix({4, 4}, row.layout);

            EXPECT_THROW((void)Dispatch::describe_band_operand(t, row.kl, row.ku, "op", "A"),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BandOperandOverflows, ::testing::ValuesIn(capacity_rows(false)),
            [](const ::testing::TestParamInfo<CapacityRow>& info) { return info.param.name; });
    } // namespace Test
} // namespace SushiBLAS
