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
 * @file test_band_operand.cpp
 * @brief Tests the record-time checks on a band operand and on a triangle band operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/band_operand.hpp>
#include "structured_operand_test.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Uplo;

            /// @brief Names the suite of the band operand cases.
            using Medium_BandOperand = StructuredOperandTest;
        } // namespace

        /**
         * @brief Tests that a column-major band tensor gives its columns as lines.
         */
        TEST_F(Medium_BandOperand, AColumnMajorTensorStoresOneLinePerColumn)
        {
            const Tensor t = make_tensor({4, 5}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 2, "op", "A");

            EXPECT_EQ(5, band.lines);
            EXPECT_EQ(4, band.leading_dimension);
            EXPECT_EQ(1, band.lower);
            EXPECT_EQ(2, band.upper);
            EXPECT_EQ(Layout::ColumnMajor, band.layout);
        }

        /**
         * @brief Tests that a row-major band tensor gives its rows as lines.
         */
        TEST_F(Medium_BandOperand, ARowMajorTensorStoresOneLinePerRow)
        {
            const Tensor t = make_tensor({4, 4}, Layout::RowMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 2, "op", "A");

            EXPECT_EQ(4, band.lines);
            EXPECT_EQ(4, band.leading_dimension);
            EXPECT_EQ(1, band.lower);
            EXPECT_EQ(2, band.upper);
            EXPECT_EQ(Layout::RowMajor, band.layout);
        }

        /**
         * @brief Tests that a line may hold more slots than the band needs.
         */
        TEST_F(Medium_BandOperand, SpareSlotsPerLineAreAccepted)
        {
            const Tensor t = make_tensor({6, 5}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 2, "op", "A");

            EXPECT_EQ(5, band.lines);
            EXPECT_EQ(6, band.leading_dimension);
        }

        /**
         * @brief Tests that a band width may exceed what the matrix has diagonals for.
         */
        TEST_F(Medium_BandOperand, AWidthBeyondTheMatrixIsAccepted)
        {
            const Tensor t = make_tensor({4, 2}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 3, 0, "op", "A");

            EXPECT_EQ(2, band.lines);
            EXPECT_EQ(4, band.leading_dimension);
            EXPECT_EQ(3, band.lower);
            EXPECT_EQ(0, band.upper);
        }

        /**
         * @brief Tests that a band tensor of no line is accepted and keeps its line length.
         */
        TEST_F(Medium_BandOperand, AnEmptyTensorHasNoLine)
        {
            const Tensor t = make_tensor_without_storage({3, 0}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 1, "op", "A");

            EXPECT_EQ(0, band.lines);
            EXPECT_EQ(3, band.leading_dimension);
        }

        /**
         * @brief Tests that a triangle band puts its k off-diagonals on the side uplo names.
         */
        TEST_F(Medium_BandOperand, ATriangleBandPutsItsWidthOnOneSide)
        {
            const Tensor t = make_tensor({4, 5}, Layout::ColumnMajor);

            const Dispatch::BandOperand upper =
                Dispatch::describe_triangle_band_operand(t, Uplo::Upper, 3, "op", "A");
            EXPECT_EQ(0, upper.lower);
            EXPECT_EQ(3, upper.upper);
            EXPECT_EQ(5, upper.lines);
            EXPECT_EQ(4, upper.leading_dimension);

            const Dispatch::BandOperand lower =
                Dispatch::describe_triangle_band_operand(t, Uplo::Lower, 3, "op", "A");
            EXPECT_EQ(3, lower.lower);
            EXPECT_EQ(0, lower.upper);
        }

        /**
         * @brief Tests that a negative band width is refused under the name the call gave it.
         */
        TEST_F(Medium_BandOperand, ANegativeWidthIsRefusedByName)
        {
            const Tensor t = make_tensor({4, 5}, Layout::ColumnMajor);

            const auto band_refusal = [&](int64_t kl, int64_t ku)
            {
                return message_of([&]()
                                  {
                                      (void)Dispatch::describe_band_operand(t, kl, ku, "op", "A");
                                  });
            };
            const auto triangle_refusal = [&](Uplo uplo)
            {
                return message_of([&]()
                                  {
                                      (void)Dispatch::describe_triangle_band_operand(
                                          t, uplo, -1, "op", "A");
                                  });
            };

            EXPECT_THROW((void)Dispatch::describe_band_operand(t, -1, 2, "op", "A"),
                         std::invalid_argument);
            EXPECT_THROW((void)Dispatch::describe_band_operand(t, 1, -1, "op", "A"),
                         std::invalid_argument);
            EXPECT_NE(std::string::npos, band_refusal(-1, 2).find("'kl'"));
            EXPECT_NE(std::string::npos, band_refusal(1, -1).find("'ku'"));

            for (const Uplo uplo : {Uplo::Upper, Uplo::Lower})
            {
                SCOPED_TRACE(Core::to_string(uplo));

                EXPECT_THROW(
                    (void)Dispatch::describe_triangle_band_operand(t, uplo, -1, "op", "A"),
                    std::invalid_argument);
                EXPECT_NE(std::string::npos, triangle_refusal(uplo).find("'k'"));
            }
        }

        /**
         * @brief Tests that a band operand must be one dense matrix of rank 2.
         */
        TEST_F(Medium_BandOperand, ATensorThatIsNoDenseMatrixIsRefused)
        {
            const Tensor vector = make_tensor({20}, Layout::ColumnMajor);
            EXPECT_THROW((void)Dispatch::describe_band_operand(vector, 0, 0, "op", "A"),
                         std::invalid_argument);

            const Tensor batch = make_tensor({2, 4, 5}, Layout::ColumnMajor);
            EXPECT_THROW((void)Dispatch::describe_band_operand(batch, 0, 0, "op", "A"),
                         std::invalid_argument);

            const Tensor matrix = make_tensor({4, 5}, Layout::ColumnMajor);
            ASSERT_NO_THROW((void)Dispatch::describe_band_operand(matrix, 0, 0, "op", "A"));
            const Tensor transposed = matrix.transpose(0, 1);
            EXPECT_THROW((void)Dispatch::describe_band_operand(transposed, 0, 0, "op", "A"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that a band wider than a stored line is refused, without overflow for
         *        the largest widths.
         */
        TEST_F(Medium_BandOperand, ABandWiderThanALineIsRefused)
        {
            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout == Layout::RowMajor ? "row-major" : "column-major");

                // Both extents are 4, so a line holds 4 slots under either layout.
                const Tensor t = make_tensor({4, 4}, layout);

                EXPECT_NO_THROW((void)Dispatch::describe_band_operand(t, 1, 2, "op", "A"));
                EXPECT_NO_THROW((void)Dispatch::describe_band_operand(t, 3, 0, "op", "A"));
                EXPECT_NO_THROW((void)Dispatch::describe_band_operand(t, 0, 3, "op", "A"));

                EXPECT_THROW((void)Dispatch::describe_band_operand(t, 2, 2, "op", "A"),
                             std::invalid_argument);
                EXPECT_THROW((void)Dispatch::describe_band_operand(t, 4, 0, "op", "A"),
                             std::invalid_argument);
                EXPECT_THROW((void)Dispatch::describe_band_operand(t, 0, 4, "op", "A"),
                             std::invalid_argument);
                EXPECT_THROW(
                    (void)Dispatch::describe_band_operand(t, INT64_MAX, INT64_MAX, "op", "A"),
                    std::invalid_argument);
                EXPECT_THROW((void)Dispatch::describe_band_operand(t, INT64_MAX, 0, "op", "A"),
                             std::invalid_argument);
                EXPECT_THROW((void)Dispatch::describe_band_operand(t, 0, INT64_MAX, "op", "A"),
                             std::invalid_argument);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
