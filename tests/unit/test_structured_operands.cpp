/**************************************************************************/
/* test_structured_operands.cpp                                           */
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
 * @file test_structured_operands.cpp
 * @brief Tests the record-time checks on a band, a packed and a scalar operand.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/blas/band_operand.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
#include <SushiBLAS/blas/scalar_operand.hpp>
#include "../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;
            using Core::Layout;
            using Core::Uplo;

            /// @brief Holds the two layouts every case runs under.
            constexpr Layout K_LAYOUTS[] = {Layout::ColumnMajor, Layout::RowMajor};

            /// @brief Holds an order whose packed element count needs more than 53 bits.
            constexpr int64_t K_LARGE_ORDER = 3000000000;

            /** @brief Holds one element count and the largest packed order that fits in it. */
            struct PackedCount
            {
                int64_t elements; /**< The elements at hand. */
                int64_t order;    /**< The expected largest order. */
            };

            /**
             * @brief Builds the tensors the three operand suites describe.
             */
            class StructuredOperandTest : public SushiBLASTest
            {
                protected:
                    /** @brief Returns a dense tensor over a shared storage of its own. */
                    [[nodiscard]] static Tensor make_tensor(std::initializer_list<int64_t> dims,
                                                            Layout layout = Layout::RowMajor,
                                                            DataType dtype = DataType::Float32)
                    {
                        int64_t elements = 1;
                        for (const int64_t extent : dims)
                            elements *= extent;
                        return Tensor(make_storage(elements, dtype), dims, dtype, 0, layout);
                    }

                    /** @brief Returns a tensor of a shape that owns no storage. */
                    [[nodiscard]] static Tensor
                    make_tensor_without_storage(std::initializer_list<int64_t> dims,
                                                Layout layout = Layout::RowMajor)
                    {
                        return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, DataType::Float32,
                                      0, layout);
                    }

                    /**
                     * @brief Returns a rank-1 view of every second element of a storage.
                     * @param elements Elements the view holds.
                     */
                    [[nodiscard]] static Tensor make_every_second_element(int64_t elements,
                                                                         Layout layout)
                    {
                        const int64_t dims[] = {elements};
                        std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                        strides[0] = 2;
                        return Tensor(make_storage(2 * elements, DataType::Float32),
                                      SushiRuntime::span<const int64_t>(dims, 1), strides,
                                      DataType::Float32, 0, layout);
                    }

                    /** @brief Returns the text of the invalid_argument a check throws, or "". */
                    template<typename Check>
                    [[nodiscard]] static std::string message_of(const Check& check)
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

                private:
                    /** @brief Returns a shared storage of a count of elements of a dtype. */
                    [[nodiscard]] static SushiRuntime::sushi_ptr<Storage>
                    make_storage(int64_t elements, DataType dtype)
                    {
                        return SushiRuntime::make_sushi<Storage>(
                            TestEnvironment::get_context(),
                            static_cast<std::size_t>(elements) * Core::bytes_per_element(dtype),
                            SushiRuntime::API::Residency::Shared);
                    }
            };

            /// @brief Names the suite of the band operand cases.
            using Unit_BandOperand = StructuredOperandTest;

            /// @brief Names the suite of the packed operand cases.
            using Unit_PackedOperand = StructuredOperandTest;

            /// @brief Names the suite of the scalar operand cases.
            using Unit_ScalarOperand = StructuredOperandTest;
        } // namespace

        /**
         * @brief Tests that a column-major band tensor gives its columns as lines.
         */
        TEST_F(Unit_BandOperand, AColumnMajorTensorStoresOneLinePerColumn)
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
        TEST_F(Unit_BandOperand, ARowMajorTensorStoresOneLinePerRow)
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
        TEST_F(Unit_BandOperand, SpareSlotsPerLineAreAccepted)
        {
            const Tensor t = make_tensor({6, 5}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 2, "op", "A");

            EXPECT_EQ(5, band.lines);
            EXPECT_EQ(6, band.leading_dimension);
        }

        /**
         * @brief Tests that a band width may exceed what the matrix has diagonals for.
         */
        TEST_F(Unit_BandOperand, AWidthBeyondTheMatrixIsAccepted)
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
        TEST_F(Unit_BandOperand, AnEmptyTensorHasNoLine)
        {
            const Tensor t = make_tensor_without_storage({3, 0}, Layout::ColumnMajor);
            const Dispatch::BandOperand band = Dispatch::describe_band_operand(t, 1, 1, "op", "A");

            EXPECT_EQ(0, band.lines);
            EXPECT_EQ(3, band.leading_dimension);
        }

        /**
         * @brief Tests that a triangle band puts its k off-diagonals on the side uplo names.
         */
        TEST_F(Unit_BandOperand, ATriangleBandPutsItsWidthOnOneSide)
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
        TEST_F(Unit_BandOperand, ANegativeWidthIsRefusedByName)
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
        TEST_F(Unit_BandOperand, ATensorThatIsNoDenseMatrixIsRefused)
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
        TEST_F(Unit_BandOperand, ABandWiderThanALineIsRefused)
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

        /**
         * @brief Tests the largest packed order of small counts and of counts beyond 2^53.
         */
        TEST_F(Unit_PackedOperand, TheLargestOrderIsExact)
        {
            const PackedCount counts[] = {{0, 0}, {1, 1}, {2, 1}, {3, 2},
                                          {5, 2}, {6, 3}, {7, 3}, {10, 4}};
            for (const PackedCount& count : counts)
                EXPECT_EQ(count.order, Dispatch::largest_packed_order(count.elements))
                    << count.elements << " elements";

            // Written out, not taken from packed_elements, which is the brick under test.
            constexpr int64_t large_count = K_LARGE_ORDER / 2 * (K_LARGE_ORDER + 1);
            static_assert(large_count == 4500000001500000000,
                          "The packed triangle of order 3 000 000 000.");

            EXPECT_EQ(K_LARGE_ORDER, Dispatch::largest_packed_order(large_count));
            EXPECT_EQ(K_LARGE_ORDER - 1, Dispatch::largest_packed_order(large_count - 1));
            EXPECT_EQ(K_LARGE_ORDER, Dispatch::largest_packed_order(large_count + 1));
            EXPECT_EQ(large_count, Dispatch::packed_elements(K_LARGE_ORDER));
        }

        /**
         * @brief Tests that a rank-1 tensor of n * (n + 1) / 2 elements has order n and keeps
         *        its layout.
         */
        TEST_F(Unit_PackedOperand, ATriangleCountGivesItsOrder)
        {
            const PackedCount counts[] = {{1, 1}, {3, 2}, {6, 3}, {10, 4}};

            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout == Layout::RowMajor ? "row-major" : "column-major");

                for (const PackedCount& count : counts)
                {
                    const Tensor t = make_tensor({count.elements}, layout);
                    const Dispatch::PackedOperand packed =
                        Dispatch::describe_packed_operand(t, "op", "AP");

                    EXPECT_EQ(count.order, packed.order) << count.elements << " elements";
                    EXPECT_EQ(layout, packed.layout) << count.elements << " elements";
                }

                const Tensor empty = make_tensor_without_storage({0}, layout);
                const Dispatch::PackedOperand packed =
                    Dispatch::describe_packed_operand(empty, "op", "AP");
                EXPECT_EQ(0, packed.order);
                EXPECT_EQ(layout, packed.layout);
            }
        }

        /**
         * @brief Tests that an element count between two triangle counts is refused.
         */
        TEST_F(Unit_PackedOperand, ACountThatIsNoTriangleIsRefused)
        {
            for (const int64_t elements : {int64_t{2}, int64_t{4}, int64_t{5}, int64_t{7}})
            {
                SCOPED_TRACE(::testing::Message() << elements << " elements");

                const Tensor t = make_tensor({elements});
                EXPECT_THROW((void)Dispatch::describe_packed_operand(t, "op", "AP"),
                             std::invalid_argument);
            }
        }

        /**
         * @brief Tests that a packed operand must be a dense rank-1 tensor.
         */
        TEST_F(Unit_PackedOperand, ATensorThatIsNoDenseVectorIsRefused)
        {
            // Both tensors hold a triangle count, so only the rank or the stride refuses them.
            const Tensor matrix = make_tensor({2, 3});
            EXPECT_THROW((void)Dispatch::describe_packed_operand(matrix, "op", "AP"),
                         std::invalid_argument);

            const Tensor strided = make_every_second_element(3, Layout::RowMajor);
            ASSERT_EQ(3, strided.num_elements);
            EXPECT_THROW((void)Dispatch::describe_packed_operand(strided, "op", "AP"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that one stored element is a scalar whatever the rank of its tensor.
         */
        TEST_F(Unit_ScalarOperand, OneElementOfAnyRankIsAccepted)
        {
            const Tensor rank0 = make_tensor(std::initializer_list<int64_t>{});
            ASSERT_EQ(0, rank0.rank);
            ASSERT_EQ(1, rank0.num_elements);
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank0, DataType::Float32, "op", "alpha"));

            const Tensor rank1 = make_tensor({1});
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank1, DataType::Float32, "op", "alpha"));

            const Tensor rank2 = make_tensor({1, 1});
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank2, DataType::Float32, "op", "alpha"));
        }

        /**
         * @brief Tests that a tensor of another element count is no scalar.
         */
        TEST_F(Unit_ScalarOperand, AnotherElementCountIsRefused)
        {
            const Tensor two = make_tensor({2});
            EXPECT_THROW(Dispatch::require_scalar_operand(two, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);

            const Tensor none = make_tensor_without_storage({0});
            EXPECT_THROW(Dispatch::require_scalar_operand(none, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that a scalar operand must own storage.
         */
        TEST_F(Unit_ScalarOperand, AMissingStorageIsRefused)
        {
            const Tensor t = make_tensor_without_storage({1});
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that a scalar operand must hold the dtype the call expects.
         */
        TEST_F(Unit_ScalarOperand, AnotherDtypeIsRefused)
        {
            const Tensor t = make_tensor({1});
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float64, "op", "alpha"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Int32, "op", "alpha"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
