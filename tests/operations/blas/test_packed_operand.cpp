/**************************************************************************/
/* test_packed_operand.cpp                                                */
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
 * @file test_packed_operand.cpp
 * @brief Tests the packed triangle order and the record-time checks on a packed operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
#include "structured_operand_test.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;

            /** @brief Holds one element count and the largest packed order that fits in it. */
            struct PackedCount
            {
                const char* name; /**< Names the row. */
                int64_t elements; /**< The elements at hand. */
                int64_t order;    /**< The expected largest order. */
            };

            /** @brief One triangle count under one layout. */
            struct LayoutCount
            {
                std::string name; /**< Names the row. */
                int64_t elements; /**< The elements the rank-1 tensor holds. */
                int64_t order;    /**< The order the operand must give. */
                Layout layout;    /**< The layout the tensor has and the operand must keep. */
            };

            /// @brief Holds an order whose packed element count needs more than 53 bits.
            constexpr int64_t K_LARGE_ORDER = 3000000000;

            /// @brief Holds the packed element count of K_LARGE_ORDER, written out and not taken from packed_elements.
            constexpr int64_t K_LARGE_COUNT = 4500000001500000000;

            /** @brief Returns the counts below 2^53 and the order each one holds. */
            std::vector<PackedCount> small_counts()
            {
                return {{"Zero", 0, 0}, {"One", 1, 1}, {"Two", 2, 1}, {"Three", 3, 2},
                        {"Five", 5, 2}, {"Six", 6, 3}, {"Seven", 7, 3}, {"Ten", 10, 4}};
            }

            /** @brief Returns the triangle counts under each layout. */
            std::vector<LayoutCount> triangle_rows()
            {
                const PackedCount counts[] = {{"One", 1, 1}, {"Three", 3, 2}, {"Six", 6, 3}, {"Ten", 10, 4}};
                std::vector<LayoutCount> rows;
                for (const Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
                    for (const PackedCount& count : counts)
                        rows.push_back({std::string(layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor") +
                                            count.name,
                                        count.elements, count.order, layout});
                return rows;
            }

            /** @brief Returns the counts that fall between two triangle counts. */
            std::vector<PackedCount> non_triangle_counts()
            {
                return {{"Two", 2, 0}, {"Four", 4, 0}, {"Five", 5, 0}, {"Seven", 7, 0}};
            }

            /** @brief Names the suite of the largest order of a small count. */
            class Small_PackedOperandLargestOrder : public ::testing::TestWithParam<PackedCount>
            {
            };

            /** @brief Names the suite of the triangle counts. */
            class Small_PackedOperandOrder : public ::testing::TestWithParam<LayoutCount>
            {
            };

            /** @brief Names the suite of the counts that are no triangle. */
            class Small_PackedOperandNoTriangle : public ::testing::TestWithParam<PackedCount>
            {
            };
        } // namespace

        TEST_P(Small_PackedOperandLargestOrder, TheLargestOrderIsExactForASmallCount)
        {
            const PackedCount count = GetParam();

            EXPECT_EQ(count.order, Dispatch::largest_packed_order(count.elements));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedOperandLargestOrder, ::testing::ValuesIn(small_counts()),
            [](const ::testing::TestParamInfo<PackedCount>& info) { return std::string(info.param.name); });

        TEST(Small_PackedOperand, TheLargestOrderIsExactBeyondTwoToThe53)
        {
            static_assert(K_LARGE_COUNT == K_LARGE_ORDER / 2 * (K_LARGE_ORDER + 1),
                          "The packed triangle of order 3 000 000 000.");

            EXPECT_EQ(K_LARGE_ORDER, Dispatch::largest_packed_order(K_LARGE_COUNT));
            EXPECT_EQ(K_LARGE_ORDER - 1, Dispatch::largest_packed_order(K_LARGE_COUNT - 1));
            EXPECT_EQ(K_LARGE_ORDER, Dispatch::largest_packed_order(K_LARGE_COUNT + 1));
        }

        TEST(Small_PackedOperand, ThePackedElementCountOfALargeOrderIsExact)
        {
            EXPECT_EQ(K_LARGE_COUNT, Dispatch::packed_elements(K_LARGE_ORDER));
        }

        TEST_P(Small_PackedOperandOrder, ATriangleCountGivesItsOrderAndKeepsItsLayout)
        {
            const LayoutCount row = GetParam();
            const Tensor t = host_matrix({row.elements}, row.layout);

            const Dispatch::PackedOperand packed = Dispatch::describe_packed_operand(t, "op", "AP");

            EXPECT_EQ(row.order, packed.order);
            EXPECT_EQ(row.layout, packed.layout);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedOperandOrder, ::testing::ValuesIn(triangle_rows()),
            [](const ::testing::TestParamInfo<LayoutCount>& info) { return info.param.name; });

        TEST(Small_PackedOperand, AnEmptyTensorHasOrderZeroUnderEachLayout)
        {
            const Tensor row_major = detached_matrix({0}, Layout::RowMajor);
            const Tensor column_major = detached_matrix({0}, Layout::ColumnMajor);

            const Dispatch::PackedOperand row = Dispatch::describe_packed_operand(row_major, "op", "AP");
            const Dispatch::PackedOperand column = Dispatch::describe_packed_operand(column_major, "op", "AP");

            EXPECT_EQ(0, row.order);
            EXPECT_EQ(Layout::RowMajor, row.layout);
            EXPECT_EQ(0, column.order);
            EXPECT_EQ(Layout::ColumnMajor, column.layout);
        }

        TEST_P(Small_PackedOperandNoTriangle, ACountThatIsNoTriangleIsRefused)
        {
            const Tensor t = host_matrix({GetParam().elements});

            EXPECT_THROW((void)Dispatch::describe_packed_operand(t, "op", "AP"), std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedOperandNoTriangle, ::testing::ValuesIn(non_triangle_counts()),
            [](const ::testing::TestParamInfo<PackedCount>& info) { return std::string(info.param.name); });

        TEST(Small_PackedOperand, ARankTwoTensorOfATriangleCountIsRefused)
        {
            const Tensor matrix = host_matrix({2, 3});

            EXPECT_THROW((void)Dispatch::describe_packed_operand(matrix, "op", "AP"), std::invalid_argument);
        }

        TEST(Small_PackedOperand, AStridedVectorOfATriangleCountIsRefused)
        {
            const Tensor strided = host_every_second_element(3, Layout::RowMajor);
            ASSERT_EQ(3, strided.num_elements);

            EXPECT_THROW((void)Dispatch::describe_packed_operand(strided, "op", "AP"), std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
