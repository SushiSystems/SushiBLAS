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
 * @file test_packed_operand.cpp
 * @brief Tests the packed triangle order and the record-time checks on a packed operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
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
                int64_t elements; /**< The elements at hand. */
                int64_t order;    /**< The expected largest order. */
            };

            /// @brief Holds an order whose packed element count needs more than 53 bits.
            constexpr int64_t K_LARGE_ORDER = 3000000000;

            /// @brief Names the suite of the packed operand cases.
            using Unit_PackedOperand = StructuredOperandTest;
        } // namespace

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
    } // namespace Test
} // namespace SushiBLAS
