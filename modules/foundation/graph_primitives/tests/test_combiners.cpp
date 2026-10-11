/**************************************************************************/
/* test_combiners.cpp                                                     */
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
 * @file test_combiners.cpp
 * @brief Tests what each combiner returns, its identity and that a kernel can capture it.
 * @author Mustafa Garip
 * @see modules/foundation/graph_primitives/tests/README.md
 */

#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/combiners.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Maximum;
            using GraphPrimitives::Minimum;
            using GraphPrimitives::Product;
            using GraphPrimitives::Sum;

            static_assert(std::is_trivially_copyable<Sum<float>>::value,
                          "A device kernel captures the combiner by value.");
            static_assert(std::is_trivially_copyable<Product<float>>::value,
                          "A device kernel captures the combiner by value.");
            static_assert(std::is_trivially_copyable<Minimum<float>>::value,
                          "A device kernel captures the combiner by value.");
            static_assert(std::is_trivially_copyable<Maximum<float>>::value,
                          "A device kernel captures the combiner by value.");
        } // namespace

        /** @brief Tests the value each combiner returns for two operands in both orders. */
        TEST(Unit_GraphPrimitivesCombiners, EachCombinesTwoValues)
        {
            EXPECT_EQ(Sum<std::int32_t>{}(3, 4), 7);
            EXPECT_EQ(Product<std::int32_t>{}(3, 4), 12);
            EXPECT_EQ(Minimum<std::int32_t>{}(3, 4), 3);
            EXPECT_EQ(Minimum<std::int32_t>{}(4, 3), 3);
            EXPECT_EQ(Maximum<std::int32_t>{}(3, 4), 4);
            EXPECT_EQ(Maximum<std::int32_t>{}(4, 3), 4);
        }

        /** @brief Tests that combining a value with the identity returns the value. */
        TEST(Unit_GraphPrimitivesCombiners, IdentityLeavesAValueUnchanged)
        {
            constexpr std::int32_t K_VALUE = -17;

            EXPECT_EQ(Sum<std::int32_t>{}(Sum<std::int32_t>::identity(), K_VALUE), K_VALUE);
            EXPECT_EQ(Product<std::int32_t>{}(Product<std::int32_t>::identity(), K_VALUE),
                      K_VALUE);
            EXPECT_EQ(Minimum<std::int32_t>{}(Minimum<std::int32_t>::identity(), K_VALUE),
                      K_VALUE);
            EXPECT_EQ(Maximum<std::int32_t>{}(Maximum<std::int32_t>::identity(), K_VALUE),
                      K_VALUE);
        }

        /** @brief Tests the identity of each combiner on a floating-point element type. */
        TEST(Unit_GraphPrimitivesCombiners, FloatingPointIdentities)
        {
            // An identity is a constant of the type, so the comparisons below are exact.
            EXPECT_EQ(Sum<double>::identity(), 0.0);
            EXPECT_EQ(Product<double>::identity(), 1.0);
            EXPECT_EQ(Minimum<double>::identity(), std::numeric_limits<double>::max());
            EXPECT_EQ(Maximum<double>::identity(), std::numeric_limits<double>::lowest());
        }

    } // namespace Test
} // namespace SushiBLAS
