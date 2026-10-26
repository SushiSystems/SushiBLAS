/**************************************************************************/
/* test_cost.cpp                                                          */
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
 * @file test_cost.cpp
 * @brief Tests that Cost separates a stated cost from an absent one, and names the real class of a dtype.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost.hpp>

#include <string>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_Cost, ADefaultCostIsCostedAndEmpty)
        {
            const Cost cost;
            EXPECT_TRUE(cost.is_costed());
            EXPECT_EQ(cost.flops, 0U);
            EXPECT_EQ(cost.bytes, 0U);
        }

        TEST(Small_Cost, AnUncostedCostKeepsItsReasonAndStatesNothing)
        {
            const Cost cost = Cost::uncosted(UncostedReason::DataDependent);
            EXPECT_FALSE(cost.is_costed());
            EXPECT_EQ(cost.reason, UncostedReason::DataDependent);
            EXPECT_EQ(cost.flops, 0U);
        }

        TEST(Small_Cost, WithBytesSetsTheBytesOfACostedCostOnly)
        {
            Cost costed;
            costed.flops = 7;
            EXPECT_EQ(costed.with_bytes(64).bytes, 64U);
            EXPECT_EQ(costed.with_bytes(64).flops, 7U);
            EXPECT_EQ(Cost::uncosted(UncostedReason::NoRule).with_bytes(64).bytes, 0U);
        }

        TEST(Small_Cost, TheClassFollowsTheDtype)
        {
            EXPECT_EQ(real_class_of(Core::DataType::Half), Core::DataType::Half);
            EXPECT_EQ(real_class_of(Core::DataType::Float32), Core::DataType::Float32);
            EXPECT_EQ(real_class_of(Core::DataType::Float64), Core::DataType::Float64);
            EXPECT_EQ(real_class_of(Core::DataType::Int32), Core::DataType::Int32);
        }

        TEST(Small_Cost, AComplexTypeIsCountedInItsRealClass)
        {
            EXPECT_EQ(real_class_of(Core::DataType::Complex32), Core::DataType::Float32);
            EXPECT_EQ(real_class_of(Core::DataType::Complex64), Core::DataType::Float64);
        }

        TEST(Small_Cost, EveryReasonHasADistinctName)
        {
            const UncostedReason reasons[] = {UncostedReason::None, UncostedReason::NoRule,
                                              UncostedReason::NoShape, UncostedReason::DataDependent,
                                              UncostedReason::Unaudited};
            std::string joined;
            for (const UncostedReason reason : reasons)
                joined += std::string(to_string(reason)) + ",";
            EXPECT_EQ(joined, "none,no_rule,no_shape,data_dependent,unaudited,");
        }
    } // namespace Test
} // namespace SushiBLAS
