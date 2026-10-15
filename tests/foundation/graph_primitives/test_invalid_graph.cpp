/**************************************************************************/
/* test_invalid_graph.cpp                                                 */
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
 * @file test_invalid_graph.cpp
 * @brief Tests the InvalidGraph error a primitive returns and the outcome of a recording.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_GraphPrimitivesInvalidGraph, CarriesTheInvalidGraphCodeAndTheMessage)
        {
            const SushiRuntime::Result<void> refused =
                GraphPrimitives::Detail::invalid_graph("count exceeds the capacity");

            ASSERT_FALSE(refused.has_value());
            EXPECT_EQ(refused.error().code, SushiRuntime::ErrorCode::InvalidGraph);
            EXPECT_EQ(refused.error().message, std::string("count exceeds the capacity"));
        }

        TEST(Unit_GraphPrimitivesInvalidGraph, AFreshRecorderHasNothingToReport)
        {
            auto runtime = SushiRuntime::Runtime::create().value();
            auto graph = runtime.graph();

            EXPECT_TRUE(GraphPrimitives::Detail::recording_outcome(graph.recorder()).has_value());
        }
    } // namespace Test
} // namespace SushiBLAS
