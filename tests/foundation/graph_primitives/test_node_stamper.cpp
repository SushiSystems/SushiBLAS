/**************************************************************************/
/* test_node_stamper.cpp                                                  */
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
 * @file test_node_stamper.cpp
 * @brief Tests the NodeOptions a NodeStamper builds for a named and an unnamed call.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include <SushiRuntime/core/determinism.hpp>

#include <SushiBLAS/graph_primitives/node_stamper.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::PrimitiveOptions;
            using GraphPrimitives::Detail::NodeStamper;
        } // namespace

        /** @brief Tests that an unnamed call gets the width and an operation id, and no name or class. */
        TEST(Small_GraphPrimitivesNodeStamper, AnUnnamedCallStatesTheWidthAndTheOperationOnly)
        {
            const std::optional<PrimitiveOptions> none;
            NodeStamper stamper(none, "graph.test");

            const SushiRuntime::NodeOptions staged = stamper.stage("level", std::size_t{7});
            EXPECT_TRUE(staged.name.empty());
            EXPECT_EQ(staged.width, 7u);
            EXPECT_EQ(staged.determinism, SushiRuntime::Determinism::Unstated);

            const SushiRuntime::NodeOptions single = stamper.node();
            EXPECT_TRUE(single.name.empty());
            EXPECT_EQ(single.width, SushiRuntime::NodeOptions{}.width);
        }

        /** @brief Tests that every node carries the hash of its label as its operation. */
        TEST(Small_GraphPrimitivesNodeStamper, EveryNodeCarriesTheHashOfItsLabelAsItsOperation)
        {
            const std::optional<PrimitiveOptions> none;
            NodeStamper unnamed(none, "graph.test");
            EXPECT_EQ(unnamed.stage("level").operation, SushiRuntime::OperationID(
                SushiRuntime::OperationID::hash("graph.test.level", 16)));
            EXPECT_EQ(unnamed.node().operation, SushiRuntime::OperationID(
                SushiRuntime::OperationID::hash("graph.test", 10)));

            const std::optional<PrimitiveOptions> named = PrimitiveOptions{"gather"};
            NodeStamper stamper(named, "graph.test");
            EXPECT_EQ(stamper.node().operation, SushiRuntime::OperationID(
                SushiRuntime::OperationID::hash("gather", 6)));
        }

        /** @brief Tests that a single-node primitive reports under the bare name. */
        TEST(Small_GraphPrimitivesNodeStamper, ANodeCarriesTheBareNameAndTheClass)
        {
            const std::optional<PrimitiveOptions> options =
                PrimitiveOptions{"gather", SushiRuntime::Determinism::Tolerant};
            NodeStamper stamper(options, "graph.test");

            const SushiRuntime::NodeOptions node = stamper.node(std::size_t{3});
            EXPECT_EQ(std::string(node.name), "gather");
            EXPECT_EQ(node.width, 3u);
            EXPECT_EQ(node.determinism, SushiRuntime::Determinism::Tolerant);
        }

        /** @brief Tests that a stage is appended to the name after a dot. */
        TEST(Small_GraphPrimitivesNodeStamper, AStageIsAppendedAfterADot)
        {
            const std::optional<PrimitiveOptions> options =
                PrimitiveOptions{"sort", SushiRuntime::Determinism::Bitwise};
            NodeStamper stamper(options, "graph.test");

            EXPECT_EQ(std::string(stamper.stage("resolve").name), "sort.resolve");
            EXPECT_EQ(std::string(stamper.stage("placement").name), "sort.placement");
            EXPECT_EQ(stamper.stage("placement").determinism, SushiRuntime::Determinism::Bitwise);
        }
    } // namespace Test
} // namespace SushiBLAS
