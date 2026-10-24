/**************************************************************************/
/* plan_labels.hpp                                                        */
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
 * @file plan_labels.hpp
 * @brief Declares the probe that reads the names and determinism classes of a graph's nodes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/graph/plan_view.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the probe the naming tests of the primitives share. */
        namespace PlanLabels
        {
            /** @brief Returns the name of each node of the graph's plan, in recording order. */
            [[nodiscard]] inline std::vector<std::string> names_of(SushiRuntime::Graph& graph)
            {
                std::vector<std::string> names;
                const SushiRuntime::PlanView plan = graph.plan().value();
                for (const SushiRuntime::PlanNode& node : plan.nodes)
                    names.push_back(node.name);
                return names;
            }

            /** @brief Returns the determinism class of each node, in recording order. */
            [[nodiscard]] inline std::vector<SushiRuntime::Determinism>
            determinisms_of(SushiRuntime::Graph& graph)
            {
                std::vector<SushiRuntime::Determinism> classes;
                const SushiRuntime::PlanView plan = graph.plan().value();
                for (const SushiRuntime::PlanNode& node : plan.nodes)
                    classes.push_back(node.determinism);
                return classes;
            }

            /** @brief Returns the name a node reports when its recorder named it nothing. */
            [[nodiscard]] inline std::string unnamed_name(SushiRuntime::Runtime& runtime)
            {
                auto graph = runtime.graph();
                graph.add(std::size_t{1}, SushiRuntime::Reads(), SushiRuntime::Writes(),
                          [](std::size_t)
                          {

                          });
                return graph.plan().value().nodes.front().name;
            }

            /** @brief Returns "name.stage" for each stage, in order. */
            [[nodiscard]] inline std::vector<std::string>
            staged(const std::string& name, const std::vector<std::string>& stages)
            {
                std::vector<std::string> labels;
                for (const std::string& stage : stages)
                    labels.push_back(name + "." + stage);
                return labels;
            }
        } // namespace PlanLabels
    } // namespace Test
} // namespace SushiBLAS
