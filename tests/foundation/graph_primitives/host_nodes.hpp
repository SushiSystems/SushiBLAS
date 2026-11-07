/**************************************************************************/
/* host_nodes.hpp                                                         */
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
 * @file host_nodes.hpp
 * @brief Declares the readers the naming, width and window tests use on a HostPrimitiveRecorder.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <SushiRuntime/core/determinism.hpp>

#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the readers of the nodes a host recorder kept. */
        namespace HostNodes
        {
            /** @brief Returns the name of each recorded node, in record order. */
            [[nodiscard]] inline std::vector<std::string>
            names_of(const GraphPrimitives::HostPrimitiveRecorder& recorder)
            {
                std::vector<std::string> names;
                for (const GraphPrimitives::PrimitiveNodeRecord& node : recorder.nodes())
                    names.push_back(node.name);
                return names;
            }

            /** @brief Returns the determinism class of each recorded node, in record order. */
            [[nodiscard]] inline std::vector<SushiRuntime::Determinism>
            determinisms_of(const GraphPrimitives::HostPrimitiveRecorder& recorder)
            {
                std::vector<SushiRuntime::Determinism> classes;
                for (const GraphPrimitives::PrimitiveNodeRecord& node : recorder.nodes())
                    classes.push_back(node.determinism);
                return classes;
            }

            /** @brief Returns the launch width stamped into each recorded node, in record order. */
            [[nodiscard]] inline std::vector<std::size_t>
            widths_of(const GraphPrimitives::HostPrimitiveRecorder& recorder)
            {
                std::vector<std::size_t> widths;
                for (const GraphPrimitives::PrimitiveNodeRecord& node : recorder.nodes())
                    widths.push_back(node.width);
                return widths;
            }
        } // namespace HostNodes
    } // namespace Test
} // namespace SushiBLAS
