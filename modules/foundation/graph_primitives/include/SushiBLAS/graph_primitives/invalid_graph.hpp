/**************************************************************************/
/* invalid_graph.hpp                                                      */
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
 * @file invalid_graph.hpp
 * @brief Declares the two outcomes a primitive returns: a broken precondition and a refusal.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <utility>

#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Returns the InvalidGraph error of a broken recording precondition.
             * @param message Names the primitive, the argument at fault and the values involved.
             * @see modules/foundation/graph_primitives/README.md
             */
            [[nodiscard]] inline SushiRuntime::ErrorWrapper<SushiRuntime::Error>
            invalid_graph(std::string message)
            {
                return SushiRuntime::make_error(SushiRuntime::Error{
                    SushiRuntime::ErrorCode::InvalidGraph, std::move(message)});
            }

            /**
             * @brief Returns what a primitive reports once it has recorded its nodes.
             * @return Nothing, or the first recording the recorder refused since it was cleared.
             * @see modules/foundation/graph_primitives/README.md
             */
            template <class Port>
            [[nodiscard]] SushiRuntime::Result<void>
            recording_outcome(const Port& recorder)
            {
                if (const SushiRuntime::Error* refused = recorder.refusal())
                    return SushiRuntime::make_error(*refused);

                return {};
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
