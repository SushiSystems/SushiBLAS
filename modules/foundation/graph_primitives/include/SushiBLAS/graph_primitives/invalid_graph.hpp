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
 * @file invalid_graph.hpp
 * @brief Declares the two outcomes a primitive returns: a broken precondition and a refusal.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <utility>

#include <SushiRuntime/api/graph/recorder.hpp>
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
            [[nodiscard]] inline SushiRuntime::Result<void>
            recording_outcome(const SushiRuntime::Recorder& recorder)
            {
                if (const SushiRuntime::Error* refused = recorder.refusal())
                    return SushiRuntime::make_error(*refused);

                return {};
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
