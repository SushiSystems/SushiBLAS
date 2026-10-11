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
 * @brief Declares the throw a primitive reports a broken recording precondition with.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <utility>

#include <SushiRuntime/core/error.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Throws the runtime's exception with Errc::invalid_graph.
             * @param message Names the primitive, the argument at fault and the values involved.
             * @throws SushiRuntime::Core::SushiException Always.
             * @see modules/foundation/graph_primitives/README.md
             */
            [[noreturn]] inline void throw_invalid_graph(std::string message)
            {
                throw SushiRuntime::Core::SushiException(SushiRuntime::Core::Error{
                    SushiRuntime::Core::Errc::invalid_graph, std::move(message)});
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
