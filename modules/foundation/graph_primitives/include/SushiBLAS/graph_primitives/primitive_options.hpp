/**************************************************************************/
/* primitive_options.hpp                                                  */
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
 * @file primitive_options.hpp
 * @brief Declares PrimitiveOptions, what a caller states about the nodes one primitive records.
 * @author Mustafa Garip
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Holds the name and the determinism class every node of one primitive carries.
         *
         * A primitive that records one node reports under name; one that records several
         * reports under name.stage, with the stage names its section of the README lists.
         * @see modules/foundation/graph_primitives/README.md
         */
        struct PrimitiveOptions
        {
            /** @brief The label the nodes report under; it must not be null or empty. */
            std::string_view name;

            /** @brief The contract each node carries to the runtime's plan. */
            SushiRuntime::Determinism determinism = SushiRuntime::Determinism::Unstated;
        };

        namespace Detail
        {
            /**
             * @brief Checks that a stated PrimitiveOptions names its nodes and holds a real class.
             * @param primitive Name the message opens with.
             * @param options   The trailing argument of the call; none means an unnamed call.
             * @return Nothing, or an InvalidGraph error naming the operand 'options'.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            require_options(const char* primitive,
                            const std::optional<PrimitiveOptions>& options)
            {
                if (!options)
                    return {};

                if (options->name.empty())
                    return invalid_graph(std::string(primitive) +
                                         ": the name of 'options' is null or empty.");

                switch (options->determinism)
                {
                    case SushiRuntime::Determinism::Unstated:
                    case SushiRuntime::Determinism::Bitwise:
                    case SushiRuntime::Determinism::Tolerant:
                    case SushiRuntime::Determinism::Cosmetic:
                        return {};
                }
                return invalid_graph(std::string(primitive) +
                                     ": the determinism of 'options' is not a class of the "
                                     "runtime.");
            }
        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
