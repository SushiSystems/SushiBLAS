/**************************************************************************/
/* node_stamper.hpp                                                       */
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
 * @file node_stamper.hpp
 * @brief Declares NodeStamper, which builds the NodeOptions of each node a primitive records.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/report/dispatch_width.hpp>

#include <SushiBLAS/graph_primitives/primitive_options.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Builds the NodeOptions of the nodes of one call from its PrimitiveOptions.
             *
             * A call that stated no options gets a plain NodeOptions with the width and nothing
             * else, so an unnamed call records what it recorded before the options existed.
             * @see modules/foundation/graph_primitives/README.md
             */
            class NodeStamper
            {
                public:
                    /** @brief Takes the options of a call; they must outlive this stamper. */
                    explicit NodeStamper(const std::optional<PrimitiveOptions>& options) noexcept
                        : options_(options)
                    {
                    }

                    /**
                     * @brief Returns the options of the one node a primitive records.
                     * @param width Launch width stamped into the node's event records.
                     * @return Options named name; the name is valid until the next call.
                     */
                    [[nodiscard]] SushiRuntime::NodeOptions
                    node(std::size_t width = SushiRuntime::K_WIDTH_UNKNOWN)
                    {
                        return stamp(std::string_view{}, width);
                    }

                    /**
                     * @brief Returns the options of one of several nodes a primitive records.
                     * @param stage_name Stage name the label ends with.
                     * @param width      Launch width stamped into the node's event records.
                     * @return Options named name.stage; the name is valid until the next call.
                     */
                    [[nodiscard]] SushiRuntime::NodeOptions
                    stage(std::string_view stage_name,
                          std::size_t width = SushiRuntime::K_WIDTH_UNKNOWN)
                    {
                        return stamp(stage_name, width);
                    }

                private:
                    /** @brief Builds the options of one node; an empty stage adds no suffix. */
                    SushiRuntime::NodeOptions stamp(std::string_view stage, std::size_t width)
                    {
                        SushiRuntime::NodeOptions node_options;
                        node_options.width = width;
                        if (!options_)
                            return node_options;

                        label_.assign(options_->name);
                        if (!stage.empty())
                        {
                            label_ += '.';
                            label_ += stage;
                        }
                        node_options.name = label_;
                        node_options.determinism = options_->determinism;
                        return node_options;
                    }

                    /// @brief The options of the call.
                    const std::optional<PrimitiveOptions>& options_;

                    /// @brief The label the last returned options views.
                    std::string label_;
            };
        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
